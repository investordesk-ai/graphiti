"""Neo4j writes remain gated by validated model decisions and safe under episode replay."""

import json
import os

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.errors import LLMResponseError
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.maintenance import edge_operations, node_operations

# construct the disposable driver before the async test so schema initialisation runs only once
@pytest.fixture
def sandbox_neo4j_driver():

    uri = os.getenv('NEO4J_RECOVERY_TEST_URI')
    if not uri:
        pytest.skip('Disposable Neo4j is not configured')
    assert uri == 'bolt://graphdb:7687', 'This test only accepts the disposable Neo4j endpoint'
    return Neo4jDriver(uri, 'neo4j', 'sandbox-password')

# unresolved model output must not write an episode; replay must retain the original graph identities
@pytest.mark.asyncio
async def test_invalid_output_preserves_neo4j_and_replay_preserves_canonical_fact(monkeypatch, sandbox_neo4j_driver):

    driver = sandbox_neo4j_driver
    model = Mock(spec = LLMClient)
    embedder = Mock(spec = EmbedderClient)
    embedder.create = AsyncMock(return_value = [1.0] + [0.0] * 1023)
    embedder.create_batch = AsyncMock(side_effect = lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
    graph = Graphiti(graph_driver = driver, llm_client = model, embedder = embedder,
                     cross_encoder = Mock(spec = CrossEncoderClient))
    group = f'recovery_{uuid4().hex}'
    episode_id = str(uuid4())
    source_date = datetime(2026, 3, 31, tzinfo = timezone.utc)
    canonical = EntityNode(name = 'Canonical A', group_id = group, name_embedding = [1.0] + [0.0] * 1023)
    target = EntityNode(name = 'B', group_id = group, name_embedding = [1.0] + [0.0] * 1023)
    incoming = EntityNode(name = 'A', group_id = group)
    existing = EntityEdge(group_id = group, name = 'RELATED_TO', source_node_uuid = canonical.uuid,
                          target_node_uuid = target.uuid, fact = 'A has a relationship with B',
                          created_at = source_date, valid_at = source_date,
                          fact_embedding = [1.0] + [0.0] * 1023)
    extracted = existing.model_copy(update = {'uuid': str(uuid4()), 'source_node_uuid': incoming.uuid,
                                             'fact': 'A is associated with B'})
    rejected = True

    async def candidates(clients, nodes):
        result = []
        for node in nodes:
            identifier = canonical.uuid if node.name == 'A' else target.uuid
            result.append([await EntityNode.get_by_uuid(driver, identifier)])
        return result

    async def search(clients, fact, **kwargs):
        edges = await EntityEdge.get_between_nodes(driver, canonical.uuid, target.uuid)
        return Mock(edges = edges if kwargs['search_filter'].edge_uuids else [])

    async def generate(messages, **kwargs):
        prompt = kwargs['prompt_name']
        if prompt in ('dedupe_nodes.nodes', 'dedupe_nodes.nodes_repair'):
            candidate_id = 999 if prompt == 'dedupe_nodes.nodes' else 0
            return {'entity_resolutions': [{'id': 0, 'name': 'A', 'duplicate_candidate_id': candidate_id}]}
        if prompt.startswith('dedupe_edges.resolve_edge'):
            return {'duplicate_facts': [999 if rejected else 0], 'contradicted_existing_facts': [],
                    'contradicted_invalidation_candidates': []}
        if prompt == 'extract_nodes.extract_summaries_batch':
            text = messages[-1].content.split('<ENTITIES>', 1)[1].split('</ENTITIES>', 1)[0]
            return {'summaries': [{'entity_id': entity['entity_id'], 'summary': 'A is associated with B'}
                                  for entity in json.loads(text)]}
        raise AssertionError(f'Unexpected inference request: {prompt}')

    model.generate_response = AsyncMock(side_effect = generate)
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', candidates)
    monkeypatch.setattr(edge_operations, 'search', search)
    arguments = {
        'name': 'Original source', 'episode_body': 'A is Canonical A. A is associated with B.',
        'source_description': 'Original source', 'reference_time': source_date,
        'source': EpisodeType.text, 'group_id': group, 'uuid': episode_id,
        'pre_extracted_nodes': [incoming, target], 'pre_extracted_edges': [extracted],
        'previous_episode_uuids': [], 'timestamps_prepared': True,
    } # arguments

    try:
        await graph.build_indices_and_constraints()
        await canonical.save(driver)
        await target.save(driver)
        await existing.save(driver)

        with pytest.raises(LLMResponseError) as caught:
            await graph.add_episode(**arguments)
        assert caught.value.recovery_action == 'review'
        assert caught.value.diagnostics['edge_id'] == extracted.uuid
        with pytest.raises(NodeNotFoundError):
            await EpisodicNode.get_by_uuid(driver, episode_id)
        stored = await EntityEdge.get_by_uuid(driver, existing.uuid)
        assert stored.episodes == [] and stored.invalid_at is None

        # the same source and episode ID can resume after correction without creating another identity
        rejected = False
        mention_snapshot = None
        for _ in range(2):
            result = await graph.add_episode(**arguments)
            assert result.episode.uuid == episode_id
            assert result.edges[0].uuid == existing.uuid
            mentions, _, _ = await driver.execute_query(
                'MATCH (:Episodic {uuid: $identifier})-[mention:MENTIONS]->(entity:Entity) '
                'RETURN mention.uuid AS uuid, mention.created_at AS created_at, entity.uuid AS entity_uuid '
                'ORDER BY entity_uuid', identifier = episode_id, routing_ = 'r',
            ) # execute_query
            snapshot = [dict(record) for record in mentions]
            assert len(snapshot) == 2
            assert {edge.uuid for edge in result.episodic_edges} == {row['uuid'] for row in snapshot}
            if mention_snapshot is not None:
                assert snapshot == mention_snapshot
            mention_snapshot = snapshot
        stored = await EntityEdge.get_by_uuid(driver, existing.uuid)
        assert stored.episodes == [episode_id]
        assert stored.valid_at == source_date
        with pytest.raises(NodeNotFoundError):
            await EntityNode.get_by_uuid(driver, incoming.uuid)
        records, _, _ = await driver.execute_query(
            'MATCH (episode:Episodic {uuid: $identifier}) RETURN count(episode) AS count',
            identifier = episode_id, routing_ = 'r',
        ) # execute_query
        assert records[0]['count'] == 1
        records, _, _ = await driver.execute_query(
            'MATCH (:Episodic {uuid: $identifier})-[:MENTIONS]->(entity:Entity) '
            'RETURN count(DISTINCT entity) AS count', identifier = episode_id, routing_ = 'r',
        ) # execute_query
        assert records[0]['count'] == 2
    finally:
        await graph.close()
