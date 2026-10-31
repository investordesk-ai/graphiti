"""Episode writes through Graphiti remain replayable after commit interruption and concurrent delivery."""

import asyncio
import json

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.utils.maintenance import node_operations
from test_neo4j_ingest_recovery_int import sandbox_neo4j_driver

pytestmark = pytest.mark.integration

# every data write in these scenarios goes through Graphiti.add_episode
@pytest.fixture
def episode_scenario(monkeypatch, sandbox_neo4j_driver):

    driver = sandbox_neo4j_driver
    model = Mock(spec = LLMClient)
    embedder = Mock(spec = EmbedderClient)
    embedder.create = AsyncMock(return_value = [1.0] + [0.0] * 1023)
    embedder.create_batch = AsyncMock(side_effect = lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
    graph = Graphiti(graph_driver = driver, llm_client = model, embedder = embedder,
                     cross_encoder = Mock(spec = CrossEncoderClient))
    group = f'replay_{uuid4().hex}'
    episode_id = str(uuid4())
    source_date = datetime(2026, 3, 31, tzinfo = timezone.utc)
    nodes = [EntityNode(name = name, group_id = group) for name in ('A', 'B')]

    async def generate(messages, **kwargs):
        assert kwargs['prompt_name'] == 'extract_nodes.extract_summaries_batch'
        text = messages[-1].content.split('<ENTITIES>', 1)[1].split('</ENTITIES>', 1)[0]
        return {'summaries': [{'entity_id': entity['entity_id'], 'summary': 'Supported source'}
                              for entity in json.loads(text)]}

    model.generate_response = AsyncMock(side_effect = generate)
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', AsyncMock(return_value = [[], []]))
    arguments = {
        'name': 'Original source', 'episode_body': 'A and B appear in this source.',
        'source_description': 'Original source', 'reference_time': source_date,
        'source': EpisodeType.text, 'group_id': group, 'uuid': episode_id,
        'pre_extracted_nodes': nodes, 'pre_extracted_edges': [],
        'previous_episode_uuids': [], 'timestamps_prepared': True,
    } # arguments
    return graph, driver, arguments

# concurrent redeliveries of an existing episode must reuse its stored episode/entity associations
@pytest.mark.asyncio
async def test_concurrent_stored_episode_replay_reuses_mentions(episode_scenario):

    graph, driver, arguments = episode_scenario
    try:
        await graph.build_indices_and_constraints()
        await graph.add_episode(**arguments)
        results = await asyncio.gather(*(graph.add_episode(**arguments) for _ in range(12)))
        records, _, _ = await driver.execute_query(
            'MATCH (:Episodic {uuid: $identifier})-[mention:MENTIONS]->(entity:Entity) '
            'RETURN mention.uuid AS uuid, entity.uuid AS entity_uuid',
            identifier = arguments['uuid'], routing_ = 'r',
        ) # execute_query
        identities, _, _ = await driver.execute_query(
            'MATCH (episode:Episodic {uuid: $identifier}) '
            'OPTIONAL MATCH (episode)-[:MENTIONS]->(entity:Entity) '
            'RETURN count(DISTINCT episode) AS episode_count, count(DISTINCT entity) AS entity_count',
            identifier = arguments['uuid'], routing_ = 'r',
        ) # execute_query
        assert len(records) == 2, [dict(record) for record in identities]
        assert identities[0]['episode_count'] == 1 and identities[0]['entity_count'] == 2
        stored = {record['uuid'] for record in records}
        assert all({edge.uuid for edge in result.episodic_edges} == stored for result in results)
        assert {record['entity_uuid'] for record in records} == {node.uuid for node in arguments['pre_extracted_nodes']}
    finally:
        await graph.close()

# a failure after episode persistence must not cause duplicate links when the episode is delivered again
@pytest.mark.asyncio
async def test_episode_replay_after_post_commit_interruption(monkeypatch, episode_scenario):

    graph, driver, arguments = episode_scenario
    original = graph._get_or_create_saga
    first = True
    async def interrupted(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise ConnectionError('Injected interruption after episode persistence')
        return await original(*args, **kwargs)
    monkeypatch.setattr(graph, '_get_or_create_saga', interrupted)
    arguments['saga'] = str(uuid4())
    try:
        await graph.build_indices_and_constraints()
        with pytest.raises(ConnectionError):
            await graph.add_episode(**arguments)
        before, _, _ = await driver.execute_query(
            'MATCH (:Episodic {uuid: $identifier})-[mention:MENTIONS]->(entity:Entity) '
            'RETURN mention.uuid AS uuid, mention.created_at AS created_at, entity.uuid AS entity_uuid '
            'ORDER BY entity_uuid', identifier = arguments['uuid'], routing_ = 'r',
        ) # execute_query
        assert len(before) == 2
        for _ in range(3):
            result = await graph.add_episode(**arguments)
            assert {edge.uuid for edge in result.episodic_edges} == {record['uuid'] for record in before}
        after, _, _ = await driver.execute_query(
            'MATCH (:Episodic {uuid: $identifier})-[mention:MENTIONS]->(entity:Entity) '
            'RETURN mention.uuid AS uuid, mention.created_at AS created_at, entity.uuid AS entity_uuid '
            'ORDER BY entity_uuid', identifier = arguments['uuid'], routing_ = 'r',
        ) # execute_query
        assert [dict(record) for record in after] == [dict(record) for record in before]
        links, _, _ = await driver.execute_query(
            'MATCH (:Saga {name: $artifact})-[link:HAS_EPISODE]->(:Episodic {uuid: $identifier}) '
            'RETURN count(link) AS count', artifact = arguments['saga'], identifier = arguments['uuid'], routing_ = 'r',
        ) # execute_query
        assert links[0]['count'] == 1
    finally:
        await graph.close()
