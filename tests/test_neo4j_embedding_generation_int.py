"""Parallel embedding rebuilds preserve canonical episodes and facts in disposable Neo4j."""

import asyncio

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import EmbeddingGenerationChangedError
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, CommunityNode, EpisodicNode, EpisodeType
from graphiti_core.search.search_filters import SearchFilters
from test_neo4j_ingest_recovery_int import sandbox_neo4j_driver

pytestmark = pytest.mark.integration

# canonical fixture writes use public Graphiti namespaces; no application writes directly to the database
@pytest_asyncio.fixture
async def scenario(sandbox_neo4j_driver):

    driver = sandbox_neo4j_driver
    old = Mock(spec = EmbedderClient)
    old.create = AsyncMock(return_value = [1.0, 0.0])
    old.create_batch = AsyncMock(side_effect = lambda texts: [[1.0, 0.0] for text in texts])
    model = Mock(spec = LLMClient)
    model.generate_response = AsyncMock(side_effect = AssertionError('Extraction must not run'))
    graph = Graphiti(graph_driver = driver, llm_client = model, embedder = old,
                     cross_encoder = Mock(spec = CrossEncoderClient))
    group = f'embedding_{uuid4().hex}'
    nodes = [EntityNode(name = name, group_id = group, summary = 'Preserve source summary') for name in ('A', 'B')]
    for node in nodes:
        await graph.nodes.entity.save(node)
    edge = EntityEdge(group_id = group, name = 'RELATED_TO', source_node_uuid = nodes[0].uuid,
                     target_node_uuid = nodes[1].uuid, fact = 'A owns B',
                     created_at = datetime(2026, 1, 1, tzinfo = timezone.utc), valid_at = datetime(2026, 1, 1, tzinfo = timezone.utc))
    await graph.edges.entity.save(edge)
    community = CommunityNode(name = 'Original\ncommunity', group_id = group, summary = 'Original community summary')
    await graph.nodes.community.save(community)
    episode = EpisodicNode(name = 'Source episode', group_id = group, source = EpisodeType.text,
        source_description = 'Original report', content = 'A owns B', valid_at = datetime(2026, 1, 1, tzinfo = timezone.utc))
    await graph.nodes.episode.save(episode)
    target = Mock(spec = EmbedderClient)
    target.create_batch = AsyncMock(side_effect = lambda texts: [[0.1, 0.2, 0.3] for text in texts])
    state = SimpleNamespace(graph = graph, driver = driver, group = group, nodes = nodes, edge = edge,
                            target = target, identifier = str(uuid4()), model = model)
    try:
        yield state
    finally:
        await graph.clear()
        await graph.close()

# snapshots cover canonical properties and legacy vectors, not just entity/relationship counts
async def snapshot(scenario):

    nodes, _, _ = await scenario.driver.execute_query(
        'MATCH (n) WHERE n.group_id = $group AND NOT n:GraphEmbedding '
        'RETURN labels(n) AS labels, properties(n) AS properties ORDER BY n.uuid', group=scenario.group,
    ) # execute_query
    edges, _, _ = await scenario.driver.execute_query(
        'MATCH ()-[r:RELATES_TO]->() WHERE r.group_id = $group '
        'RETURN properties(r) AS properties ORDER BY r.uuid', group=scenario.group,
    ) # execute_query
    return [dict(row) for row in nodes], [dict(row) for row in edges]

# production consumers call only this public Graphiti rebuild seam
async def rebuild(scenario, **changes):

    arguments = {'group_ids': [scenario.group], 'batch_size': 1, 'embedder': scenario.target}
    arguments.update(changes)
    return await scenario.graph.rebuild_embedding_generation(scenario.identifier, 'a' * 64, 3, **arguments)

@pytest.mark.asyncio
async def test_neo4j_generation_preserves_source_and_legacy_vectors(scenario):

    before = await snapshot(scenario)
    result = await rebuild(scenario)
    assert result['counts'] == {'entity': 2, 'edge': 1, 'community': 1}
    assert result['points'] == result['rebuilt'] == 4 and result['verified_for_cutover'] is False
    assert await snapshot(scenario) == before
    legacy_results = await scenario.driver.search_ops.node_similarity_search(
        scenario.driver, [1.0, 0.0], SearchFilters(), group_ids = [scenario.group], limit = 10, min_score = 0.1,
    ) # node_similarity_search
    assert {node.uuid for node in legacy_results} == {node.uuid for node in scenario.nodes}
    texts = [text for call in scenario.target.create_batch.await_args_list for text in call.args[0]]
    assert set(texts) == {'A', 'B', 'A owns B', 'Original community'}
    scenario.target.create_batch.reset_mock()
    assert (await rebuild(scenario))['reused'] == 4
    scenario.target.create_batch.assert_not_awaited()
    scenario.model.generate_response.assert_not_awaited()

@pytest.mark.asyncio
async def test_neo4j_generation_reconciles_changes_and_deletion(scenario):

    await rebuild(scenario)
    scenario.edge.fact = 'A now owns a different share of B'
    await scenario.graph.edges.entity.save(scenario.edge)
    scenario.target.create_batch.reset_mock()
    result = await rebuild(scenario)
    assert result['rebuilt'] == 1 and result['reused'] == 3
    await scenario.graph.nodes.entity.delete(scenario.nodes[1])
    result = await rebuild(scenario)
    assert result['points'] == 2 and result['removed'] == 2

@pytest.mark.asyncio
async def test_neo4j_generation_interrupted_batch_resumes(scenario):

    calls = []
    async def embed(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise ConnectionError('Provider temporarily unavailable')
        return [[0.1, 0.2, 0.3] for text in texts]
    scenario.target.create_batch.side_effect = embed
    with pytest.raises(ConnectionError):
        await rebuild(scenario)
    scenario.target.create_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for text in texts]
    result = await rebuild(scenario)
    assert result['reused'] == 1 and result['rebuilt'] == 3

@pytest.mark.asyncio
async def test_neo4j_generation_lost_write_reply_resumes_without_duplicate_inference(scenario, monkeypatch):

    original = scenario.driver.execute_query
    lost = False
    async def uncertain(query, **values):
        nonlocal lost
        result = await original(query, **values)
        if 'MERGE (v:GraphEmbedding' in query and not lost:
            lost = True
            raise ConnectionError('Write committed but reply was lost')
        return result
    monkeypatch.setattr(scenario.driver, 'execute_query', uncertain)
    with pytest.raises(ConnectionError):
        await rebuild(scenario)
    scenario.target.create_batch.reset_mock()
    result = await rebuild(scenario)
    assert result['reused'] == 1 and result['rebuilt'] == 3
    assert scenario.target.create_batch.await_count == 3

@pytest.mark.asyncio
async def test_neo4j_generation_changed_source_is_not_certified(scenario):

    changed = False
    async def embed(texts):
        nonlocal changed
        if not changed:
            changed = True
            for node in scenario.nodes:
                node.name = f'{node.name} changed'
                await scenario.graph.nodes.entity.save(node)
        return [[0.1, 0.2, 0.3] for text in texts]
    scenario.target.create_batch.side_effect = embed
    with pytest.raises(EmbeddingGenerationChangedError):
        await rebuild(scenario)
    scenario.target.create_batch.side_effect = lambda texts: [[0.1, 0.2, 0.3] for text in texts]
    assert (await rebuild(scenario))['points'] == 4

@pytest.mark.asyncio
async def test_neo4j_generation_rejects_foreign_owner_and_empty_scope(scenario):

    await rebuild(scenario)
    with pytest.raises(ValueError, match = 'another model'):
        await scenario.graph.rebuild_embedding_generation(scenario.identifier, 'b' * 64, 3,
            group_ids = [scenario.group], embedder = scenario.target)
    with pytest.raises(ValueError, match = 'another model'):
        await scenario.graph.rebuild_embedding_generation(scenario.identifier, 'a' * 64, 4,
            group_ids = [scenario.group], embedder = scenario.target)
    empty = await scenario.graph.rebuild_embedding_generation(str(uuid4()), 'a' * 64, 3,
        group_ids = [], embedder = scenario.target)
    assert empty['points'] == 0

@pytest.mark.asyncio
async def test_neo4j_generation_late_old_worker_cannot_overwrite_new_target(scenario):

    first = await rebuild(scenario)
    old = scenario.identifier
    scenario.identifier = str(uuid4())
    second = await rebuild(scenario)
    scenario.identifier = old
    await rebuild(scenario)
    records, _, _ = await scenario.driver.execute_query(
        'MATCH (v:GraphEmbedding) WHERE v.generation IN $ids RETURN v.generation AS generation, count(v) AS total',
        ids=[first['generation_id'], second['generation_id']],
    ) # execute_query
    assert {row['generation']: row['total'] for row in records} == {old: 4, second['generation_id']: 4}

@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [[True, 0, 1], [float('nan'), 0, 1], [0, 0, 0]])
async def test_neo4j_generation_invalid_batch_cannot_write(scenario, bad):

    before = await snapshot(scenario)
    scenario.target.create_batch.side_effect = lambda texts: [bad for text in texts]
    with pytest.raises(ValueError):
        await rebuild(scenario, batch_size = 2)
    saved, _, _ = await scenario.driver.execute_query(
        'MATCH (v:GraphEmbedding {generation: $generation}) RETURN count(v) AS total',
        generation=scenario.identifier,
    ) # execute_query
    assert saved[0]['total'] == 0
    assert await snapshot(scenario) == before

@pytest.mark.asyncio
async def test_neo4j_generation_missing_source_text_requires_review(scenario):

    scenario.nodes[0].name = ''
    await scenario.graph.nodes.entity.save(scenario.nodes[0])
    with pytest.raises(ValueError, match = 'complete identity'):
        await rebuild(scenario)

@pytest.mark.asyncio
async def test_neo4j_generation_cancellation_stops_before_vector_write(scenario):

    calls = []
    async def checkpoint():
        calls.append(True)
        if len(calls) == 3:
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await rebuild(scenario, checkpoint = checkpoint)
    saved, _, _ = await scenario.driver.execute_query(
        'MATCH (v:GraphEmbedding {generation: $generation}) RETURN count(v) AS total',
        generation=scenario.identifier,
    ) # execute_query
    assert saved[0]['total'] == 0

# public inventory covers canonical sources rather than generated vector nodes and advances across real database pages
@pytest.mark.asyncio
async def test_neo4j_group_inventory_survives_shadow_generation(scenario):

    before = await scenario.graph.get_embedding_group_ids(limit = 1)
    assert before == [scenario.group]
    assert await scenario.graph.get_embedding_group_ids(after = scenario.group, limit = 1) == []
    await rebuild(scenario)
    assert await scenario.graph.get_embedding_group_ids(limit = 1) == before

# namespaces supported only by episodes remain visible even when they have no vector-bearing source
@pytest.mark.asyncio
async def test_neo4j_group_inventory_includes_episode_only_namespace(scenario):

    group = f'z_{uuid4().hex}'
    episode = EpisodicNode(name = 'Episode only', group_id = group, source = EpisodeType.text,
        source_description = 'Source', content = 'No extracted graph',
        valid_at = datetime(2026, 1, 1, tzinfo = timezone.utc))
    await scenario.graph.nodes.episode.save(episode)
    first = await scenario.graph.get_embedding_group_ids(limit = 1)
    second = await scenario.graph.get_embedding_group_ids(after = first[-1], limit = 1)
    assert first + second == sorted([scenario.group, group])
    assert await scenario.graph.get_embedding_group_ids(after = second[-1], limit = 1) == []

# a Saga-only namespace is occupied even though it does not appear in the canonical embedding-source inventory
@pytest.mark.asyncio
async def test_neo4j_empty_namespace_includes_auxiliary_nodes(scenario):

    from graphiti_core.nodes import SagaNode
    group = f'group_{uuid4().hex}'
    assert await scenario.graph.is_group_namespace_empty(group) is True
    assert await scenario.graph.is_group_namespace_empty(scenario.group) is False
    saga = SagaNode(name = str(uuid4()), group_id = group, created_at = datetime.now(timezone.utc))
    await scenario.graph.nodes.saga.save(saga)
    assert await scenario.graph.is_group_namespace_empty(group) is False

# public provenance pages enumerate canonical source records exactly once, excluding shadow-generation vectors
@pytest.mark.asyncio
async def test_neo4j_ownership_inventory_pages(scenario):

    before, cursor = [], None
    while True:
        page = await scenario.graph.get_group_ownership_page(scenario.group, after = cursor, limit = 2)
        if not page:
            break
        before.extend(page)
        cursor = page[-1]['key']
    assert len(before) == 5 and len({row['key'] for row in before}) == 5
    assert {row['kind'] for row in before} == {'node', 'edge'}
    assert all(row['group_id'] == scenario.group for row in before)
    assert all('embedding' not in row and 'content' not in row and 'fact' not in row for row in before)
    await rebuild(scenario)
    after, cursor = [], None
    while True:
        page = await scenario.graph.get_group_ownership_page(scenario.group, after = cursor, limit = 2)
        if not page:
            break
        after.extend(page)
        cursor = page[-1]['key']
    assert after == before

# links incident to a namespace are included even when their own group metadata belongs elsewhere
@pytest.mark.asyncio
async def test_neo4j_ownership_inventory_exposes_cross_group_links(scenario):

    foreign = EntityNode(name = 'Foreign', group_id = f'foreign_{uuid4().hex}')
    await scenario.graph.nodes.entity.save(foreign)
    edge = EntityEdge(group_id = foreign.group_id, name = 'RELATED_TO', source_node_uuid = scenario.nodes[0].uuid,
        target_node_uuid = foreign.uuid, fact = 'Must not be adopted', created_at = datetime.now(timezone.utc))
    await scenario.graph.edges.entity.save(edge)
    rows = await scenario.graph.get_group_ownership_page(scenario.group)
    found = next(row for row in rows if row['uuid'] == edge.uuid)
    assert found['group_id'] != scenario.group and found['target_group'] != scenario.group
