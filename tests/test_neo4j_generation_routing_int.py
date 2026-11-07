"""Exercise real Neo4j reads/writes across independently sized Graphiti embedding generations."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from graphiti_core.embedding_routing import current_graph_embedding_route, generation_embeddings
from graphiti_core.errors import EmbeddingGenerationChangedError
from graphiti_core.search import search_utils
from graphiti_core.search.search_config import SearchConfig, NodeSearchConfig, NodeSearchMethod, NodeReranker
from graphiti_core.search.search_config import EdgeSearchConfig, EdgeSearchMethod, EdgeReranker
from graphiti_core.search.search_config import CommunitySearchConfig, CommunitySearchMethod, CommunityReranker
from graphiti_core.search.search_filters import SearchFilters
from test_neo4j_embedding_generation_int import scenario, rebuild
from test_neo4j_ingest_recovery_int import sandbox_neo4j_driver

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# the actual high-level search and both internal search paths must agree on selected source vectors
@pytest.mark.parametrize('reranker', ['rrf', 'mmr'])
async def test_selected_generation_search_and_mmr(scenario, reranker):

    await rebuild(scenario)
    scenario.target.create = AsyncMock(return_value = [0.1, 0.2, 0.3])
    config = SearchConfig(
        reranker_min_score = -1,
        node_config = NodeSearchConfig(search_methods = [NodeSearchMethod.cosine_similarity],
                                        reranker = NodeReranker[reranker]),
        edge_config = EdgeSearchConfig(search_methods = [EdgeSearchMethod.cosine_similarity],
                                        reranker = EdgeReranker[reranker]),
        community_config = CommunitySearchConfig(search_methods = [CommunitySearchMethod.cosine_similarity],
                                                reranker = CommunityReranker[reranker]),
    ) # config
    original = scenario.graph.embedder
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target) as selected:
        result = await selected.search_('Find source', config, group_ids = [scenario.group])
        assert {node.uuid for node in result.nodes} == {node.uuid for node in scenario.nodes}
        assert [edge.uuid for edge in result.edges] == [scenario.edge.uuid]
        assert len(result.communities) == 1
        await selected.nodes.entity.load_embeddings_bulk(scenario.nodes)
        await selected.edges.entity.load_embeddings_bulk([scenario.edge])
        await scenario.nodes[0].load_name_embedding(scenario.driver)
        await scenario.edge.load_fact_embedding(scenario.driver)
        await result.communities[0].load_name_embedding(scenario.driver)
        await scenario.driver.community_node_ops.load_name_embedding(scenario.driver, result.communities[0])
        assert all(len(node.name_embedding) == 3 for node in scenario.nodes)
        assert len(scenario.edge.fact_embedding) == len(result.communities[0].name_embedding) == 3
        with pytest.raises(ValueError):
            await selected.search_('Wrong driver', config, group_ids = [scenario.group], driver = object())
        assert scenario.graph.embedder is original
        vectors = await search_utils.get_embeddings_for_nodes(scenario.driver, scenario.nodes)
        assert all(len(vector) == 3 for vector in vectors.values())
    assert current_graph_embedding_route(scenario.driver) is None
    assert scenario.graph.embedder is original
    with pytest.raises(ValueError):
        await selected.embedder.create('An escaped generation client')
    # no generation selection retains the old model and the old canonical-vector search path
    nodes = await search_utils.node_similarity_search(scenario.driver, [1.0, 0.0], SearchFilters(),
                                                       group_ids = [scenario.group], min_score = 0.1)
    assert {node.uuid for node in nodes} == {node.uuid for node in scenario.nodes}

# response loss/cancellation restores the task route and never silently falls back to legacy geometry
@pytest.mark.parametrize('fault', ['provider', 'cancelled', 'bad-vector'])
async def test_generation_failure_restores_route(scenario, fault):

    await rebuild(scenario)
    error = asyncio.CancelledError() if fault == 'cancelled' else RuntimeError('Provider unavailable')
    scenario.target.create = AsyncMock(return_value = [1.0, 0.0] if fault == 'bad-vector' else None,
                                      side_effect = None if fault == 'bad-vector' else error)
    with pytest.raises(ValueError if fault == 'bad-vector' else type(error)):
        async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                        embedder = scenario.target) as selected:
            await selected.embedder.create('Query')
    assert current_graph_embedding_route(scenario.driver) is None

# changed/deleted canonical sources cannot be paired with an old vector or loaded by MMR
async def test_generation_excludes_stale_sources_and_rebuild_repairs_them(scenario):

    await rebuild(scenario)
    scenario.edge.fact = 'A has changed its source assertion about B'
    await scenario.graph.edges.entity.save(scenario.edge)
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target):
        assert await search_utils.edge_similarity_search(scenario.driver, [0.1, 0.2, 0.3], None, None,
            SearchFilters(), group_ids = [scenario.group], min_score = 0.1) == []
        with pytest.raises(EmbeddingGenerationChangedError):
            await generation_embeddings(scenario.driver, 'edge', [scenario.edge.uuid])
    await rebuild(scenario)
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target):
        result = await search_utils.edge_similarity_search(scenario.driver, [0.1, 0.2, 0.3], None, None,
            SearchFilters(), group_ids = [scenario.group], min_score = 0.1)
        assert [edge.fact for edge in result] == [scenario.edge.fact]

# a successful write refreshes the chosen generation before the caller can acknowledge its episode operation
async def test_generation_write_reconciles_before_return(scenario):

    await rebuild(scenario)
    scenario.target.create = AsyncMock(return_value = [0.1, 0.2, 0.3])
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target, write = True) as selected:
        scenario.nodes[0].name = 'A new canonical source name'
        await selected.nodes.entity.save(scenario.nodes[0])
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target):
        values = await generation_embeddings(scenario.driver, 'entity', [scenario.nodes[0].uuid])
        assert len(values[scenario.nodes[0].uuid]) == 3

# malformed ownership or missing implementation metadata never creates a generation or invokes a model
@pytest.mark.parametrize('fault', ['missing', 'fingerprint', 'dimensions', 'groups'])
async def test_generation_owner_checks_precede_model_calls(scenario, fault):

    await rebuild(scenario)
    arguments = {'generation_id': scenario.identifier, 'fingerprint': 'a' * 64, 'dimensions': 3,
                 'embedder': scenario.target}
    arguments.update({'missing': {'generation_id': str(uuid4())}, 'fingerprint': {'fingerprint': 'b' * 64},
                      'dimensions': {'dimensions': 4}, 'groups': {'group_ids': ['another-tenant']}}[fault])
    scenario.target.create_batch.reset_mock()
    with pytest.raises(ValueError):
        async with scenario.graph.embedding_generation(**arguments):
            pytest.fail('Invalid ownership must not enter the operation')
    scenario.target.create_batch.assert_not_awaited()

# a tenant generation cannot widen scope by omitting its group list, even on direct lower-level search calls
async def test_generation_defaults_to_its_registered_groups(scenario):

    await rebuild(scenario)
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                    embedder = scenario.target):
        nodes = await search_utils.node_similarity_search(scenario.driver, [0.1, 0.2, 0.3], SearchFilters(),
                                                           min_score = 0.1)
        assert {node.uuid for node in nodes} == {node.uuid for node in scenario.nodes}
        with pytest.raises(ValueError):
            await search_utils.node_similarity_search(scenario.driver, [0.1, 0.2, 0.3], SearchFilters(),
                                                       group_ids = ['another-tenant'])

# concurrent calls on the same driver use independent generation IDs, dimensions and embedders
async def test_generation_tasks_are_isolated(scenario):

    await rebuild(scenario)
    ready, release = asyncio.Event(), asyncio.Event()
    async def selected_reader():
        async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
                                                        embedder = scenario.target):
            ready.set()
            await release.wait()
            result = await search_utils.node_similarity_search(scenario.driver, [0.1, 0.2, 0.3], SearchFilters(),
                                                               group_ids = [scenario.group], min_score = 0.1)
            return len(result)
    task = asyncio.create_task(selected_reader())
    await ready.wait()
    assert current_graph_embedding_route(scenario.driver) is None
    result = await search_utils.node_similarity_search(scenario.driver, [1.0, 0.0], SearchFilters(),
                                                       group_ids = [scenario.group], min_score = 0.1)
    release.set()
    assert len(result) == await task == 2

# a new empty namespace can receive its first episode after publication without rerunning a whole-corpus rebuild
async def test_write_admits_only_new_empty_namespace(scenario):

    from graphiti_core.nodes import EntityNode
    await rebuild(scenario)
    group = f'new_group_{uuid4().hex}'
    scenario.target.create = AsyncMock(return_value = [0.1, 0.2, 0.3])
    node = EntityNode(name = 'New company source', group_id = group)
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
        embedder = scenario.target, group_ids = [group], write = True) as selected: # async with
        await selected.nodes.entity.save(node)
    async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
        embedder = scenario.target, group_ids = [group]) as selected: # async with
        await selected.nodes.entity.load_embeddings_bulk([node])
        assert len(node.name_embedding) == 3
    scenario.model.generate_response.assert_not_awaited()

# populated namespaces cannot be moved into another model's generation by an ordinary write request
async def test_write_cannot_adopt_populated_namespace(scenario):

    from graphiti_core.nodes import EntityNode
    await rebuild(scenario)
    group = f'existing_group_{uuid4().hex}'
    await scenario.graph.nodes.entity.save(EntityNode(name = 'Existing other source', group_id = group))
    with pytest.raises(EmbeddingGenerationChangedError):
        async with scenario.graph.embedding_generation(scenario.identifier, 'a' * 64, 3,
            embedder = scenario.target, group_ids = [group], write = True): # async with
            pytest.fail('Populated namespaces require their own verified source rebuild')
