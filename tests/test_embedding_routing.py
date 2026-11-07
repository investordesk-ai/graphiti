"""Task-local generation routing validates model geometry without affecting existing graph clients."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.embedding_routing import GraphEmbeddingRoute, graph_embedding_route
from graphiti_core.embedding_routing import current_graph_embedding_route, generation_group_ids
from graphiti_core.embedding_routing import validate_generation_vector

# reject malformed route metadata before it can name a stored generation or select a model
@pytest.mark.parametrize('changes', [
    {'generation_id': 'not-a-uuid'}, {'fingerprint': 'x' * 64}, {'fingerprint': 'a' * 63},
    {'dimensions': 0}, {'dimensions': True}, {'group_ids': ('same', 'same')}, {'group_ids': ('',)},
])
def test_invalid_graph_route_is_rejected(changes):

    definition = {'generation_id': str(uuid4()), 'fingerprint': 'a' * 64, 'dimensions': 3, 'group_ids': ('group',)}
    with pytest.raises((ValueError, TypeError)):
        GraphEmbeddingRoute(**{**definition, **changes})

# no route means the existing path, while a nested route always restores its parent after failure
@pytest.mark.asyncio
async def test_graph_route_nested_cancelled_and_concurrent_tasks():

    driver = SimpleNamespace(provider = GraphProvider.NEO4J)
    other = SimpleNamespace(provider = GraphProvider.NEO4J)
    first = GraphEmbeddingRoute(str(uuid4()), 'a' * 64, 3, ('first',))
    second = GraphEmbeddingRoute(str(uuid4()), 'b' * 64, 4, ('second',))
    ready, release = asyncio.Event(), asyncio.Event()
    async def concurrent():
        assert current_graph_embedding_route(driver) is None
        with graph_embedding_route(driver, second):
            ready.set()
            await release.wait()
            assert current_graph_embedding_route(driver) == second
        assert current_graph_embedding_route(driver) is None
    task = asyncio.create_task(concurrent())
    await ready.wait()
    with graph_embedding_route(driver, first):
        assert current_graph_embedding_route(other) is None
        with pytest.raises(asyncio.CancelledError):
            with graph_embedding_route(driver, second):
                assert current_graph_embedding_route(driver) == second
                raise asyncio.CancelledError()
        assert current_graph_embedding_route(driver) == first
        release.set()
        await task
        assert current_graph_embedding_route(driver) == first
    assert current_graph_embedding_route(driver) is None

# inherited graph namespaces can narrow but cannot widen the generation's verified group set
@pytest.mark.parametrize('groups,expected', [(None, ['one', 'two']), ([], []), (['two'], ['two'])])
def test_generation_namespace_narrowing(groups, expected):

    driver = SimpleNamespace(provider = GraphProvider.NEO4J)
    route = GraphEmbeddingRoute(str(uuid4()), 'a' * 64, 3, ('one', 'two'))
    with graph_embedding_route(driver, route):
        assert generation_group_ids(driver, groups) == expected
        with pytest.raises(ValueError):
            generation_group_ids(driver, ['other-tenant'])

# geometry validation applies to every alternate query, candidate and loaded MMR vector
@pytest.mark.parametrize('vector', [[], [1, 0], [True, 0, 1], [0, 0, 0], [1, '2', 3],
                                     [float('nan'), 0, 1], [float('inf'), 1, 0], [10 ** 999, 0, 1]])
def test_generation_rejects_malformed_vectors(vector):

    driver = SimpleNamespace(provider = GraphProvider.NEO4J)
    route = GraphEmbeddingRoute(str(uuid4()), 'a' * 64, 3, None)
    with graph_embedding_route(driver, route):
        with pytest.raises(ValueError):
            validate_generation_vector(driver, vector)
        valid = validate_generation_vector(driver, [0.1, 0.2, 0.3])
        assert len(valid) == 3 and sum(value * value for value in valid) == pytest.approx(1.0)
