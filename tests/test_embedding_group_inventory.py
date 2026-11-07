"""Read-only graph namespace inventory validates its public bounded paging contract."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti

# no graph write, extraction model or embedding provider is needed to enumerate canonical namespaces
@pytest.mark.asyncio
async def test_group_inventory_is_read_only_and_bounded():

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = ([{'group_id': 'one'}, {'group_id': 'two'}], None, None)))
    assert await graph.get_embedding_group_ids(limit = 2) == ['one', 'two']
    call = graph.driver.execute_query.await_args
    assert call.kwargs == {'after': None, 'limit': 2, 'routing_': 'r'}
    assert 'RELATES_TO' in call.args[0] and 'Episodic' in call.args[0]
    assert not any(word in call.args[0] for word in ('CREATE', 'MERGE', 'DELETE', 'SET '))

@pytest.mark.asyncio
@pytest.mark.parametrize('page', [[None], [''], ['two', 'one'], ['one', 'one'], [42]])
async def test_group_inventory_rejects_unusable_namespaces(page):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = ([{'group_id': value} for value in page], None, None)))
    with pytest.raises(ValueError):
        await graph.get_embedding_group_ids()

@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [{'limit': 0}, {'limit': True}, {'limit': 129}, {'after': ''}, {'after': 42}])
async def test_group_inventory_validates_before_database(changes):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J, execute_query = AsyncMock())
    with pytest.raises(ValueError):
        await graph.get_embedding_group_ids(**changes)
    graph.driver.execute_query.assert_not_awaited()

@pytest.mark.asyncio
async def test_group_inventory_refuses_unsupported_backend():

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = 'unsupported', execute_query = AsyncMock())
    with pytest.raises(NotImplementedError):
        await graph.get_embedding_group_ids()
    graph.driver.execute_query.assert_not_awaited()

# empty checks inspect all node and relationship namespaces, including groups with no embedding-bearing source
@pytest.mark.asyncio
@pytest.mark.parametrize('empty', [True, False])
async def test_group_namespace_probe(empty):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = ([{'empty': empty}], None, None)))
    assert await graph.is_group_namespace_empty('group_example') is empty
    call = graph.driver.execute_query.await_args
    assert call.kwargs == {'group': 'group_example', 'routing_': 'r'}
    assert 'MATCH (n {group_id:' in call.args[0] and 'MATCH ()-[r {group_id:' in call.args[0]

@pytest.mark.asyncio
@pytest.mark.parametrize('group', ['', 'unsafe group', None, 'a' * 513, 42])
async def test_group_namespace_invalid(group):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J, execute_query = AsyncMock())
    with pytest.raises(ValueError):
        await graph.is_group_namespace_empty(group)
    graph.driver.execute_query.assert_not_awaited()

@pytest.mark.asyncio
@pytest.mark.parametrize('rows', [[], [{'empty': None}], [{'empty': 1}]])
async def test_group_namespace_receipt(rows):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = (rows, None, None)))
    with pytest.raises(ValueError):
        await graph.is_group_namespace_empty('group_example')

@pytest.mark.asyncio
async def test_group_namespace_backend():

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = 'unsupported', execute_query = AsyncMock())
    with pytest.raises(NotImplementedError):
        await graph.is_group_namespace_empty('group_example')
    graph.driver.execute_query.assert_not_awaited()

# source inventory contains identities and provenance only, with authoritative reads routed to the leader
@pytest.mark.asyncio
async def test_ownership_page_is_bounded_and_content_free():

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = ([{'key': 'n:one', 'labels': ['Saga']}], None, None)))
    assert (await graph.get_group_ownership_page('group', limit = 1))[0]['key'] == 'n:one'
    call = graph.driver.execute_query.await_args
    assert call.kwargs == {'group': 'group', 'after': None, 'limit': 1, 'routing_': 'w'}
    assert 'properties(' not in call.args[0] and '.content' not in call.args[0] and '.embedding' not in call.args[0]

@pytest.mark.asyncio
@pytest.mark.parametrize('arguments', [{'group_id': ''}, {'group_id': 'bad group'}, {'group_id': None},
    {'group_id': 'group', 'limit': True}, {'group_id': 'group', 'limit': 129}, {'group_id': 'group', 'after': ''}])
async def test_ownership_page_validates_before_read(arguments):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J, execute_query = AsyncMock())
    with pytest.raises(ValueError):
        await graph.get_group_ownership_page(**arguments)
    graph.driver.execute_query.assert_not_awaited()

@pytest.mark.asyncio
@pytest.mark.parametrize('rows', [[{'key': 'same'}, {'key': 'same'}], [{'key': None}], [{'key': 'z'}, {'key': 'a'}]])
async def test_ownership_page_rejects_invalid_keysets(rows):

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J,
        execute_query = AsyncMock(return_value = (rows, None, None)))
    with pytest.raises(ValueError):
        await graph.get_group_ownership_page('group')

@pytest.mark.asyncio
async def test_ownership_page_requires_neo4j():

    graph = object.__new__(Graphiti)
    graph.driver = SimpleNamespace(provider = 'unsupported', execute_query = AsyncMock())
    with pytest.raises(NotImplementedError):
        await graph.get_group_ownership_page('group')
    graph.driver.execute_query.assert_not_awaited()
