"""Neo4j mention persistence reuses stored identities through the existing Graphiti operations."""

import asyncio

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.driver.neo4j.operations.episodic_edge_ops import Neo4jEpisodicEdgeOperations
from graphiti_core.edges import EpisodicEdge

# the same legacy mention can be reached through either public Neo4j operation and transaction mode
@pytest.mark.parametrize('transactional', [True, False])
@pytest.mark.parametrize('bulk', [True, False])
def test_save_adopts_existing_identity_and_creation_time(transactional, bulk):

    original_time = datetime(2026, 1, 1, tzinfo = timezone.utc)
    edge = EpisodicEdge(uuid = 'new-attempt', group_id = 'company', source_node_uuid = 'episode',
                        target_node_uuid = 'entity', created_at = datetime.now(timezone.utc))
    records = [{'requested_uuid': edge.uuid, 'uuid': 'original-mention', 'created_at': original_time}]
    executor = SimpleNamespace(execute_query = AsyncMock(return_value = (records, None, None)))
    tx = SimpleNamespace(run = AsyncMock(return_value = SimpleNamespace(data = AsyncMock(return_value = records))))
    operations = Neo4jEpisodicEdgeOperations()
    if bulk:
        asyncio.run(operations.save_bulk(executor, [edge], tx = tx if transactional else None))
    else:
        asyncio.run(operations.save(executor, edge, tx = tx if transactional else None))
    assert edge.uuid == 'original-mention' and edge.created_at == original_time
    call = tx.run.await_args if transactional else executor.execute_query.await_args
    assert call.kwargs['episodic_edges'][0]['uuid'] == 'new-attempt'
    assert 'ON CREATE SET' in call.args[0]
    assert 'MENTIONS {group_id: edge.group_id}' in call.args[0]

# unordered database records must be correlated by the original requested identity
def test_bulk_result_order_does_not_rebind_mentions():

    now = datetime.now(timezone.utc)
    edges = [EpisodicEdge(uuid = f'request-{index}', group_id = 'company', source_node_uuid = 'episode',
                         target_node_uuid = f'entity-{index}', created_at = now) for index in range(2)]
    records = [{'requested_uuid': f'request-{index}', 'uuid': f'stored-{index}', 'created_at': now}
               for index in (1, 0)]
    executor = SimpleNamespace(execute_query = AsyncMock(return_value = (records, None, None)))
    asyncio.run(Neo4jEpisodicEdgeOperations().save_bulk(executor, edges))
    assert [edge.uuid for edge in edges] == ['stored-0', 'stored-1']

# incomplete, conflicting or ambiguous storage responses cannot partially mutate caller-owned identities
@pytest.mark.parametrize('fault', ['missing', 'unknown', 'ambiguous', 'empty-id', 'missing-time'])
def test_invalid_storage_result_does_not_mutate_any_edge(fault):

    now = datetime.now(timezone.utc)
    edges = [EpisodicEdge(uuid = f'request-{index}', group_id = 'company', source_node_uuid = 'episode',
                         target_node_uuid = f'entity-{index}', created_at = now) for index in range(2)]
    original = [edge.model_dump() for edge in edges]
    records = [{'requested_uuid': f'request-{index}', 'uuid': f'stored-{index}', 'created_at': now}
               for index in range(2)]
    if fault == 'missing':
        records.pop()
    elif fault == 'unknown':
        records[-1]['requested_uuid'] = 'unknown'
    elif fault == 'ambiguous':
        records.append({**records[-1], 'uuid': 'different-stored-id'})
    elif fault == 'empty-id':
        records[-1]['uuid'] = ''
    else:
        records[-1]['created_at'] = None
    executor = SimpleNamespace(execute_query = AsyncMock(return_value = (records, None, None)))
    with pytest.raises(ValueError):
        asyncio.run(Neo4jEpisodicEdgeOperations().save_bulk(executor, edges))
    assert [edge.model_dump() for edge in edges] == original

# conflicting endpoint bindings for one UUID must be rejected before reaching persistence
def test_conflicting_requested_uuid_does_not_reach_persistence():

    now = datetime.now(timezone.utc)
    first = EpisodicEdge(uuid = 'same', group_id = 'company', source_node_uuid = 'episode',
                         target_node_uuid = 'first', created_at = now)
    second = first.model_copy(update = {'target_node_uuid': 'second'})
    executor = SimpleNamespace(execute_query = AsyncMock())
    with pytest.raises(ValueError):
        asyncio.run(Neo4jEpisodicEdgeOperations().save_bulk(executor, [first, second]))
    executor.execute_query.assert_not_awaited()

# dependency failures and cancellation retain their existing retry and shutdown semantics
@pytest.mark.parametrize('failure', [ConnectionError('Unavailable'), asyncio.CancelledError()])
def test_storage_interruption_preserves_original_exception(failure):

    edge = EpisodicEdge(uuid = 'requested', group_id = 'company', source_node_uuid = 'episode',
                        target_node_uuid = 'entity', created_at = datetime.now(timezone.utc))
    executor = SimpleNamespace(execute_query = AsyncMock(side_effect = failure))
    with pytest.raises(type(failure)) as caught:
        asyncio.run(Neo4jEpisodicEdgeOperations().save(executor, edge))
    assert caught.value is failure and edge.uuid == 'requested'
    executor.execute_query.assert_awaited_once()


