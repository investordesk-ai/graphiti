"""Temporal recovery distinguishes explicit unknown evidence from malformed or conflicting model output."""

import asyncio

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.errors import LLMResponseError
from graphiti_core.nodes import EpisodicNode, EpisodeType
from graphiti_core.utils.maintenance.edge_operations import prepare_edge_timestamps

# construct identified source facts for the public temporal-preparation entry point
def evidence(count):

    episode = EpisodicNode(
        name = 'Source', group_id = 'company', source = EpisodeType.text, source_description = 'Source',
        content = 'Original source', valid_at = datetime(2026, 5, 15, tzinfo = timezone.utc),
    ) # EpisodicNode
    edges = [EntityEdge(
        group_id = 'company', source_node_uuid = f'source-{index}', target_node_uuid = f'target-{index}',
        name = 'OCCURRED', fact = f'Event {index} occurred on 31 March 2026', created_at = episode.valid_at,
    ) for index in range(count)]
    return episode, edges

# malformed batch output reaches individual validation and a distinct correction prompt
def test_malformed_temporal_response_gets_corrected():

    episode, edges = evidence(1)
    history = []

    async def generate(messages, **kwargs):
        name = kwargs['prompt_name']
        history.append(name)
        if name in ('extract_edges.extract_timestamps_batch', 'extract_edges.extract_timestamps_batch_repair'):
            return {'timestamps': [{'edge_id': edges[0].uuid}]}
        if name == 'extract_edges.extract_timestamps':
            return {'valid_at': 'not a date', 'invalid_at': None}
        return {'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None}

    client = SimpleNamespace(generate_response = AsyncMock(side_effect = generate))
    asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert edges[0].valid_at == datetime(2026, 3, 31, tzinfo = timezone.utc)
    assert history == ['extract_edges.extract_timestamps_batch', 'extract_edges.extract_timestamps_batch_repair',
                       'extract_edges.extract_timestamps',
                       'extract_edges.extract_timestamps_repair']

# a rejected correction cannot discard already validated source dates or repeat their inference
def test_batch_correction_failure_preserves_valid_peers():

    episode, edges = evidence(2)
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        {'timestamps': [{'edge_id': edges[0].uuid, 'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None}]},
        LLMResponseError('Budget exhausted', recovery_action = 'reduce'),
        {'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None},
    ])) # SimpleNamespace
    asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert all(edge.valid_at == datetime(2026, 3, 31, tzinfo = timezone.utc) for edge in edges)
    calls = client.generate_response.await_args_list
    assert len(calls) == 3
    assert calls[-1].kwargs['prompt_name'] == 'extract_edges.extract_timestamps'
    assert edges[1].fact in calls[-1].args[0][-1].content
    assert edges[0].fact not in calls[-1].args[0][-1].content

# a duplicated source ID is ambiguous even when each of its dates is individually parseable
def test_duplicate_timestamp_ids_require_individual_recovery():

    episode, edges = evidence(2)
    conflicting = {'timestamps': [
        {'edge_id': edges[0].uuid, 'valid_at': '2026-03-01T00:00:00Z', 'invalid_at': None},
        {'edge_id': edges[0].uuid, 'valid_at': '2026-03-02T00:00:00Z', 'invalid_at': None},
        {'edge_id': edges[1].uuid, 'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None},
    ]} # conflicting
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        conflicting, conflicting, {'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None},
    ])) # SimpleNamespace
    asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert all(edge.valid_at == datetime(2026, 3, 31, tzinfo = timezone.utc) for edge in edges)
    assert client.generate_response.await_count == 3

# a later outage carries explicit completed unknown results, not merely the non-null date fields
def test_completed_unknown_dates_survive_a_later_outage():

    episode, edges = evidence(9)
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        {'timestamps': [{'edge_id': edge.uuid, 'valid_at': None, 'invalid_at': None} for edge in edges[:8]]},
        LLMResponseError('Endpoint unavailable', recovery_action = 'retry'),
    ])) # SimpleNamespace
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert caught.value.recovery_action == 'retry'
    assert caught.value.diagnostics['validated_timestamps'] == {
        edge.uuid: {'valid_at': None, 'invalid_at': None} for edge in edges[:8]
    }
    assert caught.value.diagnostics['source_edge_ids'] == [edge.uuid for edge in edges]
    assert client.generate_response.await_count == 2

# repeated invalid intervals require review rather than silently clearing the failed interpretation
def test_exhausted_temporal_correction_preserves_evidence_for_review():

    episode, edges = evidence(1)
    bad = {'valid_at': '2026-04-01T00:00:00Z', 'invalid_at': '2026-03-01T00:00:00Z'}
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        {'timestamps': [{'edge_id': edges[0].uuid, **bad}]},
        {'timestamps': [{'edge_id': edges[0].uuid, **bad}]}, bad, bad,
    ])) # SimpleNamespace
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(prepare_edge_timestamps(client, edges, episode))
    error = caught.value
    assert error.recovery_action == 'review'
    assert error.diagnostics['source']['fact'] == edges[0].fact
    assert error.diagnostics['validation_errors']
    assert error.diagnostics['corrected_response'] == bad
    assert error.diagnostics['validated_timestamps'] == {}
    assert edges[0].valid_at is None and edges[0].invalid_at is None
    assert client.generate_response.await_count == 4

# cancellation never falls through to additional model calls
def test_temporal_cancellation_propagates():

    episode, edges = evidence(1)
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert client.generate_response.await_count == 1

# completed peers in the current batch survive an outage before that batch's correction can return
def test_current_temporal_batch_survives_correction_outage():

    episode, edges = evidence(2)
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        {'timestamps': [{'edge_id': edges[0].uuid, 'valid_at': None, 'invalid_at': None}]},
        LLMResponseError('Unavailable', recovery_action = 'retry'),
    ])) # SimpleNamespace
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert caught.value.recovery_action == 'retry'
    assert caught.value.diagnostics['validated_timestamps'] == {
        edges[0].uuid: {'valid_at': None, 'invalid_at': None},
    }
    assert client.generate_response.await_count == 2

# malformed fields in one timestamp record leave its validated peer intact
def test_malformed_temporal_peer_is_recovered_individually():

    episode, edges = evidence(2)
    batch = {'timestamps': [
        {'edge_id': edges[0].uuid, 'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None},
        {'edge_id': edges[1].uuid, 'valid_at': {'invalid': 'shape'}, 'invalid_at': None},
    ]} # batch
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = [
        batch, batch, {'valid_at': '2026-03-31T00:00:00Z', 'invalid_at': None},
    ])) # SimpleNamespace
    asyncio.run(prepare_edge_timestamps(client, edges, episode))
    assert all(edge.valid_at == datetime(2026, 3, 31, tzinfo = timezone.utc) for edge in edges)
    calls = client.generate_response.await_args_list
    assert len(calls) == 3
    assert edges[1].fact in calls[-1].args[0][-1].content
    assert edges[0].fact not in calls[-1].args[0][-1].content

