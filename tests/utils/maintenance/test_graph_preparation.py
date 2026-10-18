
"""Temporal preparation uses the same validated Graphiti decisions without a graph connection."""

import asyncio

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode, EpisodeType
from graphiti_core.utils.maintenance.edge_operations import prepare_edge_timestamps

# construct source evidence independently of any database driver
def evidence():

    episode = EpisodicNode(
        name = "Report", group_id = "", source = EpisodeType.text, content = "Source evidence",
        source_description = "Report", valid_at = datetime(2026, 5, 15, tzinfo = timezone.utc),
    )
    edge = EntityEdge(
        group_id = "", source_node_uuid = "one", target_node_uuid = "two",
        name = "REPORTED", fact = "Revenue was reported in March 2026",
        created_at = episode.valid_at,
    )
    return episode, edge

# explicit null bounds are valid completed interpretation, not an invented reference date
def test_unknown_temporal_bounds_remain_unknown():

    episode, edge = evidence()
    client = SimpleNamespace(generate_response = AsyncMock(return_value = {
        "timestamps": [{"edge_id": edge.uuid, "valid_at": None, "invalid_at": None}],
    }))
    asyncio.run(prepare_edge_timestamps(client, [edge], episode))
    assert edge.valid_at is None
    assert edge.invalid_at is None
    assert client.generate_response.await_count == 1

# inference failure cannot be mistaken for completed preparation
def test_failed_preparation_propagates():

    episode, edge = evidence()
    client = SimpleNamespace(generate_response = AsyncMock(side_effect = RuntimeError("model unavailable")))
    with pytest.raises(RuntimeError, match = "model unavailable"):
        asyncio.run(prepare_edge_timestamps(client, [edge], episode))

# valid source dates are retained rather than replaced by the artifact reference date
def test_preparation_retains_source_dates():

    episode, edge = evidence()
    client = SimpleNamespace(generate_response = AsyncMock(return_value = {
        "timestamps": [{
            "edge_id": edge.uuid, "valid_at": "2026-03-31T00:00:00Z", "invalid_at": None,
        }],
    }))
    asyncio.run(prepare_edge_timestamps(client, [edge], episode))
    assert edge.valid_at == datetime(2026, 3, 31, tzinfo = timezone.utc)
