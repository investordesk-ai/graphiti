
"""Bounded edge inference through the public Neo4j resolution entry point."""

import asyncio
import json

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.utils.maintenance import edge_operations

# run real reconciliation with isolated public search, embedding and model dependencies
def resolve(monkeypatch, count, mode = "normal", *, failures = None, history = None):

    now = datetime(2026, 1, 1, tzinfo = timezone.utc)
    episode = EpisodicNode(
        name = "Report", group_id = "company", source = EpisodeType.text,
        content = "Reported results", source_description = "Report", valid_at = now,
    ) # EpisodicNode
    nodes = []
    incoming = []
    candidates = {}
    calls = []

    for index in range(count):
        source = EntityNode(name = f"Company {index}", group_id = "company")
        target = EntityNode(name = f"Metric {index}", group_id = "company")
        nodes.extend([source, target])
        edge = EntityEdge(
            group_id = "company", name = "REPORTED", source_node_uuid = source.uuid,
            target_node_uuid = target.uuid, fact = f"Revenue is {index + 10} million",
            created_at = now, valid_at = now, episodes = [episode.uuid],
        ) # EntityEdge
        prior = EntityEdge(
            group_id = "company", name = "REPORTED", source_node_uuid = source.uuid,
            target_node_uuid = target.uuid, fact = f"Reported revenue of {index + 10} million",
            created_at = now, valid_at = now, episodes = [str(uuid4())],
        ) # EntityEdge
        incoming.append(edge)
        candidates[edge.fact] = prior

    async def between(driver, source, target):
        for candidate in candidates.values():
            if candidate.source_node_uuid == source and candidate.target_node_uuid == target:
                return [candidate]
        return []

    async def search(clients, fact, **kwargs):
        selected = []
        if kwargs["search_filter"].edge_uuids:
            selected.append(candidates[fact])
        return SimpleNamespace(edges = selected)

    async def generate(messages, **kwargs):
        prompt = kwargs["prompt_name"]
        calls.append(prompt)
        decision = {
            "duplicate_facts": [0],
            "contradicted_existing_facts": [],
            "contradicted_invalidation_candidates": [],
        } # decision

        if prompt == "dedupe_edges.resolve_edge_batch":
            cases = json.loads(messages[-1].content)
            if history is not None:
                history.append({
                    "ids": [case["edge_id"] for case in cases],
                    "schema": kwargs["response_model"].model_json_schema(),
                    "cases": cases,
                }) # history.append
            batch_index = calls.count(prompt) - 1
            if failures and batch_index in failures:
                raise failures[batch_index]
            decisions = []
            for case in cases:
                decisions.append({"edge_id": case["edge_id"], **decision})

            if mode == "missing":
                decisions.pop()
            elif mode == "duplicate":
                decisions.append(decisions[0])
            elif mode == "invalid_index":
                decisions[0]["duplicate_facts"] = [99]
            elif mode == "malformed":
                return {"decisions": "invalid"}
            elif mode == "reordered":
                decisions.reverse()
            elif mode == "unknown":
                decisions[0]["edge_id"] = "unknown"
            elif mode == "coercible":
                decisions[0]["duplicate_facts"] = [False]
            elif mode == "repeated":
                decisions[0]["duplicate_facts"] = [0, 0]

            return {"decisions": decisions}

        return decision

    monkeypatch.setattr(EntityEdge, "get_between_nodes", between)
    monkeypatch.setattr(edge_operations, "search", search)
    monkeypatch.setattr(edge_operations, "create_entity_edge_embeddings", AsyncMock())
    clients = SimpleNamespace(
        driver = object(), embedder = object(),
        llm_client = SimpleNamespace(generate_response = generate),
    ) # SimpleNamespace

    result = asyncio.run(edge_operations.resolve_extracted_edges(
        clients, incoming, episode, nodes, {}, {}, preserve_existing_attributes = True,
    )) # asyncio.run

    assert len(result[0]) == count
    for index, edge in enumerate(result[0]):
        assert episode.uuid in edge.episodes
        assert edge.uuid == candidates[incoming[index].fact].uuid
    return calls

# four independent decisions share a call without changing canonical provenance
def test_four_edges_share_one_call(monkeypatch):
    assert resolve(monkeypatch, 4) == ["dedupe_edges.resolve_edge_batch"]

# a final single decision retains the existing public single-edge behaviour
def test_batch_bound_and_single_remainder(monkeypatch):
    calls = resolve(monkeypatch, 5)
    assert calls.count("dedupe_edges.resolve_edge_batch") == 1
    assert calls.count("dedupe_edges.resolve_edge") == 1

# an omitted decision is independently resolved rather than silently accepted
def test_missing_decision_retries_only_that_edge(monkeypatch):
    calls = resolve(monkeypatch, 4, "missing")
    assert calls.count("dedupe_edges.resolve_edge") == 1

# duplicate IDs cannot choose arbitrarily between conflicting responses
def test_duplicate_decision_retries_affected_edge(monkeypatch):
    calls = resolve(monkeypatch, 4, "duplicate")
    assert calls.count("dedupe_edges.resolve_edge") == 1

# a bad candidate index enters the same per-edge repair validation as before
def test_invalid_index_uses_existing_repair(monkeypatch):
    calls = resolve(monkeypatch, 4, "invalid_index")
    assert calls.count("dedupe_edges.resolve_edge_repair") == 1

# an invalid batch shape falls back to validated individual decisions
def test_malformed_batch_preserves_single_edge_resolution(monkeypatch):
    calls = resolve(monkeypatch, 4, "malformed")
    assert calls.count("dedupe_edges.resolve_edge") == 4

# the prompt bound sends oversized evidence through the existing path without truncation
def test_prompt_limit_preserves_individual_evidence(monkeypatch):

    monkeypatch.setattr(edge_operations, "EDGE_RESOLUTION_MAX_CHARACTERS", 1)
    calls = resolve(monkeypatch, 4)

    assert calls.count("dedupe_edges.resolve_edge_batch") == 0
    assert calls.count("dedupe_edges.resolve_edge") == 4

# no extracted facts require no model resolution
def test_no_edges_need_no_model_call(monkeypatch):
    assert resolve(monkeypatch, 0) == []
