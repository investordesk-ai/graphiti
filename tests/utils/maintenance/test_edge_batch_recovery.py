"""Batch recovery preserves every relationship through the public resolution entry point."""

import asyncio

import pytest

from graphiti_core.llm_client.errors import LLMResponseError, RateLimitError, RefusalError
from test_bounded_edge_resolution import resolve

# a batch boundary error is compensated by smaller requests without losing provenance
@pytest.mark.parametrize("failure", [
    RuntimeError("Unexpected adapter error"), ValueError("Malformed response"),
    LLMResponseError("Output budget exhausted", recovery_action = "reduce"),
])
def test_failed_batch_reduces_to_smaller_batches(monkeypatch, failure):

    history = []
    calls = resolve(monkeypatch, 4, failures = {0: failure}, history = history)
    assert [len(batch["ids"]) for batch in history] == [4, 2, 2]
    assert history[1]["ids"] + history[2]["ids"] == history[0]["ids"]
    original = {case["edge_id"]: case["instructions"] for case in history[0]["cases"]}
    for batch in history[1:]:
        for case in batch["cases"]:
            assert case["instructions"] == original[case["edge_id"]]
    assert calls.count("dedupe_edges.resolve_edge") == 0

# smaller requests that still fail must reach the existing validated individual-edge path
def test_repeated_batch_failure_falls_back_to_individual_edges(monkeypatch):

    history = []
    failures = {index: RuntimeError("Bad adapter output") for index in range(3)}
    calls = resolve(monkeypatch, 4, failures = failures, history = history)
    assert [len(batch["ids"]) for batch in history] == [4, 2, 2]
    assert calls.count("dedupe_edges.resolve_edge") == 4

# completed earlier batches must never be repeated while a later batch is repaired
def test_completed_batches_survive_a_later_failure(monkeypatch):

    history = []
    resolve(monkeypatch, 8, failures = {1: RuntimeError("Bad later batch")}, history = history)
    assert [len(batch["ids"]) for batch in history] == [4, 4, 2, 2]
    completed = set(history[0]["ids"])
    assert all(not completed.intersection(batch["ids"]) for batch in history[1:])
    assert history[2]["ids"] + history[3]["ids"] == history[1]["ids"]

# output grammars must bound repetition by the actual number of cases and supplied candidate lists
def test_response_schema_bounds_output_repetition(monkeypatch):

    history = []
    resolve(monkeypatch, 4, history = history)
    schema = history[0]["schema"]
    assert schema["properties"]["decisions"]["maxItems"] == 4
    decision = schema["$defs"]["BoundedEdgeBatchDecision"]["properties"]
    for field in ("duplicate_facts", "contradicted_existing_facts", "contradicted_invalidation_candidates"):
        assert decision[field]["maxItems"] == 1
    assert decision["edge_id"]["maxLength"] == max(len(identifier) for identifier in history[0]["ids"])

# correlation is by stable source ID rather than the model's ordering
def test_reordered_decisions_preserve_canonical_relationships(monkeypatch):

    calls = resolve(monkeypatch, 4, "reordered")
    assert calls == ["dedupe_edges.resolve_edge_batch"]

# malformed peers cannot discard or reinterpret the other valid decisions
@pytest.mark.parametrize("mode", ["unknown", "coercible", "repeated"])
def test_only_unresolved_peer_uses_individual_processing(monkeypatch, mode):

    calls = resolve(monkeypatch, 4, mode)
    assert calls.count("dedupe_edges.resolve_edge") == 1

# unavailable dependencies and explicit review requirements are not solved by changing batch size
@pytest.mark.parametrize("failure", [
    LLMResponseError("Backend unavailable", recovery_action = "retry"),
    LLMResponseError("Configuration rejected", recovery_action = "review"),
    RateLimitError(), RefusalError("Refused"),
])
def test_non_output_failure_returns_to_owning_stage(monkeypatch, failure):

    history = []
    with pytest.raises(type(failure)):
        resolve(monkeypatch, 4, failures = {0: failure}, history = history)
    assert len(history) == 1

# cancellation cannot spend more inference by falling through to smaller batches
def test_batch_cancellation_is_preserved(monkeypatch):

    history = []
    with pytest.raises(asyncio.CancelledError):
        resolve(monkeypatch, 4, failures = {0: asyncio.CancelledError()}, history = history)
    assert len(history) == 1

# an odd batch reduces safely and leaves its final case to the normal individual path
def test_odd_batch_reduction_preserves_every_case(monkeypatch):

    from graphiti_core.utils.maintenance import edge_operations
    monkeypatch.setattr(edge_operations, "EDGE_RESOLUTION_BATCH_SIZE", 3)
    history = []
    calls = resolve(monkeypatch, 3, failures = {0: RuntimeError("Bad batch")}, history = history)
    assert [len(batch["ids"]) for batch in history] == [3, 2]
    assert calls.count("dedupe_edges.resolve_edge") == 1
