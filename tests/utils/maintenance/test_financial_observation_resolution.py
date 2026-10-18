import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from graphiti_core.nodes import EntityNode
from graphiti_core.utils.maintenance import node_operations

# use the public driver interface for exact identity retrieval
class Store:

    # retain stored nodes and record exact lookups without implementing semantic search
    def __init__(self):
        self.nodes = {}
        self.lookups = []
        self.graph_operations_interface = self

    # honour Graphiti's public node lookup contract with isolated copies
    async def node_get_by_uuids(self, cls, driver, uuids, group_id):
        self.lookups.append(list(uuids))
        results = []
        for uuid in uuids:
            if uuid in self.nodes:
                results.append(self.nodes[uuid].model_copy(deep = True))
        return results

# each call represents a newly extracted observation from a different episode
def observation(key = "a" * 64, group = "company_a", amount = 17_000_000):
    return EntityNode(
        name = "Very similar observation name",
        group_id = group,
        labels = ["Entity", "FinancialObservation"],
        attributes = {"observation_key": key, "amount": amount, "source_text": "first evidence"},
    )

# inference mocks must remain unused for qualified observations
def clients():
    return SimpleNamespace(driver = Store(), embedder = AsyncMock(), llm_client = AsyncMock())

# exercise the public resolver through its normal synchronous test boundary
def resolve(client, nodes, overrides = None):
    return asyncio.run(node_operations.resolve_extracted_nodes(
        client, nodes, existing_nodes_override = overrides,
    ))

# same key and group reuse identity without semantic retrieval or an inference call
def test_cross_episode_replay():
    client = clients()
    first = observation()
    first_nodes, first_map, _ = resolve(client, [first])
    canonical = first_nodes[0]
    client.driver.nodes[canonical.uuid] = canonical
    second = observation()
    second.attributes["source_text"] = "second evidence"
    second_nodes, second_map, _ = resolve(client, [second])
    assert second_nodes[0].uuid == canonical.uuid
    assert second_map[second.uuid] == first_map[first.uuid]
    assert second_nodes[0].attributes["source_text"] == "first evidence"
    assert second.attributes["source_text"] == "second evidence"
    client.embedder.create_batch.assert_not_awaited()
    client.llm_client.generate_response.assert_not_awaited()

# matching display names cannot merge different observation keys or graph groups
@pytest.mark.parametrize("key, group", [("b" * 64, "company_a"), ("a" * 64, "company_b")])
def test_distinct_identities(key, group):
    client = clients()
    first_nodes, _, _ = resolve(client, [observation()])
    first = first_nodes[0]
    client.driver.nodes[first.uuid] = first
    second_nodes, _, _ = resolve(client, [observation(key = key, group = group)])
    assert second_nodes[0].uuid != first.uuid

# a new amount cannot replace the stored amount for the same qualified identity
def test_stored_conflict():
    client = clients()
    nodes, _, _ = resolve(client, [observation()])
    client.driver.nodes[nodes[0].uuid] = nodes[0]
    before = deepcopy(client.driver.nodes)
    with pytest.raises(ValueError, match = "conflicts"):
        resolve(client, [observation(amount = 30_000_000)])
    assert client.driver.nodes == before

# conflicts within one submitted batch are rejected before querying the store
def test_batch_conflict():
    client = clients()
    with pytest.raises(ValueError, match = "Conflicting amounts"):
        resolve(client, [observation(), observation(amount = 30_000_000)])
    assert client.driver.lookups == []

# malformed keys and non-finite or boolean amounts cannot acquire a canonical identity
@pytest.mark.parametrize("key, amount", [
    ("", 17), ("not-a-key", 17), ("a" * 64, float("nan")), ("a" * 64, True),
])
def test_invalid_observation(key, amount):
    client = clients()
    with pytest.raises(ValueError):
        resolve(client, [observation(key = key, amount = amount)])
    assert client.driver.lookups == []

# an existing observation is not a candidate for an ordinary entity with the same name
def test_ordinary_entity_cannot_merge_into_observation(monkeypatch):
    client = clients()
    existing = observation()
    ordinary = EntityNode(name = existing.name, group_id = existing.group_id, labels = ["Entity", "Company"])
    client.embedder.create_batch.return_value = [[0.5]]
    search = AsyncMock(return_value = [existing])
    monkeypatch.setattr(node_operations, "node_similarity_search", search)
    result, mapping, _ = resolve(client, [ordinary])
    assert result[0].uuid == ordinary.uuid
    assert mapping[ordinary.uuid] == ordinary.uuid
    client.llm_client.generate_response.assert_not_awaited()

# mixed input preserves order and remaps each original endpoint UUID
def test_mixed_input_order(monkeypatch):
    client = clients()
    protected = observation()
    ordinary = EntityNode(name = "Techstep", group_id = "company_a", labels = ["Entity", "Company"])
    client.embedder.create_batch.return_value = [[0.5]]
    monkeypatch.setattr(node_operations, "node_similarity_search", AsyncMock(return_value = []))
    result, mapping, _ = resolve(client, [protected, ordinary])
    assert result[0].uuid == mapping[protected.uuid]
    assert result[1].uuid == ordinary.uuid
    assert protected.uuid != result[0].uuid

# financing disclosures reuse exact provenance, never an observation's key namespace
def test_financing_replay_and_observation_separation():

    client = clients()
    first = observation()
    first.labels = ["Entity", "FinancingRecord"]
    first.attributes["financing_key"] = first.attributes.pop("observation_key")
    result, _, _ = resolve(client, [first, observation()])
    financing = result[0]
    assert financing.uuid != result[1].uuid
    client.driver.nodes[financing.uuid] = financing

    second = first.model_copy(deep = True)
    second.uuid = "another-extraction"
    repeated, mapping, _ = resolve(client, [second])
    assert repeated[0].uuid == financing.uuid
    assert mapping[second.uuid] == financing.uuid
    client.llm_client.generate_response.assert_not_awaited()

# a stored financing amount cannot be overwritten under the same disclosure identity
def test_financing_conflict():

    client = clients()
    first = observation()
    first.labels = ["Entity", "FinancingRecord"]
    first.attributes["financing_key"] = first.attributes.pop("observation_key")
    result, _, _ = resolve(client, [first])
    client.driver.nodes[result[0].uuid] = result[0]
    second = first.model_copy(deep = True)
    second.attributes["amount"] = 99

    with pytest.raises(ValueError, match = "conflicts"):
        resolve(client, [second])

# normal company deduplication cannot consume a financing disclosure with the same display name
def test_ordinary_entity_cannot_merge_into_financing(monkeypatch):

    client = clients()
    existing = observation()
    existing.labels = ["Entity", "FinancingRecord"]
    existing.attributes["financing_key"] = existing.attributes.pop("observation_key")
    ordinary = EntityNode(name = existing.name, group_id = existing.group_id, labels = ["Entity", "Company"])
    client.embedder.create_batch.return_value = [[0.5]]
    monkeypatch.setattr(node_operations, "node_similarity_search", AsyncMock(return_value = [existing]))
    result, mapping, _ = resolve(client, [ordinary])
    assert result[0].uuid == ordinary.uuid
    assert mapping[ordinary.uuid] == ordinary.uuid
    client.llm_client.generate_response.assert_not_awaited()

# source-backed claims use their own identity namespace and reject a changed stored upper bound
def test_claim_resolution():

    client = clients()
    claim = observation()
    claim.labels = ["Entity", "FinancialClaim"]
    claim.attributes["claim_key"] = claim.attributes.pop("observation_key")
    claim.attributes["amount_upper"] = 20_000_000
    result, _, _ = resolve(client, [claim, observation()])
    canonical = result[0]
    assert canonical.uuid != result[1].uuid
    client.driver.nodes[canonical.uuid] = canonical
    replay, _, _ = resolve(client, [claim])
    assert replay[0].uuid == canonical.uuid
    changed = claim.model_copy(deep = True)
    changed.attributes["amount_upper"] = 21_000_000
    with pytest.raises(ValueError, match = "upper amount"):
        resolve(client, [changed])
