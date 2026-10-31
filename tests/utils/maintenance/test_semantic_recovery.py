"""Neo4j ingestion validates corrections, retains source identity and preserves unresolved evidence."""

import asyncio
import json

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic import ValidationError

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import LLMResponseError
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate
from graphiti_core.prompts.extract_edges import EdgeTimestamps, BatchEdgeTimestamp
from graphiti_core.prompts.extract_nodes import SummarizedEntity
from graphiti_core.utils.maintenance import edge_operations, node_operations

# read source sections from the actual prompt rather than duplicating its request-building logic
def section(messages, name):
    content = messages[-1].content
    return json.loads(content.split(f'<{name}>', 1)[1].split(f'</{name}>', 1)[0])

# these public entry-point scenarios use Neo4j-shaped dependencies without any external database writes
@pytest.fixture
def scenario(monkeypatch):

    episode = EpisodicNode(
        name = 'Evidence', group_id = 'company', source = EpisodeType.text,
        source_description = 'Source', content = 'Facts A, B, C and D are supported source evidence.',
        valid_at = datetime(2026, 1, 1, tzinfo = timezone.utc),
    ) # EpisodicNode
    incoming = [EntityNode(name = name, group_id = 'company', attributes = {'marker': name})
                for name in ('A', 'B', 'C', 'D')]
    candidates = [EntityNode(name = f'Canonical {node.name}', group_id = 'company') for node in incoming]
    model = SimpleNamespace(generate_response = AsyncMock())
    clients = SimpleNamespace(driver = MagicMock(spec = Neo4jDriver), embedder = AsyncMock(), llm_client = model)
    by_source = {node.uuid: candidate for node, candidate in zip(incoming, candidates)}
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', AsyncMock(
        side_effect = lambda clients, nodes: [[by_source[node.uuid]] for node in nodes],
    )) # monkeypatch.setattr
    monkeypatch.setattr(node_operations, 'create_entity_node_embeddings', AsyncMock())
    return episode, incoming, candidates, model, clients

# semantic failures get the existing, distinct correction prompt before any canonical identity is applied
@pytest.mark.parametrize('bad', [999, -2, True, '0'])
def test_invalid_node_candidate_is_corrected(scenario, bad):

    episode, incoming, candidates, model, clients = scenario
    clients.llm_client.generate_response.side_effect = [
        {'entity_resolutions': [{'id': 0, 'name': 'A', 'duplicate_candidate_id': bad}]},
        {'entity_resolutions': [{'id': 0, 'name': 'Canonical A', 'duplicate_candidate_id': 0}]},
    ]
    result = asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming[:1], episode))
    assert result[1][incoming[0].uuid] == candidates[0].uuid
    assert [call.kwargs['prompt_name'] for call in model.generate_response.await_args_list] == [
        'dedupe_nodes.nodes', 'dedupe_nodes.nodes_repair',
    ]

# valid peers cannot be replaced by a conflicting correction or repeated during smaller-work fallback
def test_node_recovery_preserves_valid_peers(scenario):

    episode, incoming, candidates, model, clients = scenario
    history = []

    async def generate(messages, **kwargs):
        entities = section(messages, 'ENTITIES')
        history.append([entity['name'] for entity in entities])
        if len(history) == 1:
            return {'entity_resolutions': [
                {'id': index, 'name': name, 'duplicate_candidate_id': index}
                for index, name in enumerate(('A', 'B'))
            ]}
        if len(history) == 2:
            return {'entity_resolutions': [{'id': 0, 'name': 'A', 'duplicate_candidate_id': -1}]}
        name = entities[0]['name']
        index = ('A', 'B', 'C', 'D').index(name)
        return {'entity_resolutions': [{'id': 0, 'name': name, 'duplicate_candidate_id': index}]}

    model.generate_response.side_effect = generate
    result = asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming, episode))
    assert history == [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], ['C'], ['D']]
    assert result[1] == {node.uuid: candidate.uuid for node, candidate in zip(incoming, candidates)}

# an explicit request to reduce work cannot trigger another unchanged batch correction
def test_node_budget_rejection_reduces_work(scenario):

    episode, incoming, candidates, model, clients = scenario
    history = []

    async def generate(messages, **kwargs):
        entities = section(messages, 'ENTITIES')
        history.append([entity['name'] for entity in entities])
        if len(history) == 1:
            raise LLMResponseError('Budget exhausted', recovery_action = 'reduce')
        return {'entity_resolutions': [
            {'id': index, 'name': entity['name'],
             'duplicate_candidate_id': ('A', 'B', 'C', 'D').index(entity['name'])}
            for index, entity in enumerate(entities)
        ]}

    model.generate_response.side_effect = generate
    result = asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming, episode))
    assert [len(batch) for batch in history] == [4, 2, 2]
    assert len(result[0]) == 4

# a transport interruption retains validated node decisions for the owning service's checkpoint
def test_node_correction_outage_preserves_valid_decisions(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.side_effect = [
        {'entity_resolutions': [{'id': 0, 'name': 'A', 'duplicate_candidate_id': 0}]},
        LLMResponseError('Unavailable', recovery_action = 'retry'),
    ]
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming[:2], episode))
    assert caught.value.recovery_action == 'retry'
    evidence = caught.value.diagnostics
    assert evidence['validated_node_decisions']['0']['duplicate_candidate_id'] == 0
    assert evidence['source_node_ids'] == [node.uuid for node in incoming[:2]]
    assert model.generate_response.await_count == 2

# a valid summary remains attached to its source UUID while only unresolved entities are retried
def test_summary_recovery_preserves_valid_peers(scenario):

    episode, incoming, candidates, model, clients = scenario
    history = []

    async def generate(messages, **kwargs):
        entities = section(messages, 'ENTITIES')
        history.append([entity['attributes']['marker'] for entity in entities])
        if len(history) == 1:
            return {'summaries': [{'entity_id': 0, 'summary': 'Fact A'}]}
        if len(history) == 2:
            return {'summaries': [{'entity_id': 0, 'summary': 'Must not replace A'}]}
        return {'summaries': [{'entity_id': 0, 'summary': 'Fact B'}]}

    model.generate_response.side_effect = generate
    result = asyncio.run(node_operations.extract_attributes_from_nodes(
        clients, incoming[:2], episode, skip_fact_appending = True, extract_attributes = False,
    )) # extract_attributes_from_nodes
    assert [node.summary for node in result] == ['Fact A', 'Fact B']
    assert history == [['A', 'B'], ['A', 'B'], ['B']]

# conflicting duplicate summary IDs require recovery; the first arbitrary value cannot become durable state
def test_duplicate_summary_ids_never_choose_an_arbitrary_value(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.side_effect = [
        {'summaries': [{'entity_id': 0, 'summary': 'Wrong first'}, {'entity_id': 0, 'summary': 'Wrong second'}]},
        {'summaries': [{'entity_id': 0, 'summary': 'Corrected source summary'}]},
    ]
    result = asyncio.run(node_operations.extract_attributes_from_nodes(
        clients, incoming[:1], episode, skip_fact_appending = True, extract_attributes = False,
    )) # extract_attributes_from_nodes
    assert result[0].summary == 'Corrected source summary'
    assert model.generate_response.await_args.kwargs['prompt_name'] == 'extract_nodes.extract_summaries_batch_repair'

# exhausted single-item summary recovery retains its source rather than claiming an empty successful summary
def test_unresolved_summary_requests_review(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.return_value = {'summaries': []}
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(node_operations.extract_attributes_from_nodes(
            clients, incoming[:1], episode, skip_fact_appending = True, extract_attributes = False,
        )) # extract_attributes_from_nodes
    assert caught.value.recovery_action == 'review'
    assert caught.value.diagnostics['unresolved_node_ids'] == [incoming[0].uuid]
    assert incoming[0].summary == ''
    assert model.generate_response.await_count == 2

# already validated summaries survive dependency failure while correcting another source entity
def test_summary_correction_outage_retains_checkpoint(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.side_effect = [
        {'summaries': [{'entity_id': 0, 'summary': 'Fact A'}]},
        LLMResponseError('Unavailable', recovery_action = 'retry'),
    ]
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(node_operations.extract_attributes_from_nodes(
            clients, incoming[:2], episode, skip_fact_appending = True, extract_attributes = False,
        )) # extract_attributes_from_nodes
    assert caught.value.recovery_action == 'retry'
    assert caught.value.diagnostics['validated_summaries'] == {incoming[0].uuid: 'Fact A'}

# actual reference IDs may not be inferred by coercing booleans or strings into integers
@pytest.mark.parametrize('value', [True, False, '0', 0.5])
def test_response_identifiers_are_strict(value):

    with pytest.raises(ValidationError):
        NodeDuplicate(id = value, name = 'A', duplicate_candidate_id = -1)
    with pytest.raises(ValidationError):
        SummarizedEntity(entity_id = value, summary = 'Source')
    with pytest.raises(ValidationError):
        EdgeDuplicate(duplicate_facts = [value], contradicted_existing_facts = [],
                      contradicted_invalidation_candidates = [])

# missing bounds are incomplete model output; explicit nullable bounds are completed unknown evidence
@pytest.mark.parametrize('model,extra', [(EdgeTimestamps, {}), (BatchEdgeTimestamp, {'edge_id': 'source'})])
def test_temporal_completion_requires_explicit_fields(model, extra):

    for omitted in ({}, {'valid_at': None}, {'invalid_at': None}):
        with pytest.raises(ValidationError):
            model.model_validate({**extra, **omitted})
    assert model.model_validate({**extra, 'valid_at': None, 'invalid_at': None}).valid_at is None

# the actual shared client corrects wrong ID types before the Neo4j edge operation can consume them
def test_edge_operation_uses_distinct_schema_correction(scenario):

    episode, incoming, candidates, model, clients = scenario
    client = OpenAIClient(config = LLMConfig(model = 'run008-9B'), client = object())
    wrong = {'duplicate_facts': [False], 'contradicted_existing_facts': [],
             'contradicted_invalidation_candidates': []}
    corrected = {**wrong, 'duplicate_facts': [0]}
    client._generate_response = AsyncMock(side_effect = [(wrong, 1, 1), (corrected, 1, 1)])
    edge = EntityEdge(group_id = 'company', name = 'RELATION', source_node_uuid = 'a',
                      target_node_uuid = 'b', fact = 'A likes B', created_at = episode.valid_at)
    candidate = edge.model_copy(update = {'uuid': 'canonical', 'fact': 'A enjoys B'})
    result = asyncio.run(edge_operations.resolve_extracted_edge(
        client, edge, [candidate], [], episode, extract_timestamps = False,
    )) # resolve_extracted_edge
    assert result[0].uuid == 'canonical'
    assert client._generate_response.await_count == 2
    assert 'CORRECTION REQUEST' in client._generate_response.await_args.args[0][-1].content

# correction must resolve contradictory or out-of-range references before any graph relationship changes
@pytest.mark.parametrize('bad', [
    {'duplicate_facts': [99]}, {'duplicate_facts': [-2]}, {'duplicate_facts': [0, 0]},
    {'duplicate_facts': [0], 'contradicted_existing_facts': [0]},
])
def test_semantically_invalid_edge_decisions_request_review(scenario, bad):

    episode, incoming, candidates, model, clients = scenario
    response = {'duplicate_facts': [], 'contradicted_existing_facts': [],
                'contradicted_invalidation_candidates': [], **bad}
    model.generate_response.return_value = response
    edge = EntityEdge(group_id = 'company', name = 'RELATION', source_node_uuid = 'a',
                      target_node_uuid = 'b', fact = 'A likes B', created_at = episode.valid_at)
    candidate = edge.model_copy(update = {'uuid': 'canonical', 'fact': 'A enjoys B'})
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(edge_operations.resolve_extracted_edge(
            model, edge, [candidate], [], episode, extract_timestamps = False,
        )) # resolve_extracted_edge
    assert caught.value.recovery_action == 'review'
    assert caught.value.diagnostics['edge_id'] == edge.uuid
    assert candidate.episodes == [] and candidate.invalid_at is None
    assert model.generate_response.await_args.kwargs['prompt_name'] == 'dedupe_edges.resolve_edge_repair'

# cancellation must never become a guessed model decision or an unnecessary additional inference request
def test_node_cancellation_propagates(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming, episode))
    assert model.generate_response.await_count == 1

# retained evidence must remain serialisable and independent of later mutation of source objects
def test_recovery_evidence_is_a_serialisable_snapshot():

    original = {'source': ['original'], 'reference_time': datetime(2026, 1, 1, tzinfo = timezone.utc)}
    error = LLMResponseError('Review', recovery_action = 'review', diagnostics = original)
    original['source'].clear()
    assert error.diagnostics['source'] == ['original']
    assert '2026-01-01' in error.diagnostics['reference_time']
    json.dumps(error.diagnostics, allow_nan = False)

# completed child batches must survive a failure in the following child, not just the first batch's peers
def test_completed_summary_children_survive_later_failure(scenario):

    episode, incoming, candidates, model, clients = scenario
    history = []

    async def generate(messages, **kwargs):
        entities = section(messages, 'ENTITIES')
        markers = [entity['attributes']['marker'] for entity in entities]
        history.append(markers)
        if len(markers) == 4:
            return {'summaries': []}
        if markers == ['A', 'B']:
            return {'summaries': [{'entity_id': index, 'summary': f'Fact {marker}'}
                                  for index, marker in enumerate(markers)]}
        raise LLMResponseError('Unavailable', recovery_action = 'retry')

    model.generate_response.side_effect = generate
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(node_operations.extract_attributes_from_nodes(
            clients, incoming, episode, skip_fact_appending = True, extract_attributes = False,
        )) # extract_attributes_from_nodes
    assert caught.value.recovery_action == 'retry'
    assert caught.value.diagnostics['validated_summaries'] == {
        incoming[0].uuid: 'Fact A', incoming[1].uuid: 'Fact B',
    }
    assert history == [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], ['A', 'B'], ['C', 'D']]

# one malformed row cannot discard a valid peer while recovery reduces the unresolved work
@pytest.mark.parametrize('operation', ['nodes', 'summaries'])
def test_malformed_peer_does_not_discard_valid_record(scenario, operation):

    episode, incoming, candidates, model, clients = scenario
    history = []

    async def generate(messages, **kwargs):
        entities = section(messages, 'ENTITIES')
        names = [entity['name'] if operation == 'nodes' else entity['attributes']['marker']
                 for entity in entities]
        history.append(names)
        if len(history) <= 2:
            if operation == 'nodes':
                return {'entity_resolutions': [
                    {'id': 0, 'name': 'A', 'duplicate_candidate_id': 0},
                    {'id': 1, 'name': 'B', 'duplicate_candidate_id': True},
                ]}
            return {'summaries': [{'entity_id': 0, 'summary': 'Fact A'},
                                  {'entity_id': 1, 'summary': {'invalid': 'shape'}}]}
        if operation == 'nodes':
            return {'entity_resolutions': [{'id': 0, 'name': 'B', 'duplicate_candidate_id': 1}]}
        return {'summaries': [{'entity_id': 0, 'summary': 'Fact B'}]}

    model.generate_response.side_effect = generate
    if operation == 'nodes':
        result = asyncio.run(node_operations.resolve_extracted_nodes(clients, incoming[:2], episode))
        assert result[1] == {node.uuid: candidate.uuid for node, candidate in zip(incoming[:2], candidates)}
    else:
        result = asyncio.run(node_operations.extract_attributes_from_nodes(
            clients, incoming[:2], episode, skip_fact_appending = True, extract_attributes = False,
        )) # extract_attributes_from_nodes
        assert [node.summary for node in result] == ['Fact A', 'Fact B']
    assert history == [['A', 'B'], ['A', 'B'], ['B']]

# malformed single-edge output receives the distinct semantic repair prompt before any state changes
@pytest.mark.parametrize('bad', [None, [], {'duplicate_facts': [True]}, {'duplicate_facts': 'invalid'}])
def test_malformed_single_edge_response_is_corrected(scenario, bad):

    episode, incoming, candidates, model, clients = scenario
    corrected = {'duplicate_facts': [0], 'contradicted_existing_facts': [],
                 'contradicted_invalidation_candidates': []}
    model.generate_response.side_effect = [bad, corrected]
    edge = EntityEdge(group_id = 'company', name = 'RELATION', source_node_uuid = 'a',
                      target_node_uuid = 'b', fact = 'A likes B', created_at = episode.valid_at)
    candidate = edge.model_copy(update = {'uuid': 'canonical', 'fact': 'A enjoys B'})
    result = asyncio.run(edge_operations.resolve_extracted_edge(
        model, edge, [candidate], [], episode, extract_timestamps = False,
    )) # resolve_extracted_edge
    assert result[0].uuid == 'canonical'
    assert [call.kwargs['prompt_name'] for call in model.generate_response.await_args_list] == [
        'dedupe_edges.resolve_edge', 'dedupe_edges.resolve_edge_repair',
    ]

# a failed correction preserves the source and both malformed responses for review
def test_malformed_single_edge_correction_requires_review(scenario):

    episode, incoming, candidates, model, clients = scenario
    model.generate_response.side_effect = [[], {'duplicate_facts': [True]}]
    edge = EntityEdge(group_id = 'company', name = 'RELATION', source_node_uuid = 'a',
                      target_node_uuid = 'b', fact = 'A likes B', created_at = episode.valid_at)
    candidate = edge.model_copy(update = {'uuid': 'canonical', 'fact': 'A enjoys B'})
    with pytest.raises(LLMResponseError) as caught:
        asyncio.run(edge_operations.resolve_extracted_edge(
            model, edge, [candidate], [], episode, extract_timestamps = False,
        )) # resolve_extracted_edge
    assert caught.value.recovery_action == 'review'
    assert caught.value.diagnostics['previous_response'] == []
    assert caught.value.diagnostics['corrected_response'] == {'duplicate_facts': [True]}
    assert candidate.episodes == [] and candidate.invalid_at is None

