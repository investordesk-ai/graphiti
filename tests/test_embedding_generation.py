"""Embedding generation validates its boundary before writing and never invokes an extraction model."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti

# only the public generation method is exercised; source and write acknowledgements are explicit protocol doubles
@pytest.fixture
def generation():

    graph = object.__new__(Graphiti)
    identifier = str(uuid4())
    source_id = str(uuid4())
    stored = {}
    queries = []
    async def query(text, **values):
        queries.append((text, values))
        if 'RETURN g.fingerprint' in text:
            result = [{'fingerprint': 'a' * 64, 'dimensions': 3, 'groups_json': '["group"]', 'version': 2}]
        elif 'copies' in text:
            result = []
        elif 'ORDER BY s.uuid LIMIT' in text:
            result = [{'uuid': source_id, 'text': 'Original entity', 'group_id': 'group'}] if (
                '(s:Entity)' in text and values['after'] is None) else []
        elif 'v.source_sha256 AS' in text:
            result = list(stored.values()) if values['kind'] == 'entity' else []
        elif 'MERGE (v:GraphEmbedding' in text:
            for row in values['rows']:
                stored[row['uuid']] = {'uuid': row['uuid'], 'source_sha256': row['source_sha256'], 'source_text': row['text'],
                                       'embedding': row['embedding']}
            result = [{'written': len(values['rows'])}]
        elif 'AS removed' in text:
            result = [{'removed': 0}]
        elif 'AS total' in text:
            result = [{'total': len(stored)}]
        else:
            result = []
        return result, None, None
    graph.driver = SimpleNamespace(provider = GraphProvider.NEO4J, execute_query = AsyncMock(side_effect = query))
    graph.embedder = SimpleNamespace(create_batch = AsyncMock(return_value = [[0.1, 0.2, 0.3]]))
    graph.llm_client = SimpleNamespace(generate_response = AsyncMock(side_effect = AssertionError('LLM called')))
    return SimpleNamespace(graph = graph, identifier = identifier, queries = queries, stored = stored)

# complete receipts are explicitly unsuitable for cutover and unchanged checkpoints require no new embeddings
@pytest.mark.asyncio
async def test_generation_replays_without_llm_or_embedding_calls(generation):

    graph = generation.graph
    first = await graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3, group_ids = ['group'])
    assert first['points'] == 1 and first['rebuilt'] == 1 and first['verified_for_cutover'] is False
    graph.embedder.create_batch.reset_mock()
    second = await graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3, group_ids = ['group'])
    assert second['reused'] == 1 and second['rebuilt'] == 0
    graph.embedder.create_batch.assert_not_awaited()
    graph.llm_client.generate_response.assert_not_awaited()

# malformed provider shapes and non-finite values cannot become even a partial persisted batch
@pytest.mark.asyncio
@pytest.mark.parametrize('response', [[], [[1, 2]], [[0, 0, 0]], [[True, 0, 1]], [[float('nan'), 0, 1]],
    [[float('inf'), 0, 1]], [[10 ** 1000, 0, 1]], None])
async def test_invalid_generation_vectors(generation, response):

    generation.graph.embedder.create_batch.return_value = response
    with pytest.raises(ValueError):
        await generation.graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3, group_ids = ['group'])
    assert generation.stored == {}

# input validation precedes every database request, including generation ownership and schema checks
@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [{'generation_id': 'not-a-uuid'}, {'fingerprint': 'wrong'}, {'dimensions': 0},
    {'dimensions': True}, {'batch_size': 0}, {'batch_size': 129}, {'group_ids': 'group'}, {'group_ids': [1]}])
async def test_generation_request_validation(generation, changes):

    arguments = {'generation_id': generation.identifier, 'fingerprint': 'a' * 64, 'dimensions': 3}
    arguments.update(changes)
    with pytest.raises(ValueError):
        await generation.graph.rebuild_embedding_generation(**arguments)
    generation.graph.driver.execute_query.assert_not_awaited()

# no unimplemented backend or revoked worker can reach the generation write path
@pytest.mark.asyncio
async def test_generation_never_uses_an_unimplemented_provider(generation):

    generation.graph.driver.provider = 'unsupported'
    with pytest.raises(NotImplementedError):
        await generation.graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3)
    generation.graph.driver.execute_query.assert_not_awaited()

@pytest.mark.asyncio
async def test_generation_observes_cancelled_checkpoint(generation):

    async def checkpoint():
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await generation.graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3,
                                                             checkpoint = checkpoint)
    generation.graph.driver.execute_query.assert_not_awaited()

# final publication scans stay within the public Graphiti API and execute no graph mutations or model requests
@pytest.mark.asyncio
async def test_verify_only_is_read_only(generation):

    graph = generation.graph
    first = await graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3, group_ids = ['group'])
    generation.queries.clear()
    graph.embedder.create_batch.reset_mock()
    result = await graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3,
                                                     group_ids = ['group'], verify_only = True)
    assert result['source_sha256'] == first['source_sha256']
    assert result['points'] == 1
    assert result['rebuilt'] == result['removed'] == 0
    assert all(not any(verb in query.upper() for verb in ('CREATE ', 'MERGE ', 'DELETE ', 'SET '))
               for query, values in generation.queries)
    graph.embedder.create_batch.assert_not_awaited()
    graph.llm_client.generate_response.assert_not_awaited()

# a stale checkpoint requires a fresh build outside the cutover fence, never an implicit repair during final scanning
@pytest.mark.asyncio
async def test_verify_only_defers_missing_checkpoint(generation):

    from graphiti_core.errors import EmbeddingGenerationChangedError
    with pytest.raises(EmbeddingGenerationChangedError):
        await generation.graph.rebuild_embedding_generation(generation.identifier, 'a' * 64, 3,
                                                            group_ids = ['group'], verify_only = True)
    generation.graph.embedder.create_batch.assert_not_awaited()
    assert all('MERGE ' not in query and 'DELETE ' not in query for query, values in generation.queries)
