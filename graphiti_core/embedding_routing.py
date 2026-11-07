"""Task-local Neo4j embedding generations; canonical source data remains owned by Graphiti."""

import math
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.errors import EmbeddingGenerationChangedError

# the driver identity prevents another Graphiti connection in the same task inheriting this route
GRAPH_EMBEDDING_ROUTES = ContextVar('graphiti_embedding_routes', default = ())

@dataclass(frozen = True)
class GraphEmbeddingRoute:

    generation_id: str
    fingerprint: str
    dimensions: int
    group_ids: tuple[str, ...] | None

    # refuse malformed geometry/ownership before an embedding or graph call is made
    def __post_init__(self):

        if str(UUID(self.generation_id)) != self.generation_id:
            raise ValueError('An embedding generation requires its canonical UUID')
        if (not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64
            or any(value not in '0123456789abcdef' for value in self.fingerprint)): # if
            raise ValueError('An embedding generation requires its exact implementation fingerprint')
        if type(self.dimensions) is not int or self.dimensions < 1:
            raise ValueError('Embedding dimensions must be a positive integer')
        if self.group_ids is not None and (not isinstance(self.group_ids, tuple)
            or any(not isinstance(group, str) or not group for group in self.group_ids)
            or len(set(self.group_ids)) != len(self.group_ids)): # if
            raise ValueError('Generation groups must be explicit, unique namespace identities')

# nested operations restore their parent route, including cancellation and provider failures
@contextmanager
def graph_embedding_route(driver, route):

    if driver.provider != GraphProvider.NEO4J or not isinstance(route, GraphEmbeddingRoute):
        raise ValueError('A generation route requires Neo4j and a checked generation definition')
    token = GRAPH_EMBEDDING_ROUTES.set((*GRAPH_EMBEDDING_ROUTES.get(), (driver, route)))
    try:
        yield route
    finally:
        GRAPH_EMBEDDING_ROUTES.reset(token)

# operations on unrelated clients and tasks retain their own route or the existing legacy behaviour
def current_graph_embedding_route(driver):

    for owner, route in reversed(GRAPH_EMBEDDING_ROUTES.get()):
        if owner is driver:
            return route
    return None

# a tenant generation never widens its namespace restriction through omitted or replacement query groups
def generation_group_ids(driver, group_ids):

    route = current_graph_embedding_route(driver)
    if route is None or route.group_ids is None:
        return group_ids
    if group_ids is None:
        return list(route.group_ids)
    if not set(group_ids) <= set(route.group_ids):
        raise ValueError('The requested graph namespaces are outside the selected embedding generation')
    return group_ids

# use one exact numeric contract for query vectors, loaded vectors and alternate generation embedders
def validate_generation_vector(driver, vector):

    route = current_graph_embedding_route(driver)
    if route is None:
        return vector
    if (not isinstance(vector, list) or len(vector) != route.dimensions
        or any(type(value) not in (int, float) for value in vector)): # if
        raise ValueError('The embedding does not match its selected graph generation')
    try:
        values = [float(value) for value in vector]
    except Exception:
        raise ValueError('The graph embedding contains an unrepresentable value') from None
    if any(not math.isfinite(value) for value in values) or not any(values):
        raise ValueError('The graph embedding must contain finite, nonzero numeric data')
    scale = max(abs(value) for value in values)
    scaled = [value / scale for value in values]
    length = math.sqrt(sum(value * value for value in scaled))
    return [value / length for value in scaled]

# each query joins its isolated vectors to current canonical sources; stale or deleted sources cannot leak through
# this seam is shared by cosine retrieval and MMR vector loading, including the legacy search entry points
def generation_vector_query(driver, kind, alias):

    route = current_graph_embedding_route(driver)
    if route is None:
        raise ValueError('A selected embedding generation is required')
    if (kind, alias) not in (('entity', 'n'), ('edge', 'e'), ('community', 'c')):
        raise ValueError('Only fixed Graphiti source projections may be queried')
    source = f'{alias}.fact' if kind == 'edge' else f'{alias}.name'
    match = (f' MATCH (v:GraphEmbedding {{generation: $embedding_generation, kind: $embedding_kind}}) '
             f'WHERE v.source_uuid = {alias}.uuid AND v.source_text = {source} '
             f'AND ((v.group_id IS NULL AND {alias}.group_id IS NULL) OR v.group_id = {alias}.group_id) '
             'AND size(v.embedding) = $embedding_dimensions ')
    params = {'embedding_generation': route.generation_id, 'embedding_kind': kind,
              'embedding_dimensions': route.dimensions}
    return match, params

# MMR must load vectors from the same generation as query embedding and candidate retrieval
async def generation_embeddings(driver, kind, uuids):

    route = current_graph_embedding_route(driver)
    if route is None:
        raise ValueError('A selected embedding generation is required')
    sources = {'entity': ('MATCH (n:Entity)', 'n'),
               'edge': ('MATCH (:Entity)-[e:RELATES_TO]->(:Entity)', 'e'),
               'community': ('MATCH (c:Community)', 'c')}
    match, alias = sources[kind]
    vectors, params = generation_vector_query(driver, kind, alias)
    groups = generation_group_ids(driver, None)
    query = (match + f' WHERE {alias}.uuid IN $uuids '
             f'AND ($embedding_groups IS NULL OR {alias}.group_id IN $embedding_groups) '
             + vectors + f'RETURN {alias}.uuid AS uuid, v.embedding AS embedding')
    rows, _, _ = await driver.execute_query(query, uuids = uuids, embedding_groups = groups,
                                            routing_ = 'r', **params)
    result = {}
    for row in rows:
        if row['uuid'] in result:
            raise EmbeddingGenerationChangedError('Duplicate canonical graph identities require reconciliation')
        result[row['uuid']] = validate_generation_vector(driver, row['embedding'])
    if set(result) != set(uuids):
        raise EmbeddingGenerationChangedError('Canonical graph sources changed before generation vector loading')
    return result
