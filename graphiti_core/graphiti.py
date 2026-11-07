"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import json
import logging
import math
from inspect import isawaitable
from datetime import datetime
from time import time
from uuid import UUID, uuid4

from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.decorators import handle_multiple_group_ids
from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import (
    CommunityEdge,
    Edge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError, EmbeddingGenerationChangedError
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.embedding_routing import GraphEmbeddingRoute, graph_embedding_route
from graphiti_core.embedding_routing import validate_generation_vector
from contextlib import asynccontextmanager
from copy import copy
from graphiti_core.helpers import (
    get_default_group_id,
    semaphore_gather,
    validate_excluded_entity_types,
    validate_group_id,
)
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.namespaces import EdgeNamespace, NodeNamespace
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    Node,
    SagaNode,
    create_entity_node_embeddings,
)
from graphiti_core.prompts.lib import prompt_library
from graphiti_core.prompts.summarize_sagas import SagaSummary
from graphiti_core.search.search import SearchConfig, search
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_mentioned_nodes,
)
from graphiti_core.telemetry import capture_event
from graphiti_core.tracer import Tracer, create_tracer
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    dedupe_nodes_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    extract_edges,
    resolve_extracted_edge,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_attributes_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

logger = logging.getLogger(__name__)

load_dotenv()


# coalesce repeated snapshots of one canonical relationship before persistence so a stale
# invalidation snapshot cannot overwrite newer provenance from the same processing cycle
def _coalesce_entity_edge_updates(
    entity_edges : list[EntityEdge],
) -> list[EntityEdge]:
    """Return one edge update per UUID while preserving provenance and invalidation state."""

    coalesced_edges = []
    edges_by_uuid   = {}

    for edge in entity_edges:

        existing_edge = edges_by_uuid.get(edge.uuid)

        if existing_edge is None:

            # preserve the first canonical snapshot and normalise duplicate provenance ids
            edge.episodes = list(dict.fromkeys(edge.episodes or []))

            edges_by_uuid[edge.uuid] = edge
            coalesced_edges.append(edge)

            continue

        identity_matches = (
            existing_edge.source_node_uuid == edge.source_node_uuid
            and existing_edge.target_node_uuid == edge.target_node_uuid
            and existing_edge.name == edge.name
            and existing_edge.fact == edge.fact
            and existing_edge.group_id == edge.group_id
        ) # identity_matches

        # one UUID must never describe two different canonical relationships
        if not identity_matches:
            raise RuntimeError(
                f"Conflicting EntityEdge snapshots share uuid={edge.uuid!r}"
            )

        # provenance is additive; a stale invalidation snapshot must never erase supporting episodes
        for episode_uuid in edge.episodes or []:

            if episode_uuid not in existing_edge.episodes:
                existing_edge.episodes.append(episode_uuid)

        # if any copy carries invalidation, retain the earliest known world-time invalidation
        if edge.invalid_at is not None:

            existing_invalid_at = ensure_utc(existing_edge.invalid_at)
            candidate_invalid_at = ensure_utc(edge.invalid_at)

            if (
                existing_invalid_at is None
                or (
                    candidate_invalid_at is not None
                    and candidate_invalid_at < existing_invalid_at
                )
            ):
                existing_edge.invalid_at = edge.invalid_at

        # expired_at is knowledge-time invalidation; retain the earliest persisted expiry
        if edge.expired_at is not None:

            existing_expired_at = ensure_utc(existing_edge.expired_at)
            candidate_expired_at = ensure_utc(edge.expired_at)

            if (
                existing_expired_at is None
                or (
                    candidate_expired_at is not None
                    and candidate_expired_at < existing_expired_at
                )
            ):
                existing_edge.expired_at = edge.expired_at

        # retain canonical temporal data when the preferred snapshot does not carry it
        if existing_edge.valid_at is None and edge.valid_at is not None:
            existing_edge.valid_at = edge.valid_at

        if existing_edge.reference_time is None and edge.reference_time is not None:
            existing_edge.reference_time = edge.reference_time

        if existing_edge.fact_embedding is None and edge.fact_embedding is not None:
            existing_edge.fact_embedding = edge.fact_embedding

        # fill absent attributes without allowing a stale copy to replace the preferred values
        existing_attributes = dict(existing_edge.attributes or {})

        for attribute_name, attribute_value in dict(edge.attributes or {}).items():

            if attribute_name not in existing_attributes:
                existing_attributes[attribute_name] = attribute_value

        existing_edge.attributes = existing_attributes

    return coalesced_edges


class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list[CommunityNode]
    community_edges: list[CommunityEdge]


class AddBulkEpisodeResults(BaseModel):
    episodes: list[EpisodicNode]
    episodic_edges: list[EpisodicEdge]
    nodes: list[EntityNode]
    edges: list[EntityEdge]
    communities: list[CommunityNode]
    community_edges: list[CommunityEdge]


class AddTripletResults(BaseModel):
    nodes: list[EntityNode]
    edges: list[EntityEdge]


# added by David Williamson 2026-08-31 and updated 2026-09-04
class SagaSnapshotResults(BaseModel):
    exists: bool
    episode_ids: list[str]
    sequence_edge_count: int
    entity_count: int
    relationship_count: int
    # sequence topology: an unbranched chain over N episodes has N-1 edges, no episode with more
    # than one successor or predecessor, and exactly one start and one end. the edge count alone
    # cannot tell a chain apart from a branch or from two disconnected runs.
    branch_count: int = 0
    merge_count: int = 0
    head_count: int = 0
    tail_count: int = 0


# added by David Williamson 2026-08-31
class EpisodeStructureResults(BaseModel):
    episode_uuid: str
    saga_name: str | None
    previous_episode_uuid: str | None
    next_episode_uuid: str | None


# added by David Williamson 2026-08-31
class EntityInspectionResults(BaseModel):
    uuid: str
    name: str | None
    summary: str | None
    labels: list[str]
    properties: dict


# added by David Williamson 2026-08-31
class EntityRelationshipInspectionResults(BaseModel):
    edge_uuid: str
    relationship_type: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    related_node_uuid: str
    related_node_name: str | None
    episode_ids: list[str]
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None


class Graphiti:
    
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
        graph_driver: GraphDriver | None = None,
        max_coroutines: int | None = None,
        tracer: Tracer | None = None,
        trace_span_prefix: str = 'graphiti',
    ):
        """
        Initialize a Graphiti instance.

        This constructor sets up a connection to a graph database and initializes
        the LLM client for natural language processing tasks.

        Parameters
        ----------
        uri : str
            The URI of the Neo4j database.
        user : str
            The username for authenticating with the Neo4j database.
        password : str
            The password for authenticating with the Neo4j database.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for embedding tasks.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking tasks.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the raw content of episodes. Defaults to True.
        graph_driver : GraphDriver | None, optional
            An instance of GraphDriver for database operations.
            If not provided, a default Neo4jDriver will be initialized.
        max_coroutines : int | None, optional
            The maximum number of concurrent operations allowed. Overrides SEMAPHORE_LIMIT set in the environment.
            If not set, the Graphiti default is used.
        tracer : Tracer | None, optional
            An OpenTelemetry tracer instance for distributed tracing. If not provided, tracing is disabled (no-op).
        trace_span_prefix : str, optional
            Prefix to prepend to all span names. Defaults to 'graphiti'.

        Returns
        -------
        None

        Notes
        -----
        This method establishes a connection to a graph database (Neo4j by default) using the provided
        credentials. It also sets up the LLM client, either using the provided client
        or by creating a default OpenAIClient.

        The default database name is defined during the driver’s construction. If a different database name
        is required, it should be specified in the URI or set separately after
        initialization.

        The OpenAI API key is expected to be set in the environment variables.
        Make sure to set the OPENAI_API_KEY environment variable before initializing
        Graphiti if you're using the default OpenAIClient.
        """

        if graph_driver:
            self.driver = graph_driver
        else:
            if uri is None:
                raise ValueError('uri must be provided when graph_driver is None')
            self.driver = Neo4jDriver(uri, user, password)

        self.store_raw_episode_content = store_raw_episode_content
        self.max_coroutines = max_coroutines
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient()
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder()
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient()

        # Initialize tracer
        self.tracer = create_tracer(tracer, trace_span_prefix)

        # Set tracer on clients
        self.llm_client.set_tracer(self.tracer)

        self.clients = GraphitiClients(
            driver=self.driver,
            llm_client=self.llm_client,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            tracer=self.tracer,
        )

        # Initialize namespace API (graphiti.nodes.entity.save(), etc.)
        self.nodes = NodeNamespace(self.driver, self.embedder)
        self.edges = EdgeNamespace(self.driver, self.embedder)

        # Capture telemetry event
        self._capture_initialization_telemetry()

    def _capture_initialization_telemetry(self):
        """Capture telemetry event for Graphiti initialization."""
        try:
            # Detect provider types from class names
            llm_provider = self._get_provider_type(self.llm_client)
            embedder_provider = self._get_provider_type(self.embedder)
            reranker_provider = self._get_provider_type(self.cross_encoder)
            database_provider = self._get_provider_type(self.driver)

            properties = {
                'llm_provider': llm_provider,
                'embedder_provider': embedder_provider,
                'reranker_provider': reranker_provider,
                'database_provider': database_provider,
            }

            capture_event('graphiti_initialized', properties)
        except Exception:
            # Silently handle telemetry errors
            pass

    @property
    def token_tracker(self):
        """Access the LLM client's token usage tracker.

        Returns the TokenUsageTracker from the LLM client, which can be used to:
        - Get token usage by prompt type: tracker.get_usage()
        - Get total token usage: tracker.get_total_usage()
        - Print a formatted summary: tracker.print_summary()
        - Reset tracking: tracker.reset()
        """
        return self.llm_client.token_tracker

    def _get_provider_type(self, client) -> str:
        """Get provider type from client class name."""
        if client is None:
            return 'none'

        class_name = client.__class__.__name__.lower()

        # LLM providers
        if 'openai' in class_name:
            return 'openai'
        elif 'azure' in class_name:
            return 'azure'
        elif 'anthropic' in class_name:
            return 'anthropic'
        elif 'crossencoder' in class_name:
            return 'crossencoder'
        elif 'gemini' in class_name:
            return 'gemini'
        elif 'groq' in class_name:
            return 'groq'
        # Database providers
        elif 'neo4j' in class_name:
            return 'neo4j'
        elif 'falkor' in class_name:
            return 'falkordb'
        # Embedder providers
        elif 'voyage' in class_name:
            return 'voyage'
        else:
            return 'unknown'

    async def close(self):
        """
        Close the connection to the Neo4j database.

        This method safely closes the driver connection to the Neo4j database.
        It should be called when the Graphiti instance is no longer needed or
        when the application is shutting down.

        Parameters
        ----------
        self

        Returns
        -------
        None

        Notes
        -----
        It's important to close the driver connection to release system resources
        and ensure that all pending transactions are completed or rolled back.
        This method should be called as part of a cleanup process, potentially
        in a context manager or a shutdown hook.

        Example:
            graphiti = Graphiti(uri, user, password)
            try:
                # Use graphiti...
            finally:
                graphiti.close()
        """
        await self.driver.close()

    # added by David Williamson 2026-08-31
    async def health_check(self) -> bool:
        """Return True when the graph database accepts a read query."""

        try:
            records, _, _ = await self.driver.execute_query(
                'RETURN 1 AS ok',
                routing_='r',
            )

            return bool(records)

        except Exception as e:
            logger.warning('Graph database health check failed: %s', e)
            return False

    # added by David Williamson 2026-08-31
    async def initialise(self) -> None:
        """Wait for any pending graph database initialisation to complete."""

        init_task = getattr(self.driver, "_init_task", None)

        if init_task is not None:
            await init_task

    # added by David Williamson 2026-08-31
    async def clear(self) -> None:
        """Delete all graph data while retaining indexes and constraints."""

        await self.driver.graph_ops.clear_data(self.driver)

    # rebuild derived vectors through Graphiti without touching canonical graph data or its active embeddings
    # expose bounded source identities and provenance only; graph ownership audits never need document or vector content
    async def get_group_ownership_page(self, group_id: str, *, after: str | None = None, limit: int = 128) -> list[dict]:

        if self.driver.provider != GraphProvider.NEO4J:
            raise NotImplementedError('Graph ownership inventory currently requires Neo4j')
        if (not isinstance(group_id, str) or not 1 <= len(group_id) <= 512
            or any(not (value.isascii() and (value.isalnum() or value in '_-')) for value in group_id)): # if
            raise ValueError('Ownership inventory requires an explicit safe namespace')
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError('Ownership inventory pages must contain between 1 and 128 records')
        if after is not None and (not isinstance(after, str) or not after):
            raise ValueError('Ownership inventory cursors must be non-empty record identities')
        rows, _, _ = await self.driver.execute_query(
            """CALL {
                   MATCH (n {group_id: $group}) WHERE NOT n:GraphEmbedding AND NOT n:GraphEmbeddingGeneration
                   RETURN 'n:' + elementId(n) AS key, 'node' AS kind, n.uuid AS uuid, labels(n) AS labels,
                          n.group_id AS group_id, CASE WHEN n:Saga THEN n.name ELSE NULL END AS artifact_id,
                          NULL AS relation, NULL AS source_id, NULL AS target_id,
                          NULL AS source_group, NULL AS target_group, [] AS episodes
                   UNION ALL
                   MATCH (s)-[r]->(t)
                    WHERE (r.group_id = $group OR s.group_id = $group OR t.group_id = $group)
                      AND NOT s:GraphEmbedding AND NOT t:GraphEmbedding
                      AND NOT s:GraphEmbeddingGeneration AND NOT t:GraphEmbeddingGeneration
                   RETURN 'r:' + elementId(r) AS key, 'edge' AS kind, r.uuid AS uuid, [] AS labels,
                          r.group_id AS group_id, NULL AS artifact_id, type(r) AS relation,
                          s.uuid AS source_id, t.uuid AS target_id, s.group_id AS source_group,
                          t.group_id AS target_group, coalesce(r.episodes, []) AS episodes
               }
               WITH * WHERE $after IS NULL OR key > $after
               RETURN * ORDER BY key LIMIT $limit""",
            group=group_id, after=after, limit=limit, routing_='w',
        ) # execute_query
        result = [dict(row) for row in rows]
        keys = [row['key'] for row in result]
        if (len(result) > limit or any(not isinstance(key, str) or not key for key in keys)
            or keys != sorted(set(keys)) or (after is not None and any(key <= after for key in keys))): # if
            raise ValueError('Ownership inventory returned an invalid keyset page')
        for row in result:
            if isinstance(row['labels'], list):
                row['labels'] = sorted(row['labels'])
        return result

    # new tenant namespaces must contain no graph records, including auxiliary Saga and relationship records
    async def is_group_namespace_empty(self, group_id: str) -> bool:

        if self.driver.provider != GraphProvider.NEO4J:
            raise NotImplementedError('Graph namespace verification currently requires Neo4j')
        if (not isinstance(group_id, str) or not 1 <= len(group_id) <= 512
            or any(not (value.isascii() and (value.isalnum() or value in '_-')) for value in group_id)): # if
            raise ValueError('A graph namespace requires an explicit safe group identity')
        rows, _, _ = await self.driver.execute_query(
            'RETURN NOT EXISTS { MATCH (n {group_id: $group}) } '
            'AND NOT EXISTS { MATCH ()-[r {group_id: $group}]->() } AS empty',
            group=group_id, routing_='r',
        ) # execute_query
        if len(rows) != 1 or type(rows[0]['empty']) is not bool:
            raise ValueError('The graph namespace verification returned an invalid receipt')
        return rows[0]['empty']

    # inventory canonical namespaces through the library without reading vectors or changing graph records
    async def get_embedding_group_ids(self, *, after: str | None = None, limit: int = 64) -> list[str]:

        if self.driver.provider != GraphProvider.NEO4J:
            raise NotImplementedError('Graph namespace inventory currently requires Neo4j')
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError('Graph inventory pages must contain between 1 and 128 namespaces')
        if after is not None and (not isinstance(after, str) or not after):
            raise ValueError('Graph inventory cursors must be non-empty namespace identities')
        rows, _, _ = await self.driver.execute_query(
            """CALL {
                   MATCH (s) WHERE s:Entity OR s:Community OR s:Episodic RETURN s.group_id AS group_id
                   UNION
                   MATCH ()-[s:RELATES_TO]->() RETURN s.group_id AS group_id
               }
               WITH DISTINCT group_id
               WHERE group_id IS NULL OR ($after IS NULL OR group_id > $after)
               RETURN group_id ORDER BY group_id LIMIT $limit""",
            after=after, limit=limit, routing_='r',
        ) # execute_query
        groups = [row['group_id'] for row in rows]
        if any(not isinstance(group, str) or not group for group in groups):
            raise ValueError('Canonical graph data has an absent or malformed namespace identity')
        if groups != sorted(set(groups)) or (after is not None and any(group <= after for group in groups)):
            raise ValueError('Graph namespace inventory did not preserve its bounded keyset order')
        return groups

    # bind one operation to an existing immutable generation; other tasks keep their own model and vector space
    @asynccontextmanager
    async def embedding_generation(
        self, generation_id: str, fingerprint: str, dimensions: int, *,
        embedder: EmbedderClient, group_ids: list[str] | None = None, write: bool = False,
    ): # embedding_generation

        if self.driver.provider != GraphProvider.NEO4J:
            raise NotImplementedError('Graph embedding generation routing requires Neo4j')
        if not isinstance(embedder, EmbedderClient) or type(write) is not bool:
            raise ValueError('A generation requires its explicit embedder and read/write mode')
        if group_ids is not None and (not isinstance(group_ids, list)
            or any(not isinstance(value, str) or not value for value in group_ids)): # if
            raise ValueError('Generation scope requires explicit graph namespace identities')
        # validate caller geometry before touching the graph; returned metadata must match it exactly
        GraphEmbeddingRoute(generation_id, fingerprint, dimensions,
                            tuple(sorted(set(group_ids))) if group_ids is not None else None)
        owners, _, _ = await self.driver.execute_query(
            'MATCH (g:GraphEmbeddingGeneration {uuid: $generation}) '
            'RETURN g.fingerprint AS fingerprint, g.dimensions AS dimensions, '
            'g.groups_json AS groups_json, g.projection_version AS version',
            generation = generation_id, routing_ = 'r',
        ) # execute_query
        if len(owners) != 1 or (owners[0]['fingerprint'], owners[0]['dimensions'], owners[0]['version']) != (
            fingerprint, dimensions, 2
        ): # if
            raise ValueError('The requested graph generation is missing or belongs to another implementation')
        groups = json.loads(owners[0]['groups_json'])
        if groups is not None and (not isinstance(groups, list)
            or any(not isinstance(value, str) or not value for value in groups)): # if
            raise ValueError('The stored generation namespace restriction is invalid')
        if group_ids is not None and groups is not None and not set(group_ids) <= set(groups):
            if not write:
                raise ValueError('The requested namespaces are outside the stored embedding generation')
            for attempt in range(3):
                additions = sorted(set(group_ids) - set(groups))
                if not additions:
                    break
                updated = sorted(set(groups) | set(group_ids))
                changed, _, _ = await self.driver.execute_query(
                    'MATCH (g:GraphEmbeddingGeneration {uuid: $generation, fingerprint: $fingerprint, '
                    'dimensions: $dimensions, projection_version: 2}) WHERE g.groups_json = $previous '
                    'AND NOT EXISTS { MATCH (n) WHERE n.group_id IN $additions } '
                    'AND NOT EXISTS { MATCH ()-[r]->() WHERE r.group_id IN $additions } '
                    'SET g.groups_json = $updated RETURN g.groups_json AS groups_json',
                    generation=generation_id, fingerprint=fingerprint, dimensions=dimensions,
                    previous=owners[0]['groups_json'], additions=additions,
                    updated=json.dumps(updated, ensure_ascii=False),
                ) # execute_query
                if len(changed) == 1:
                    if changed[0]['groups_json'] != json.dumps(updated, ensure_ascii=False):
                        raise ValueError('Graph namespace admission returned an inconsistent acknowledgement')
                    groups = updated
                    break
                owners, _, _ = await self.driver.execute_query(
                    'MATCH (g:GraphEmbeddingGeneration {uuid: $generation}) '
                    'RETURN g.fingerprint AS fingerprint, g.dimensions AS dimensions, '
                    'g.groups_json AS groups_json, g.projection_version AS version',
                    generation=generation_id, routing_='r',
                ) # execute_query
                if len(owners) != 1 or (owners[0]['fingerprint'], owners[0]['dimensions'], owners[0]['version']) != (
                    fingerprint, dimensions, 2
                ): # if
                    raise EmbeddingGenerationChangedError('Graph generation ownership changed during namespace admission')
                groups = json.loads(owners[0]['groups_json'])
                if not isinstance(groups, list) or any(not isinstance(group, str) or not group for group in groups):
                    raise ValueError('The stored graph namespace scope became invalid')
            if not set(group_ids) <= set(groups):
                raise EmbeddingGenerationChangedError('A populated or changing graph namespace requires a verified rebuild')
        # a subset is a read restriction; rebuilding must always retain the generation's full immutable scope
        selected_groups = group_ids if group_ids is not None else groups
        route = GraphEmbeddingRoute(generation_id, fingerprint, dimensions,
            tuple(sorted(set(selected_groups))) if selected_groups is not None else None)
        driver = self.driver

        # every Graphiti entry point receives the same alternate embedder and validates its numeric geometry
        class RoutedEmbedder(EmbedderClient):

            def __getattr__(self, name):
                return getattr(embedder, name)

            async def create(self, input_data):
                from graphiti_core.embedding_routing import current_graph_embedding_route
                if current_graph_embedding_route(driver) != route:
                    raise ValueError('The generation client escaped its operation scope')
                return validate_generation_vector(driver, await embedder.create(input_data))

            async def create_batch(self, texts):
                from graphiti_core.embedding_routing import current_graph_embedding_route
                if current_graph_embedding_route(driver) != route:
                    raise ValueError('The generation client escaped its operation scope')
                vectors = await embedder.create_batch(texts)
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise ValueError('The generation embedder omitted source records')
                return [validate_generation_vector(driver, value) for value in vectors]

        view = copy(self)
        view.embedder = RoutedEmbedder()
        view.clients = self.clients.model_copy(update = {'embedder': view.embedder})
        view.nodes = NodeNamespace(driver, view.embedder)
        view.edges = EdgeNamespace(driver, view.embedder)
        with graph_embedding_route(driver, route):
            yield view
            if write:
                # acknowledge a successful episode operation only after its current source vectors are reconciled
                await view.rebuild_embedding_generation(generation_id, fingerprint, dimensions,
                    embedder = view.embedder, group_ids = groups)

    async def rebuild_embedding_generation(
        self, generation_id: str, fingerprint: str, dimensions: int, *,
        embedder: EmbedderClient | None = None, group_ids: list[str] | None = None,
        batch_size: int = 64, checkpoint = None, verify_only: bool = False,
    ) -> dict: # rebuild_embedding_generation

        if type(verify_only) is not bool:
            raise ValueError('Graph verification mode must be explicit')
        if self.driver.provider != GraphProvider.NEO4J:
            raise NotImplementedError('Parallel graph embedding generations currently require Neo4j')
        generation_id = str(UUID(str(generation_id)))
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(value not in '0123456789abcdef' for value in fingerprint)): # if
            raise ValueError('A graph generation requires its exact approved implementation fingerprint')
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError('Graph embedding dimensions must be a positive integer')
        if type(batch_size) is not int or not 1 <= batch_size <= 128:
            raise ValueError('Graph embedding batches must contain between 1 and 128 records')
        if group_ids is not None:
            if not isinstance(group_ids, list) or any(not isinstance(value, str) for value in group_ids):
                raise ValueError('Graph group restrictions must be an explicit list of strings')
            group_ids = sorted(set(group_ids))
        groups_json = json.dumps(group_ids, ensure_ascii = False)
        embedder = self.embedder if embedder is None else embedder

        # callers may supply either an async lease check or a synchronous cancellation boundary
        async def check():
            if checkpoint is not None:
                result = checkpoint()
                if isawaitable(result):
                    await result

        # validate both provider output and persisted checkpoints; cosine scaling preserves vector direction
        def vector_values(vector):
            if (not isinstance(vector, list) or len(vector) != dimensions
                or any(type(value) not in (int, float) for value in vector)): # if
                raise ValueError('The graph embedder returned invalid numeric geometry')
            try:
                values = [float(value) for value in vector]
            except Exception:
                raise ValueError('The graph embedder returned an unrepresentable value') from None
            if any(not math.isfinite(value) for value in values):
                raise ValueError('The graph embedding contains a non-finite value')
            scale = max(abs(value) for value in values)
            if scale == 0:
                raise ValueError('The graph embedding is zero')
            scaled = [value / scale for value in values]
            length = math.sqrt(sum(value * value for value in scaled))
            return [value / length for value in scaled]

        # names and facts use the stored text consumed by bulk generation; communities retain their public projection
        def source_snapshot(row, kind):
            uuid, text, group = row['uuid'], row['text'], row['group_id']
            if not isinstance(uuid, str) or not uuid or not isinstance(text, str) or not text.strip():
                raise ValueError(f'Graph {kind} source {uuid!r} has no complete identity and embedding text')
            if group is not None and not isinstance(group, str):
                raise ValueError(f'Graph {kind} source {uuid!r} has an invalid group identity')
            projected = text.replace('\n', ' ') if kind == 'community' else text
            document = {'kind': kind, 'uuid': uuid, 'group_id': group, 'text': text, 'projected': projected}
            encoded = json.dumps(document, sort_keys = True, ensure_ascii = False, allow_nan = False)
            return projected, hashlib.sha256(encoded.encode('utf-8')).hexdigest()

        await check()
        if not verify_only:
            await self.driver.execute_query(
                'CREATE CONSTRAINT graphiti_embedding_generation_uuid IF NOT EXISTS '
                'FOR (g:GraphEmbeddingGeneration) REQUIRE g.uuid IS UNIQUE',
            ) # execute_query
            await self.driver.execute_query(
                'CREATE CONSTRAINT graphiti_embedding_vector_uuid IF NOT EXISTS '
                'FOR (v:GraphEmbedding) REQUIRE v.uuid IS UNIQUE',
            ) # execute_query
            await self.driver.execute_query(
                'CREATE INDEX graphiti_embedding_generation IF NOT EXISTS '
                'FOR (v:GraphEmbedding) ON (v.generation)',
            ) # execute_query
        owners, _, _ = await self.driver.execute_query(
            ("MATCH (g:GraphEmbeddingGeneration {uuid: $generation}) " if verify_only else
             "MERGE (g:GraphEmbeddingGeneration {uuid: $generation}) "
             "ON CREATE SET g.fingerprint = $fingerprint, g.dimensions = $dimensions, "
             "g.groups_json = $groups_json, g.projection_version = 2 ") +
            'RETURN g.fingerprint AS fingerprint, g.dimensions AS dimensions, '
            'g.groups_json AS groups_json, g.projection_version AS version',
            generation=generation_id, fingerprint=fingerprint, dimensions=dimensions, groups_json=groups_json,
        ) # execute_query
        if len(owners) != 1 or dict(owners[0]) != {
            'fingerprint': fingerprint, 'dimensions': dimensions, 'groups_json': groups_json, 'version': 2,
        }: # if
            raise ValueError('The graph destination belongs to another model, geometry or group restriction')

        # query fragments are fixed library-owned label/property choices, never caller-supplied Cypher
        sources = (
            ('entity', 'MATCH (s:Entity)', 's.name'),
            ('edge', 'MATCH (:Entity)-[s:RELATES_TO]->(:Entity)', 's.fact'),
            ('community', 'MATCH (s:Community)', 's.name'),
        ) # sources
        rebuilt, reused, removed = 0, 0, 0
        counts = {}
        digest = hashlib.sha256()
        for kind, match, property_name in sources:
            invalid, _, _ = await self.driver.execute_query(
                match + ' WHERE ($groups IS NULL OR s.group_id IN $groups) '
                'WITH s.uuid AS uuid, count(*) AS copies '
                "WHERE uuid IS NULL OR toString(uuid) <> uuid OR uuid = '' OR copies <> 1 RETURN uuid LIMIT 1",
                groups=group_ids, routing_='r',
            ) # execute_query
            if invalid:
                raise ValueError(f'Graph {kind} sources contain a missing or duplicate canonical identity')
        for phase in (('verify',) if verify_only else ('build', 'verify')):
            for kind, match, property_name in sources:
                cursor = None
                verified = 0
                while True:
                    await check()
                    page, _, _ = await self.driver.execute_query(
                        match + ' WHERE ($groups IS NULL OR s.group_id IN $groups) '
                        'AND ($after IS NULL OR s.uuid > $after) '
                        f'RETURN s.uuid AS uuid, s.group_id AS group_id, {property_name} AS text '
                        'ORDER BY s.uuid LIMIT $limit',
                        groups=group_ids, after=cursor, limit=batch_size, routing_='r',
                    ) # execute_query
                    if not page:
                        break
                    saved, _, _ = await self.driver.execute_query(
                        'MATCH (v:GraphEmbedding) WHERE v.uuid IN $target_ids '
                        'AND v.generation = $generation AND v.kind = $kind RETURN v.source_uuid AS uuid, '
                        'v.source_sha256 AS source_sha256, v.source_text AS source_text, v.embedding AS embedding',
                        generation=generation_id, kind=kind,
                        target_ids=[f'{generation_id}:{kind}:{row["uuid"]}' for row in page], routing_='r',
                    ) # execute_query
                    existing = {row['uuid']: row for row in saved}
                    pending = []
                    for row in page:
                        text, source_hash = source_snapshot(row, kind)
                        stored = existing.get(row['uuid'])
                        valid = (stored is not None and stored['source_sha256'] == source_hash
                                 and stored.get('source_text') == row['text'])
                        if valid:
                            try:
                                vector_values(stored['embedding'])
                            except ValueError:
                                valid = False
                        if phase == 'verify':
                            if not valid:
                                raise EmbeddingGenerationChangedError('Graph sources changed during rebuilding; resume')
                            digest.update(json.dumps([kind, row['uuid'], source_hash]).encode('utf-8'))
                            digest.update(b'\n')
                            verified += 1
                        elif valid:
                            reused += 1
                        else:
                            pending.append((row, text, source_hash))
                    if pending:
                        vectors = await embedder.create_batch([text for row, text, source_hash in pending])
                        if not isinstance(vectors, list) or len(vectors) != len(pending):
                            raise ValueError('The graph embedder omitted one or more source records')
                        values = []
                        for (row, text, source_hash), vector in zip(pending, vectors, strict=True):
                            values.append({'uuid': row['uuid'], 'group_id': row['group_id'], 'text': row['text'],
                                'target_uuid': f'{generation_id}:{kind}:{row["uuid"]}', 'source_sha256': source_hash,
                                'embedding': vector_values(vector)})
                        await check()
                        written, _, _ = await self.driver.execute_query(
                            'MATCH (g:GraphEmbeddingGeneration {uuid: $generation, fingerprint: $fingerprint}) '
                            'UNWIND $rows AS row ' + match + ' WHERE s.uuid = row.uuid '
                            f'AND {property_name} = row.text '
                            'AND ((s.group_id IS NULL AND row.group_id IS NULL) OR s.group_id = row.group_id) '
                            'MERGE (v:GraphEmbedding {uuid: row.target_uuid}) '
                            'SET v.generation = $generation, v.kind = $kind, v.source_uuid = row.uuid, '
                            'v.group_id = row.group_id, v.source_sha256 = row.source_sha256, v.source_text = row.text, '
                            'v.embedding = row.embedding '
                            'MERGE (g)-[:HAS_EMBEDDING]->(v) RETURN count(v) AS written',
                            generation=generation_id, fingerprint=fingerprint, rows=values, kind=kind,
                        ) # execute_query
                        if len(written) != 1 or written[0]['written'] != len(values):
                            raise EmbeddingGenerationChangedError('Graph sources changed during inference; resume')
                        rebuilt += len(values)
                    cursor = page[-1]['uuid']
                if phase == 'build':
                    # prune only this isolated generation when canonical sources have been removed or moved out of scope
                    while True:
                        await check()
                        pruned, _, _ = await self.driver.execute_query(
                            'MATCH (v:GraphEmbedding {generation: $generation, kind: $kind}) '
                            'WHERE NOT EXISTS { ' + match + ' WHERE s.uuid = v.source_uuid '
                            'AND ($groups IS NULL OR s.group_id IN $groups) } '
                            'WITH v LIMIT $limit DETACH DELETE v RETURN count(*) AS removed',
                            generation=generation_id, kind=kind, groups=group_ids, limit=batch_size,
                        ) # execute_query
                        amount = pruned[0]['removed'] if pruned else 0
                        removed += amount
                        if amount == 0:
                            break
                else:
                    counts[kind] = verified
        stored, _, _ = await self.driver.execute_query(
            'MATCH (v:GraphEmbedding {generation: $generation}) RETURN count(v) AS total',
            generation=generation_id, routing_='r',
        ) # execute_query
        points = sum(counts.values())
        if len(stored) != 1 or stored[0]['total'] != points:
            raise EmbeddingGenerationChangedError('Graph membership changed during verification; resume')
        logger.info(f'[REBUILD_EMBEDDING_GENERATION] Built generation={generation_id}; records={points}')
        return {'generation_id': generation_id, 'fingerprint': fingerprint, 'dimensions': dimensions,
                'counts': counts, 'points': points, 'rebuilt': rebuilt, 'reused': reused, 'removed': removed,
                'source_sha256': digest.hexdigest(), 'status': 'built', 'verified_for_cutover': False}

    # added by David Williamson 2026-08-31
    async def get_saga_snapshot(self, saga_name: str) -> SagaSnapshotResults:
        """Return structural and semantic counts for one Saga."""

        records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {name: $saga_name})
            OPTIONAL MATCH (s)-[:HAS_EPISODE]->(episode:Episodic)
            WITH s, collect(DISTINCT episode.uuid) AS episode_ids

            OPTIONAL MATCH
                (source_episode:Episodic)-[sequence:NEXT_EPISODE]->(target_episode:Episodic)
            WHERE source_episode.uuid IN episode_ids
              AND target_episode.uuid IN episode_ids

            WITH s, episode_ids, count(DISTINCT sequence) AS sequence_edge_count

            OPTIONAL MATCH (mentioned_episode:Episodic)-[:MENTIONS]->(entity:Entity)
            WHERE mentioned_episode.uuid IN episode_ids

            WITH s,
                 episode_ids,
                 sequence_edge_count,
                 collect(DISTINCT entity.uuid) AS entity_ids

            OPTIONAL MATCH (:Entity)-[edge:RELATES_TO]->(:Entity)
            WHERE any(
                episode_id IN episode_ids
                WHERE episode_id IN coalesce(edge.episodes, [])
            )

            RETURN episode_ids,
                   sequence_edge_count,
                   entity_ids,
                   collect(DISTINCT edge.uuid) AS edge_ids
            """,
            saga_name=saga_name,
            routing_='r',
        )

        if not records:
            return SagaSnapshotResults(
                exists=False,
                episode_ids=[],
                sequence_edge_count=0,
                entity_count=0,
                relationship_count=0,
            )

        record = records[0]

        episode_ids = [
            str(value)
            for value in record.get('episode_ids') or []
            if value
        ]
        entity_ids = [
            str(value)
            for value in record.get('entity_ids') or []
            if value
        ]
        edge_ids = [
            str(value)
            for value in record.get('edge_ids') or []
            if value
        ]

        # sequence topology is a second query on purpose: folding per-episode degrees into the
        # aggregate above makes it unreadable, and this runs on the admin audit path, not on ingest
        topology_records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {name: $saga_name})-[:HAS_EPISODE]->(episode:Episodic)
            WITH collect(DISTINCT episode.uuid) AS episode_ids

            UNWIND episode_ids AS episode_id
            MATCH (episode:Episodic {uuid: episode_id})

            OPTIONAL MATCH (episode)-[out_edge:NEXT_EPISODE]->(next_episode:Episodic)
            WHERE next_episode.uuid IN episode_ids

            OPTIONAL MATCH (previous_episode:Episodic)-[in_edge:NEXT_EPISODE]->(episode)
            WHERE previous_episode.uuid IN episode_ids

            WITH episode_id,
                 count(DISTINCT out_edge) AS out_degree,
                 count(DISTINCT in_edge) AS in_degree

            RETURN count(CASE WHEN out_degree > 1 THEN 1 END) AS branch_count,
                   count(CASE WHEN in_degree > 1 THEN 1 END) AS merge_count,
                   count(CASE WHEN in_degree = 0 THEN 1 END) AS head_count,
                   count(CASE WHEN out_degree = 0 THEN 1 END) AS tail_count
            """,
            saga_name=saga_name,
            routing_='r',
        )

        topology = topology_records[0] if topology_records else {}

        return SagaSnapshotResults(
            exists=True,
            episode_ids=episode_ids,
            sequence_edge_count=int(record.get('sequence_edge_count') or 0),
            entity_count=len(entity_ids),
            relationship_count=len(edge_ids),
            branch_count=int(topology.get('branch_count') or 0),
            merge_count=int(topology.get('merge_count') or 0),
            head_count=int(topology.get('head_count') or 0),
            tail_count=int(topology.get('tail_count') or 0),
        )

    # added by David Williamson 2026-08-31
    async def get_episode_structure(
        self,
        episode_uuid: str,
    ) -> EpisodeStructureResults | None:
        """Return Saga and sequence information for one episode."""

        records, _, _ = await self.driver.execute_query(
            """
            MATCH (current:Episodic {uuid: $episode_uuid})
            OPTIONAL MATCH (s:Saga)-[:HAS_EPISODE]->(current)
            OPTIONAL MATCH (previous:Episodic)-[:NEXT_EPISODE]->(current)
            OPTIONAL MATCH (current)-[:NEXT_EPISODE]->(next:Episodic)
            RETURN current.uuid AS episode_uuid,
                   s.name AS saga_name,
                   previous.uuid AS previous_episode_uuid,
                   next.uuid AS next_episode_uuid
            """,
            episode_uuid=episode_uuid,
            routing_='r',
        )

        if not records:
            return None

        record = records[0]

        return EpisodeStructureResults(
            episode_uuid=str(record['episode_uuid']),
            saga_name=record.get('saga_name'),
            previous_episode_uuid=record.get('previous_episode_uuid'),
            next_episode_uuid=record.get('next_episode_uuid'),
        )

    # added by David Williamson 2026-08-31
    async def list_entities(
        self,
        limit: int = 50,
    ) -> list[EntityInspectionResults]:
        """Return Entity nodes for graph inspection."""

        records, _, _ = await self.driver.execute_query(
            """
            MATCH (entity:Entity)
            RETURN entity.uuid AS uuid,
                   entity.name AS name,
                   entity.summary AS summary,
                   labels(entity) AS labels,
                   properties(entity) AS properties
            ORDER BY coalesce(entity.name, '')
            LIMIT $limit
            """,
            limit=limit,
            routing_='r',
        )

        return [
            EntityInspectionResults(
                uuid=str(record['uuid']),
                name=record.get('name'),
                summary=record.get('summary'),
                labels=list(record.get('labels') or []),
                properties=dict(record.get('properties') or {}),
            )
            for record in records
        ]

    # added by David Williamson 2026-08-31
    async def get_entity_relationships(
        self,
        node_uuid: str,
    ) -> list[EntityRelationshipInspectionResults]:
        """Return semantic relationships connected to one Entity."""

        edges = await EntityEdge.get_by_node_uuid(
            self.driver,
            node_uuid,
        )

        related_node_uuids = [
            (
                edge.target_node_uuid
                if edge.source_node_uuid == node_uuid
                else edge.source_node_uuid
            )
            for edge in edges
        ]

        related_nodes = await EntityNode.get_by_uuids(
            self.driver,
            list(set(related_node_uuids)),
        )

        related_nodes_by_uuid = {
            str(node.uuid): node
            for node in related_nodes
        }

        results = []

        for edge, related_node_uuid in zip(
            edges,
            related_node_uuids,
            strict=True,
        ):
            related_node = related_nodes_by_uuid.get(str(related_node_uuid))

            results.append(
                EntityRelationshipInspectionResults(
                    edge_uuid=str(edge.uuid),
                    relationship_type=edge.name,
                    fact=edge.fact,
                    source_node_uuid=str(edge.source_node_uuid),
                    target_node_uuid=str(edge.target_node_uuid),
                    related_node_uuid=str(related_node_uuid),
                    related_node_name=(
                        related_node.name
                        if related_node is not None
                        else None
                    ),
                    episode_ids=[
                        str(episode_id)
                        for episode_id in edge.episodes
                    ],
                    valid_at=edge.valid_at,
                    invalid_at=edge.invalid_at,
                    created_at=edge.created_at,
                    expired_at=edge.expired_at,
                )
            )

        results.sort(
            key=lambda result: (
                result.relationship_type or "",
                result.related_node_name or "",
            )
        )

        return results

    # added by David Williamson 2026-08-31
    async def remove_saga_if_empty(self, saga_name: str) -> None:
        """Remove a Saga when it no longer contains any episodes."""

        await self.driver.execute_query(
            """
            MATCH (s:Saga {name: $saga_name})
            WHERE NOT (s)-[:HAS_EPISODE]->(:Episodic)
            DETACH DELETE s
            """,
            saga_name=saga_name,
        )

    async def _get_or_create_saga(
        self, saga_name: str, group_id: str, created_at: datetime
    ) -> SagaNode:
        """
        Get an existing saga by name or create a new one.

        Parameters
        ----------
        saga_name : str
            The name of the saga.
        group_id : str
            The group id for the saga.
        created_at : datetime
            Timestamp to stamp on a newly created saga. Callers should pass the
            originating episode's reference time (``valid_at``) rather than the
            current wall-clock time so the saga's ``created_at`` reflects the
            episode it was minted from.

        Returns
        -------
        SagaNode
            The existing or newly created saga node.
        """
        from graphiti_core.helpers import parse_db_date

        records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {name: $name, group_id: $group_id})
            RETURN s.uuid AS uuid, s.name AS name, s.group_id AS group_id, s.created_at AS created_at
            """,
            name=saga_name,
            group_id=group_id,
            routing_='r',
        )

        # edited by David Williamson 2026-09-01
        if records:
            return await SagaNode.get_by_uuid(
                self.driver,
                records[0]['uuid'],
            )

        saga = SagaNode(name=saga_name, group_id=group_id, created_at=created_at)
        await saga.save(self.driver)
        return saga

    async def _saga_get_previous_episode_uuid(
        self, saga_uuid: str, current_episode_uuid: str
    ) -> str | None:
        """Find the most recent episode UUID in a saga, excluding the current one."""
        if self.driver.graph_operations_interface:
            try:
                return await self.driver.graph_operations_interface.saga_get_previous_episode_uuid(
                    self.driver, saga_uuid, current_episode_uuid
                )
            except NotImplementedError:
                pass

        records, _, _ = await self.driver.execute_query(
            """
            MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
            WHERE e.uuid <> $current_episode_uuid
            RETURN e.uuid AS uuid
            ORDER BY e.valid_at DESC, e.created_at DESC
            LIMIT 1
            """,
            saga_uuid=saga_uuid,
            current_episode_uuid=current_episode_uuid,
            routing_='r',
        )
        if records:
            return records[0]['uuid']
        return None

    async def _saga_get_episode_contents(
        self,
        saga_uuid: str,
        since: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[str, datetime | None]] | None:
        """Retrieve (content, valid_at) per episode for summarization, using IoC if available."""
        if self.driver.graph_operations_interface:
            try:
                return await self.driver.graph_operations_interface.saga_get_episode_contents(
                    self.driver, saga_uuid, since=since, limit=limit
                )
            except NotImplementedError:
                pass
        return None

    async def summarize_saga(self, saga_id: str) -> SagaNode:
        """Incrementally summarize a saga using only new episodes since the last summary.

        Two watermarks are maintained on the saga node, with deliberately
        different semantics:

        - ``last_summarized_at`` is wall-clock and is the *filter* watermark:
          the next run picks up any episode whose ``created_at`` (ingestion
          time) is greater than this value. Wall-clock is the right semantics
          here because episode ``created_at`` is monotonic with processing
          time, so a backfilled episode added today with ``valid_at`` in the
          past is still picked up next run.
        - ``last_summarized_episode_valid_at`` is the *temporal* watermark:
          the maximum ``valid_at`` (episode reference time) across the
          episodes covered by the current summary. Consumers asking "how
          recent is the content of this summary in event-time?" should use
          this field, not ``last_summarized_at``.

        If the saga has been summarized before, only episodes added after
        ``last_summarized_at`` are fetched. The existing summary is provided
        to the LLM as context so no information is lost.

        On the first call (no prior summary), all episodes are included.

        Parameters
        ----------
        saga_id : str
            The UUID of the saga to summarize.

        Returns
        -------
        SagaNode
            The updated saga node with the new summary.

        Raises
        ------
        NodeNotFoundError
            If the saga with the given UUID does not exist.
        """
        saga = await SagaNode.get_by_uuid(self.driver, saga_id)

        # Fetch only episodes added since the last summary (or all if never summarized).
        max_episodes = 200
        since = saga.last_summarized_at

        # Try IoC interface first, fall back to raw Cypher
        episodes_data = await self._saga_get_episode_contents(
            saga_id, since=since, limit=max_episodes
        )
        if episodes_data is None:
            if since is not None:
                records, _, _ = await self.driver.execute_query(
                    """
                    MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
                    WHERE e.created_at > $since
                    RETURN e.content AS content, e.valid_at AS valid_at
                    ORDER BY e.valid_at ASC, e.created_at ASC
                    LIMIT $limit
                    """,
                    saga_uuid=saga_id,
                    since=since,
                    limit=max_episodes,
                    routing_='r',
                )
            else:
                records, _, _ = await self.driver.execute_query(
                    """
                    MATCH (s:Saga {uuid: $saga_uuid})-[:HAS_EPISODE]->(e:Episodic)
                    RETURN e.content AS content, e.valid_at AS valid_at
                    ORDER BY e.valid_at DESC, e.created_at DESC
                    LIMIT $limit
                    """,
                    saga_uuid=saga_id,
                    limit=max_episodes,
                    routing_='r',
                )
                # Reverse to chronological order for the prompt
                records = list(reversed(records))

            from graphiti_core.helpers import parse_db_date

            episodes_data = [
                (r['content'], parse_db_date(r.get('valid_at')))
                for r in records
                if r.get('content')
            ]

        if not episodes_data:
            logger.info(f'No new episodes found for saga {saga_id}, skipping summary')
            return saga

        episode_contents = [content for content, _ in episodes_data]
        valid_ats = [valid_at for _, valid_at in episodes_data if valid_at is not None]

        context = {
            'saga_name': saga.name,
            'existing_summary': saga.summary or '',
            'episodes': episode_contents,
        }

        llm_response = await self.llm_client.generate_response(
            prompt_library.summarize_sagas.summarize_saga(context),
            response_model=SagaSummary,
            prompt_name='summarize_sagas.summarize_saga',
        )

        summary = llm_response.get('summary', '')
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS]

        saga.summary = summary
        # Wall-clock watermark for the next-run filter: keeps backfilled
        # episodes (valid_at in the past, created_at = now) reachable on
        # subsequent runs.
        saga.last_summarized_at = utc_now()
        # Episode-time watermark for public/temporal consumers: advance only
        # forward to the latest reference time we just summarized. If no
        # episode in this batch carried a valid_at, leave the previous value
        # unchanged so the field never regresses.
        if valid_ats:
            new_episode_watermark = max(valid_ats)
            if (
                saga.last_summarized_episode_valid_at is None
                or new_episode_watermark > saga.last_summarized_episode_valid_at
            ):
                saga.last_summarized_episode_valid_at = new_episode_watermark
        await saga.save(self.driver)

        logger.info(f'Updated summary for saga {saga_id}')

        return saga

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """
        Build indices and constraints in the Neo4j database.

        This method sets up the necessary indices and constraints in the Neo4j database
        to optimize query performance and ensure data integrity for the knowledge graph.

        Parameters
        ----------
        self
        delete_existing : bool, optional
            Whether to clear existing indices before creating new ones.


        Returns
        -------
        None

        Notes
        -----
        This method should typically be called once during the initial setup of the
        knowledge graph or when updating the database schema. It uses the
        driver's `build_indices_and_constraints` method to perform
        the actual database operations.

        The specific indices and constraints created depend on the implementation
        of the driver's `build_indices_and_constraints` method. Refer to the specific
        driver documentation for details on the exact database schema modifications.

        Caution: Running this method on a large existing database may take some time
        and could impact database performance during execution.
        """
        await self.driver.build_indices_and_constraints(delete_existing)

    async def _extract_and_resolve_nodes(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        previous_episodes: list[EpisodicNode],
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
    ) -> tuple[
        list[EntityNode], dict[str, str], list[tuple[EntityNode, EntityNode]], dict[str, list[int]]
    ]:
        """Extract nodes from episode(s) and resolve against existing graph."""
        episodes = episode if isinstance(episode, list) else [episode]
        primary_episode = episodes[0]

        extracted_nodes, node_episode_index_map = await extract_nodes(
            self.clients, episode, previous_episodes, entity_types, excluded_entity_types
        )

        nodes, uuid_map, duplicates = await resolve_extracted_nodes(
            self.clients,
            extracted_nodes,
            primary_episode,
            previous_episodes,
            entity_types,
        )

        return nodes, uuid_map, duplicates, node_episode_index_map

    async def _extract_and_resolve_edges(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        extracted_nodes: list[EntityNode],
        previous_episodes: list[EpisodicNode],
        edge_type_map: dict[tuple[str, str], list[str]],
        group_id: str,
        edge_types: dict[str, type[BaseModel]] | None,
        nodes: list[EntityNode],
        uuid_map: dict[str, str],
        custom_extraction_instructions: str | None = None,
    ) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
        """Extract edges from episode(s) and resolve against existing graph.

        Returns
        -------
        tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
            A tuple of (resolved_edges, invalidated_edges, new_edges) where:
            - resolved_edges: All edges after resolution
            - invalidated_edges: Edges invalidated by new information
            - new_edges: Only edges that are new to the graph (not duplicates)
        """
        episodes = episode if isinstance(episode, list) else [episode]
        primary_episode = episodes[0]

        extracted_edges = await extract_edges(
            self.clients,
            episode,
            extracted_nodes,
            previous_episodes,
            edge_type_map,
            group_id,
            edge_types,
            custom_extraction_instructions,
        )

        edges = resolve_edge_pointers(extracted_edges, uuid_map)

        resolved_edges, invalidated_edges, new_edges = await resolve_extracted_edges(
            self.clients,
            edges,
            primary_episode,
            nodes,
            edge_types or {},
            edge_type_map,
        )

        return resolved_edges, invalidated_edges, new_edges

    async def _process_episode_data(
        self,
        episode: EpisodicNode | list[EpisodicNode],
        nodes: list[EntityNode],
        entity_edges: list[EntityEdge],
        now: datetime,
        group_id: str,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
        node_episode_index_map: dict[str, list[int]] | None = None,
    ) -> tuple[list[EpisodicEdge], EpisodicNode]:
        """Process and save episode data to the graph.

        Parameters
        ----------
        episode : EpisodicNode | list[EpisodicNode]
            The episode(s) to process.
        nodes : list[EntityNode]
            The entity nodes extracted from the episode(s).
        entity_edges : list[EntityEdge]
            The entity edges extracted from the episode(s).
        now : datetime
            The current timestamp.
        group_id : str
            The group id for the episode.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate
            this episode with. If a string is provided, the saga will be looked up
            by name or created if it doesn't exist.
        saga_previous_episode_uuid : str | None
            Optional. UUID of the previous episode in the saga. If provided, skips
            the database query to find the most recent episode. Useful for efficiently
            adding multiple episodes to the same saga in sequence.
        node_episode_index_map : dict[str, list[int]] | None
            Optional mapping from node UUID to 0-indexed episode positions for
            building episodic edges with correct attribution.
        """
        episodes = episode if isinstance(episode, list) else [episode]
        episode_uuids = [ep.uuid for ep in episodes]

        episodic_edges = build_episodic_edges(nodes, episode_uuids, now, node_episode_index_map)

        for ep in episodes:

            # associate only relationships which actually carry provenance from this episode
            ep.entity_edges = []

            for edge in entity_edges:

                if ep.uuid in edge.episodes:
                    ep.entity_edges.append(edge.uuid)

            if not self.store_raw_episode_content:
                ep.content = ''

        await add_nodes_and_edges_bulk(
            self.driver,
            episodes,
            episodic_edges,
            nodes,
            entity_edges,
            self.embedder,
        )

        primary_episode = episodes[0]

        # Handle saga association if provided
        if saga is not None:
            # Get or create saga node based on input type
            if isinstance(saga, str):
                # Use the originating episode's reference time (valid_at) for a
                # newly created saga so its created_at matches the episode that
                # minted it, not the wall-clock time of this run.
                saga_created_at = primary_episode.valid_at or now
                saga_node = await self._get_or_create_saga(saga, group_id, saga_created_at)
            else:
                saga_node = saga

            records, _, _ = await self.driver.execute_query(
                """
                MATCH (s:Saga {uuid: $saga_uuid})
                OPTIONAL MATCH (s)-[:HAS_EPISODE]->(e:Episodic {uuid: $episode_uuid})
                RETURN count(e) > 0 AS already_attached
                """,
                saga_uuid=saga_node.uuid,
                episode_uuid=primary_episode.uuid,
                routing_='r',
            )

            already_attached = bool(
                records and records[0]['already_attached']
            )

            if not already_attached:

                previous_episode_uuid: str | None = saga_previous_episode_uuid

                if previous_episode_uuid is None:
                    previous_episode_uuid = await self._saga_get_previous_episode_uuid(
                        saga_node.uuid,
                        primary_episode.uuid,
                    )

                if previous_episode_uuid is not None:
                    next_episode_edge = NextEpisodeEdge(
                        source_node_uuid=previous_episode_uuid,
                        target_node_uuid=primary_episode.uuid,
                        group_id=group_id,
                        created_at=now,
                    )
                    await next_episode_edge.save(self.driver)

                has_episode_edge = HasEpisodeEdge(
                    source_node_uuid=saga_node.uuid,
                    target_node_uuid=primary_episode.uuid,
                    group_id=group_id,
                    created_at=now,
                )
                await has_episode_edge.save(self.driver)

                if saga_node.first_episode_uuid is None:
                    saga_node.first_episode_uuid = primary_episode.uuid

                saga_node.last_episode_uuid = primary_episode.uuid
                await saga_node.save(self.driver)

        return episodic_edges, primary_episode

    async def _extract_and_dedupe_nodes_bulk(
        self,
        episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
        edge_type_map: dict[tuple[str, str], list[str]],
        edge_types: dict[str, type[BaseModel]] | None,
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
        custom_extraction_instructions: str | None = None,
    ) -> tuple[
        dict[str, list[EntityNode]],
        dict[str, str],
        list[list[EntityEdge]],
    ]:
        """Extract nodes and edges from all episodes and deduplicate."""
        # Extract all nodes and edges for each episode
        extracted_nodes_bulk, extracted_edges_bulk = await extract_nodes_and_edges_bulk(
            self.clients,
            episode_context,
            edge_type_map=edge_type_map,
            edge_types=edge_types,
            entity_types=entity_types,
            excluded_entity_types=excluded_entity_types,
            custom_extraction_instructions=custom_extraction_instructions,
        )

        # Dedupe extracted nodes in memory
        nodes_by_episode, uuid_map = await dedupe_nodes_bulk(
            self.clients, extracted_nodes_bulk, episode_context, entity_types
        )

        return nodes_by_episode, uuid_map, extracted_edges_bulk

    async def _resolve_nodes_and_edges_bulk(
        self,
        nodes_by_episode: dict[str, list[EntityNode]],
        edges_by_episode: dict[str, list[EntityEdge]],
        episode_context: list[tuple[EpisodicNode, list[EpisodicNode]]],
        entity_types: dict[str, type[BaseModel]] | None,
        edge_types: dict[str, type[BaseModel]] | None,
        edge_type_map: dict[tuple[str, str], list[str]],
        episodes: list[EpisodicNode],
    ) -> tuple[list[EntityNode], list[EntityEdge], list[EntityEdge], dict[str, str]]:
        """Resolve nodes and edges against the existing graph."""
        nodes_by_uuid: dict[str, EntityNode] = {
            node.uuid: node for nodes in nodes_by_episode.values() for node in nodes
        }

        # Get unique nodes per episode
        nodes_by_episode_unique: dict[str, list[EntityNode]] = {}
        nodes_uuid_set: set[str] = set()
        for episode, _ in episode_context:
            nodes_by_episode_unique[episode.uuid] = []
            nodes = [nodes_by_uuid[node.uuid] for node in nodes_by_episode[episode.uuid]]
            for node in nodes:
                if node.uuid not in nodes_uuid_set:
                    nodes_by_episode_unique[episode.uuid].append(node)
                    nodes_uuid_set.add(node.uuid)

        # Resolve nodes
        node_results = await semaphore_gather(
            *[
                resolve_extracted_nodes(
                    self.clients,
                    nodes_by_episode_unique[episode.uuid],
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for episode, previous_episodes in episode_context
            ]
        )

        resolved_nodes: list[EntityNode] = []
        uuid_map: dict[str, str] = {}
        for result in node_results:
            resolved_nodes.extend(result[0])
            uuid_map.update(result[1])

        # Update nodes_by_uuid with resolved nodes
        for resolved_node in resolved_nodes:
            nodes_by_uuid[resolved_node.uuid] = resolved_node

        # Update nodes_by_episode_unique with resolved pointers
        for episode_uuid, nodes in nodes_by_episode_unique.items():
            updated_nodes: list[EntityNode] = []
            for node in nodes:
                updated_node_uuid = uuid_map.get(node.uuid, node.uuid)
                updated_node = nodes_by_uuid[updated_node_uuid]
                updated_nodes.append(updated_node)
            nodes_by_episode_unique[episode_uuid] = updated_nodes

        # Extract attributes for resolved nodes
        hydrated_nodes_results: list[list[EntityNode]] = await semaphore_gather(
            *[
                extract_attributes_from_nodes(
                    self.clients,
                    nodes_by_episode_unique[episode.uuid],
                    episode,
                    previous_episodes,
                    entity_types,
                )
                for episode, previous_episodes in episode_context
            ]
        )

        final_hydrated_nodes = [node for nodes in hydrated_nodes_results for node in nodes]

        # Resolve edges with updated pointers
        edges_by_episode_unique: dict[str, list[EntityEdge]] = {}
        edges_uuid_set: set[str] = set()
        for episode_uuid, edges in edges_by_episode.items():
            edges_with_updated_pointers = resolve_edge_pointers(edges, uuid_map)
            edges_by_episode_unique[episode_uuid] = []

            for edge in edges_with_updated_pointers:
                if edge.uuid not in edges_uuid_set:
                    edges_by_episode_unique[episode_uuid].append(edge)
                    edges_uuid_set.add(edge.uuid)

        edge_results = await semaphore_gather(
            *[
                resolve_extracted_edges(
                    self.clients,
                    edges_by_episode_unique[episode.uuid],
                    episode,
                    final_hydrated_nodes,
                    edge_types or {},
                    edge_type_map,
                )
                for episode in episodes
            ]
        )

        resolved_edges: list[EntityEdge] = []
        invalidated_edges: list[EntityEdge] = []
        for result in edge_results:
            resolved_edges.extend(result[0])
            invalidated_edges.extend(result[1])
            # result[2] is new_edges - not used in bulk flow since attributes
            # are extracted before edge resolution

        return final_hydrated_nodes, resolved_edges, invalidated_edges, uuid_map

    @handle_multiple_group_ids
    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: list[str] | None = None,
        source: EpisodeType | None = None,
        driver: GraphDriver | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        """
        Retrieve the last n episodic nodes from the graph.

        This method fetches a specified number of the most recent episodic nodes
        from the graph, relative to the given reference time.

        Parameters
        ----------
        reference_time : datetime
            The reference time to retrieve episodes before.
        last_n : int, optional
            The number of episodes to retrieve. Defaults to EPISODE_WINDOW_LEN.
        group_ids : list[str | None], optional
            The group ids to return data from.
        source : EpisodeType | None, optional
            Filter episodes by source type.
        driver : GraphDriver | None, optional
            The graph driver to use. If not provided, uses the default driver.
        saga : str | None, optional
            If provided, only retrieve episodes that belong to the saga with this name.

        Returns
        -------
        list[EpisodicNode]
            A list of the most recent EpisodicNode objects.

        Notes
        -----
        The actual retrieval is performed by the `retrieve_episodes` function
        from the `graphiti_core.utils` module, unless a saga is specified.
        """
        if driver is None:
            driver = self.clients.driver

        if driver.graph_operations_interface:
            try:
                return await driver.graph_operations_interface.retrieve_episodes(
                    driver, reference_time, last_n, group_ids, source, saga
                )
            except NotImplementedError:
                pass

        return await retrieve_episodes(driver, reference_time, last_n, group_ids, source, saga)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str | None = None,
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        previous_episode_uuids: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | SagaNode | None = None,
        saga_previous_episode_uuid: str | None = None,
        pre_extracted_nodes: list[EntityNode] | None = None,
        pre_extracted_edges: list[EntityEdge] | None = None,
        timestamps_prepared: bool = False,
    ) -> AddEpisodeResults:
        """
        Process an episode and update the graph.

        This method extracts information from the episode, creates nodes and edges,
        and updates the graph database accordingly.

        Parameters
        ----------
        name : str
            The name of the episode.
        episode_body : str
            The content of the episode.
        source_description : str
            A description of the episode's source.
        reference_time : datetime
            The reference time for the episode.
        source : EpisodeType, optional
            The type of the episode. Defaults to EpisodeType.message.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        uuid : str | None
            Optional uuid of the episode. If an episode with this uuid already exists it is reprocessed;
            otherwise the episode is created with this uuid.
        update_communities : bool
            Optional. Whether to update communities with new node information
        entity_types : dict[str, BaseModel] | None
            Optional. Dictionary mapping entity type names to their Pydantic model definitions.
        excluded_entity_types : list[str] | None
            Optional. List of entity type names to exclude from the graph. Entities classified
            into these types will not be added to the graph. Can include 'Entity' to exclude
            the default entity type.
        previous_episode_uuids : list[str] | None
            Optional.  list of episode uuids to use as the previous episodes. If this is not provided,
            the most recent episodes by created_at date will be used.
        custom_extraction_instructions : str | None
            Optional. Custom extraction instructions string to be included in the extract entities and extract edges prompts.
            This allows for additional instructions or context to guide the extraction process.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate this episode with.
            If a string is provided and a saga with this name already exists in the group, the episode
            will be added to it. Otherwise, a new saga will be created. Sagas are connected to episodes
            via HAS_EPISODE edges, and consecutive episodes are linked via NEXT_EPISODE edges.
        saga_previous_episode_uuid : str | None
            Optional. UUID of the previous episode in the saga. If provided, skips the database
            query to find the most recent episode. Useful for efficiently adding multiple episodes
            to the same saga in sequence. The returned AddEpisodeResults.episode.uuid can be passed
            as this parameter for the next episode.

        Returns
        -------
        None

        Notes
        -----
        This method performs several steps including node extraction, edge extraction,
        deduplication, and database updates. It also handles embedding generation
        and edge invalidation.

        It is recommended to run this method as a background process, such as in a queue.
        It's important that each episode is added sequentially and awaited before adding
        the next one. For web applications, consider using FastAPI's background tasks
        or a dedicated task queue like Celery for this purpose.

        Example using FastAPI background tasks:
            @app.post("/add_episode")
            async def add_episode_endpoint(episode_data: EpisodeData):
                background_tasks.add_task(graphiti.add_episode, **episode_data.dict())
                return {"message": "Episode processing started"}
        """
        start = time()
        now = utc_now()

        validate_entity_types(entity_types)
        validate_excluded_entity_types(excluded_entity_types, entity_types)

        using_pre_extracted = (
            pre_extracted_nodes is not None
            or pre_extracted_edges is not None
        )

        if using_pre_extracted and (
            pre_extracted_nodes is None
            or pre_extracted_edges is None
        ):
            raise ValueError(
                'pre_extracted_nodes and pre_extracted_edges must be supplied together'
            )

        # group_id is stored as a property on the episode, its entities, its edges and its Saga
        if group_id is None:
            group_id = get_default_group_id(self.driver.provider)
        else:
            validate_group_id(group_id)

        with self.tracer.start_span('add_episode') as span:
            try:
                # Retrieve previous episodes for context
                previous_episodes = (
                    await self.retrieve_episodes(
                        reference_time,
                        last_n=RELEVANT_SCHEMA_LIMIT,
                        group_ids=[group_id],
                        source=source,
                    )
                    if previous_episode_uuids is None
                    else await EpisodicNode.get_by_uuids(self.driver, previous_episode_uuids)
                )

                # get or create the episode
                # updated by dxb-software on 2026-08-28 to manually incorporate the fix from PR #1648
                episode = None

                if uuid is not None:

                    try:

                        episode = await EpisodicNode.get_by_uuid(self.driver, uuid)

                    except NodeNotFoundError:

                        logger.info(
                            f"[GRAPHITI/ADD_EPISODE] Episode {uuid} not found in graph; "
                            f"creating it with the supplied uuid"
                        ) # logger

                if episode is None:

                    episode = EpisodicNode(
                        name               = name,
                        group_id           = group_id,
                        labels             = [],
                        source             = source,
                        content            = episode_body,
                        source_description = source_description,
                        created_at         = now,
                        valid_at           = reference_time,
                    ) # episode

                    if uuid is not None:
                        episode.uuid = uuid

                # Create default edge type map
                edge_type_map_default = (
                    {('Entity', 'Entity'): list(edge_types.keys())}
                    if edge_types is not None
                    else {('Entity', 'Entity'): []}
                )

                # use caller-supplied semantic extraction when available;
                # otherwise preserve Graphiti's normal LLM extraction behaviour
                if pre_extracted_nodes is None:

                    extracted_nodes, node_episode_index_map = await extract_nodes(
                        self.clients,
                        episode,
                        previous_episodes,
                        entity_types,
                        excluded_entity_types,
                        custom_extraction_instructions,
                    )

                else:

                    extracted_nodes = [
                        node.model_copy(deep=True)
                        for node in pre_extracted_nodes
                    ]

                    for node in extracted_nodes:
                        node.group_id = group_id

                    node_episode_index_map = {
                        node.uuid: [0]
                        for node in extracted_nodes
                    }

                nodes, uuid_map, _ = await resolve_extracted_nodes(
                    self.clients,
                    extracted_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                )

                if pre_extracted_edges is None:

                    (
                        resolved_edges,
                        invalidated_edges,
                        new_edges,
                    ) = await self._extract_and_resolve_edges(
                        episode,
                        extracted_nodes,
                        previous_episodes,
                        edge_type_map or edge_type_map_default,
                        group_id,
                        edge_types,
                        nodes,
                        uuid_map,
                        custom_extraction_instructions,
                    )

                else:

                    extracted_edges = [
                        edge.model_copy(deep=True)
                        for edge in pre_extracted_edges
                    ]

                    for edge in extracted_edges:

                        edge.group_id = group_id
                        edge.episodes = [episode.uuid]

                        if edge.reference_time is None:
                            edge.reference_time = reference_time

                    resolved_input_edges = resolve_edge_pointers(
                        extracted_edges,
                        uuid_map,
                    )

                    (
                        resolved_edges,
                        invalidated_edges,
                        new_edges,
                    ) = await resolve_extracted_edges(
                        self.clients,
                        resolved_input_edges,
                        episode,
                        nodes,
                        edge_types or {},
                        edge_type_map or edge_type_map_default,
                        preserve_existing_attributes=True,
                        timestamps_prepared=timestamps_prepared,
                    )

                # one canonical relationship can be both resolved and invalidated during this
                # episode; merge those snapshots before deriving provenance or writing to the graph
                entity_edges = _coalesce_entity_edge_updates(
                    resolved_edges + invalidated_edges
                ) # entity_edges

                # merge attributes supplied by the external extractor onto canonical
                # nodes after Graphiti has resolved aliases and duplicates
                if pre_extracted_nodes is not None:

                    resolved_nodes_by_uuid = {
                        node.uuid: node
                        for node in nodes
                    }

                    for extracted_node in extracted_nodes:

                        resolved_uuid = uuid_map.get(
                            extracted_node.uuid,
                            extracted_node.uuid,
                        )

                        resolved_node = resolved_nodes_by_uuid.get(resolved_uuid)

                        if resolved_node is None:
                            continue

                        resolved_node.attributes = dict(
                            resolved_node.attributes or {}
                        )

                        for key, value in extracted_node.attributes.items():

                            if key == 'extraction_score':

                                existing_score = resolved_node.attributes.get(key)

                                try:

                                    if (
                                        existing_score is None
                                        or float(value) > float(existing_score)
                                    ):
                                        resolved_node.attributes[key] = value

                                except (TypeError, ValueError):

                                    resolved_node.attributes[key] = value

                            else:

                                resolved_node.attributes[key] = value

                # GLiNER already supplied entity attributes in the pre-extracted path;
                # retain Graphiti summaries and embeddings but do not ask Qwen to re-extract them
                hydrated_nodes = await extract_attributes_from_nodes(
                    self.clients,
                    nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                    edges=new_edges,
                    extract_attributes=pre_extracted_nodes is None,
                )

                # Process and save episode data (including saga association if provided)
                episodic_edges, episode = await self._process_episode_data(
                    episode,
                    hydrated_nodes,
                    entity_edges,
                    now,
                    group_id,
                    saga,
                    saga_previous_episode_uuid,
                    node_episode_index_map,
                )

                # Update communities if requested
                communities = []
                community_edges = []
                if update_communities:
                    communities, community_edges = await semaphore_gather(
                        *[
                            update_community(self.driver, self.llm_client, self.embedder, node)
                            for node in nodes
                        ],
                        max_coroutines=self.max_coroutines,
                    )

                end = time()

                # Add span attributes
                span.add_attributes(
                    {
                        'episode.uuid': episode.uuid,
                        'episode.source': source.value,
                        'episode.reference_time': reference_time.isoformat(),
                        'group_id': group_id,
                        'node.count': len(hydrated_nodes),
                        'edge.count': len(entity_edges),
                        'edge.invalidated_count': len(invalidated_edges),
                        'previous_episodes.count': len(previous_episodes),
                        'entity_types.count': len(entity_types) if entity_types else 0,
                        'edge_types.count': len(edge_types) if edge_types else 0,
                        'update_communities': update_communities,
                        'communities.count': len(communities) if update_communities else 0,
                        'duration_ms': (end - start) * 1000,
                    }
                )

                logger.info(f'Completed add_episode in {(end - start) * 1000} ms')

                return AddEpisodeResults(
                    episode=episode,
                    episodic_edges=episodic_edges,
                    nodes=hydrated_nodes,
                    edges=entity_edges,
                    communities=communities,
                    community_edges=community_edges,
                )

            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise e

    async def add_episode_bulk(
        self,
        bulk_episodes: list[RawEpisode],
        group_id: str | None = None,
        entity_types: dict[str, type[BaseModel]] | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict[str, type[BaseModel]] | None = None,
        edge_type_map: dict[tuple[str, str], list[str]] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | SagaNode | None = None,
    ) -> AddBulkEpisodeResults:
        """
        Process multiple episodes in bulk and update the graph.

        This method extracts information from multiple episodes, creates nodes and edges,
        and updates the graph database accordingly, all in a single batch operation.

        Parameters
        ----------
        bulk_episodes : list[RawEpisode]
            A list of RawEpisode objects to be processed and added to the graph.
        group_id : str | None
            An id for the graph partition the episode is a part of.
        entity_types : dict[str, type[BaseModel]] | None
            Optional. A dictionary mapping entity type names to Pydantic models.
        excluded_entity_types : list[str] | None
            Optional. A list of entity type names to exclude from extraction.
        edge_types : dict[str, type[BaseModel]] | None
            Optional. A dictionary mapping edge type names to Pydantic models.
        edge_type_map : dict[tuple[str, str], list[str]] | None
            Optional. A mapping of (source_type, target_type) to allowed edge types.
        custom_extraction_instructions : str | None
            Optional. Custom extraction instructions string to be included in the
            extract entities and extract edges prompts. This allows for additional
            instructions or context to guide the extraction process.
        saga : str | SagaNode | None
            Optional. Either a saga name (str) or a SagaNode object to associate all episodes with.
            If a string is provided and a saga with this name already exists in the group, the episodes
            will be added to it. Otherwise, a new saga will be created. Sagas are connected to episodes
            via HAS_EPISODE edges, and consecutive episodes are linked via NEXT_EPISODE edges.

        Returns
        -------
        AddBulkEpisodeResults

        Notes
        -----
        This method performs several steps including:
        - Saving all episodes to the database
        - Retrieving previous episode context for each new episode
        - Extracting nodes and edges from all episodes
        - Generating embeddings for nodes and edges
        - Deduplicating nodes and edges
        - Saving nodes, episodic edges, and entity edges to the knowledge graph

        This bulk operation is designed for efficiency when processing multiple episodes
        at once. However, it's important to ensure that the bulk operation doesn't
        overwhelm system resources. Consider implementing rate limiting or chunking for
        very large batches of episodes.

        Edge invalidation and date extraction (``valid_at`` / ``invalid_at``) are
        performed in the bulk path as well: edges flow through ``extract_edges`` and
        ``resolve_extracted_edges`` just like in ``add_episode``, and any invalidated
        edges are persisted alongside the newly resolved ones.
        """
        with self.tracer.start_span('add_episode_bulk') as bulk_span:
            bulk_span.add_attributes({'episode.count': len(bulk_episodes)})

            try:
                start = time()
                now = utc_now()

                # group_id is stored as a property on the episode, its entities and its edges
                if group_id is None:
                    group_id = get_default_group_id(self.driver.provider)
                else:
                    validate_group_id(group_id)

                # Create default edge type map
                edge_type_map_default = (
                    {('Entity', 'Entity'): list(edge_types.keys())}
                    if edge_types is not None
                    else {('Entity', 'Entity'): []}
                )

                episodes = [
                    await EpisodicNode.get_by_uuid(self.driver, episode.uuid)
                    if episode.uuid is not None
                    else EpisodicNode(
                        name=episode.name,
                        labels=[],
                        source=episode.source,
                        content=episode.content,
                        source_description=episode.source_description,
                        group_id=group_id,
                        created_at=now,
                        valid_at=episode.reference_time,
                    )
                    for episode in bulk_episodes
                ]

                # Save all episodes
                await add_nodes_and_edges_bulk(
                    driver=self.driver,
                    episodic_nodes=episodes,
                    episodic_edges=[],
                    entity_nodes=[],
                    entity_edges=[],
                    embedder=self.embedder,
                )

                # Get previous episode context for each episode
                episode_context = await retrieve_previous_episodes_bulk(self.driver, episodes)

                # Extract and dedupe nodes and edges
                (
                    nodes_by_episode,
                    uuid_map,
                    extracted_edges_bulk,
                ) = await self._extract_and_dedupe_nodes_bulk(
                    episode_context,
                    edge_type_map or edge_type_map_default,
                    edge_types,
                    entity_types,
                    excluded_entity_types,
                    custom_extraction_instructions,
                )

                # Create Episodic Edges
                episodic_edges: list[EpisodicEdge] = []
                for episode_uuid, nodes in nodes_by_episode.items():
                    episodic_edges.extend(build_episodic_edges(nodes, episode_uuid, now))

                # Re-map edge pointers and dedupe edges
                extracted_edges_bulk_updated: list[list[EntityEdge]] = [
                    resolve_edge_pointers(edges, uuid_map) for edges in extracted_edges_bulk
                ]

                edges_by_episode = await dedupe_edges_bulk(
                    self.clients,
                    extracted_edges_bulk_updated,
                    episode_context,
                    [],
                    edge_types or {},
                    edge_type_map or edge_type_map_default,
                )

                # Resolve nodes and edges against the existing graph
                (
                    final_hydrated_nodes,
                    resolved_edges,
                    invalidated_edges,
                    final_uuid_map,
                ) = await self._resolve_nodes_and_edges_bulk(
                    nodes_by_episode,
                    edges_by_episode,
                    episode_context,
                    entity_types,
                    edge_types,
                    edge_type_map or edge_type_map_default,
                    episodes,
                )

                # Resolved pointers for episodic edges
                resolved_episodic_edges = resolve_edge_pointers(episodic_edges, final_uuid_map)

                # collapse repeated canonical relationship snapshots before either the episodes or
                # the relationship rows are persisted
                entity_edges = _coalesce_entity_edge_updates(
                    resolved_edges + invalidated_edges
                ) # entity_edges

                # keep each episode's reverse relationship index consistent with edge.episodes
                for episode in episodes:

                    episode.entity_edges = [
                        edge.uuid
                        for edge in entity_edges
                        if episode.uuid in edge.episodes
                    ]

                # save data to KG
                await add_nodes_and_edges_bulk(
                    self.driver,
                    episodes,
                    resolved_episodic_edges,
                    final_hydrated_nodes,
                    entity_edges,
                    self.embedder,
                )

                # Handle saga association if provided
                if saga is not None:
                    # Get or create saga node based on input type
                    if isinstance(saga, str):
                        # Anchor a newly minted saga to the earliest episode
                        # reference time in the bulk, so created_at reflects the
                        # episode window rather than the time this run started.
                        valid_ats = [ep.valid_at for ep in episodes if ep.valid_at is not None]
                        saga_created_at = min(valid_ats) if valid_ats else now
                        saga_node = await self._get_or_create_saga(saga, group_id, saga_created_at)
                    else:
                        saga_node = saga

                    # Sort episodes by valid_at to create NEXT_EPISODE chain in correct order
                    sorted_episodes = sorted(episodes, key=lambda e: e.valid_at)

                    # Find the most recent episode already in the saga
                    previous_episode_uuid = await self._saga_get_previous_episode_uuid(
                        saga_node.uuid, ''
                    )

                    for episode in sorted_episodes:
                        # Create NEXT_EPISODE edge from the previous episode
                        if previous_episode_uuid is not None:
                            next_episode_edge = NextEpisodeEdge(
                                source_node_uuid=previous_episode_uuid,
                                target_node_uuid=episode.uuid,
                                group_id=group_id,
                                created_at=now,
                            )
                            await next_episode_edge.save(self.driver)

                        # Create HAS_EPISODE edge from saga to episode
                        has_episode_edge = HasEpisodeEdge(
                            source_node_uuid=saga_node.uuid,
                            target_node_uuid=episode.uuid,
                            group_id=group_id,
                            created_at=now,
                        )
                        await has_episode_edge.save(self.driver)

                        # Update previous_episode_uuid for the next iteration
                        previous_episode_uuid = episode.uuid

                    # Track first and last episode on the saga node
                    if sorted_episodes:
                        if saga_node.first_episode_uuid is None:
                            saga_node.first_episode_uuid = sorted_episodes[0].uuid
                        saga_node.last_episode_uuid = sorted_episodes[-1].uuid
                        await saga_node.save(self.driver)

                end = time()

                # Add span attributes
                bulk_span.add_attributes(
                    {
                        'group_id': group_id,
                        'node.count': len(final_hydrated_nodes),
                        'edge.count': len(resolved_edges + invalidated_edges),
                        'duration_ms': (end - start) * 1000,
                    }
                )

                logger.info(f'Completed add_episode_bulk in {(end - start) * 1000} ms')

                return AddBulkEpisodeResults(
                    episodes=episodes,
                    episodic_edges=resolved_episodic_edges,
                    nodes=final_hydrated_nodes,
                    edges=resolved_edges + invalidated_edges,
                    communities=[],
                    community_edges=[],
                )

            except Exception as e:
                bulk_span.set_status('error', str(e))
                bulk_span.record_exception(e)
                raise e

    @handle_multiple_group_ids
    async def build_communities(
        self, group_ids: list[str] | None = None, driver: GraphDriver | None = None
    ) -> tuple[list[CommunityNode], list[CommunityEdge]]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        group_ids : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        if driver is None:
            driver = self.clients.driver

        # Clear existing communities (scoped: only the groups being rebuilt)
        await remove_communities(driver, group_ids=group_ids)

        community_nodes, community_edges = await build_communities(
            driver, self.llm_client, group_ids
        )

        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )

        await semaphore_gather(
            *[node.save(driver) for node in community_nodes],
            max_coroutines=self.max_coroutines,
        )
        await semaphore_gather(
            *[edge.save(driver) for edge in community_edges],
            max_coroutines=self.max_coroutines,
        )

        return community_nodes, community_edges

    @handle_multiple_group_ids
    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
        search_filter: SearchFilters | None = None,
        driver: GraphDriver | None = None,
    ) -> list[EntityEdge]:
        """
        Perform a hybrid search on the knowledge graph.

        This method executes a search query on the graph, combining vector and
        text-based search techniques to retrieve relevant facts, returning the edges as a string.

        This is our basic out-of-the-box search, for more robust results we recommend using our more advanced
        search method graphiti.search_().

        Parameters
        ----------
        query : str
            The search query string.
        center_node_uuid: str, optional
            Facts will be reranked based on proximity to this node
        group_ids : list[str | None] | None, optional
            The graph partitions to return data from.
        num_results : int, optional
            The maximum number of results to return. Defaults to 10.

        Returns
        -------
        list
            A list of EntityEdge objects that are relevant to the search query.

        Notes
        -----
        This method uses a SearchConfig with num_episodes set to 0 and
        num_results set to the provided num_results parameter.

        The search is performed using the current date and time as the reference
        point for temporal relevance.
        """
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        edges = (
            await search(
                self.clients,
                query,
                group_ids,
                search_config,
                search_filter if search_filter is not None else SearchFilters(),
                driver=driver,
                center_node_uuid=center_node_uuid,
            )
        ).edges

        return edges

    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """DEPRECATED"""
        return await self.search_(
            query, config, group_ids, center_node_uuid, bfs_origin_node_uuids, search_filter
        )

    @handle_multiple_group_ids
    async def search_(
        self,
        query: str,
        config: SearchConfig = COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
        driver: GraphDriver | None = None,
    ) -> SearchResults:
        """search_ (replaces _search) is our advanced search method that returns Graph objects (nodes and edges) rather
        than a list of facts. This endpoint allows the end user to utilize more advanced features such as filters and
        different search and reranker methodologies across different layers in the graph.

        For different config recipes refer to search/search_config_recipes.
        """

        return await search(
            self.clients,
            query,
            group_ids,
            config,
            search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid,
            bfs_origin_node_uuids,
            driver=driver,
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)

        edges_list = await semaphore_gather(
            *[EntityEdge.get_by_uuids(self.driver, episode.entity_edges) for episode in episodes],
            max_coroutines=self.max_coroutines,
        )

        edges: list[EntityEdge] = [edge for lst in edges_list for edge in lst]

        nodes = await get_mentioned_nodes(self.driver, episodes)

        return SearchResults(edges=edges, nodes=nodes)

    # added by David Williamson 2026-08-31
    async def get_nodes_and_edges_by_saga(
        self,
        saga_names: list[str],
        group_ids: list[str] | None = None,
    ) -> SearchResults:
        """Return the semantic subgraph and episode provenance for the requested Sagas."""

        if not saga_names:
            return SearchResults()

        episodes = await self.nodes.episode.get_by_saga_names(
            saga_names,
            group_ids,
        )

        if not episodes:
            return SearchResults()

        results = await self.get_nodes_and_edges_by_episode(
            [episode.uuid for episode in episodes]
        )

        # an entity or fact can be referenced by more than one episode, so
        # return each semantic object only once while retaining episode provenance
        nodes_by_uuid = {
            node.uuid: node
            for node in results.nodes
        }

        edges_by_uuid = {
            edge.uuid: edge
            for edge in results.edges
        }

        return SearchResults(
            edges=list(edges_by_uuid.values()),
            nodes=list(nodes_by_uuid.values()),
            episodes=episodes,
        )

    async def add_triplet(
        self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode
    ) -> AddTripletResults:
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        try:
            resolved_source = await EntityNode.get_by_uuid(self.driver, source_node.uuid)
        except NodeNotFoundError:
            resolved_source_nodes, _, _ = await resolve_extracted_nodes(
                self.clients,
                [source_node],
            )
            resolved_source = resolved_source_nodes[0]

        try:
            resolved_target = await EntityNode.get_by_uuid(self.driver, target_node.uuid)
        except NodeNotFoundError:
            resolved_target_nodes, _, _ = await resolve_extracted_nodes(
                self.clients,
                [target_node],
            )
            resolved_target = resolved_target_nodes[0]

        nodes = [resolved_source, resolved_target]

        # Merge user-provided properties from original nodes into resolved nodes (excluding uuid)
        # Update attributes dictionary (merge rather than replace)
        if source_node.attributes:
            resolved_source.attributes.update(source_node.attributes)
        if target_node.attributes:
            resolved_target.attributes.update(target_node.attributes)

        # Update summary if provided by user (non-empty string)
        if source_node.summary:
            resolved_source.summary = source_node.summary
        if target_node.summary:
            resolved_target.summary = target_node.summary

        # Update labels (merge with existing)
        if source_node.labels:
            resolved_source.labels = list(set(resolved_source.labels) | set(source_node.labels))
        if target_node.labels:
            resolved_target.labels = list(set(resolved_target.labels) | set(target_node.labels))

        edge.source_node_uuid = resolved_source.uuid
        edge.target_node_uuid = resolved_target.uuid

        # Check if an edge with this UUID already exists with different source/target nodes.
        # If so, generate a new UUID to create a new edge instead of overwriting.
        try:
            existing_edge = await EntityEdge.get_by_uuid(self.driver, edge.uuid)
            # Edge exists - check if source/target nodes match
            if (
                existing_edge.source_node_uuid != edge.source_node_uuid
                or existing_edge.target_node_uuid != edge.target_node_uuid
            ):
                # Source/target mismatch - generate new UUID to create a new edge
                old_uuid = edge.uuid
                edge.uuid = str(uuid4())
                logger.info(
                    f'Edge UUID {old_uuid} already exists with different source/target nodes. '
                    f'Generated new UUID {edge.uuid} to avoid overwriting.'
                )
        except EdgeNotFoundError:
            # Edge doesn't exist yet, proceed normally
            pass

        valid_edges = await EntityEdge.get_between_nodes(
            self.driver, edge.source_node_uuid, edge.target_node_uuid
        )

        related_edges = (
            await search(
                self.clients,
                edge.fact,
                group_ids=[edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
        ).edges
        existing_edges = (
            await search(
                self.clients,
                edge.fact,
                group_ids=[edge.group_id],
                config=EDGE_HYBRID_SEARCH_RRF,
                search_filter=SearchFilters(),
            )
        ).edges

        resolved_edge, invalidated_edges, _ = await resolve_extracted_edge(
            self.llm_client,
            edge,
            related_edges,
            existing_edges,
            EpisodicNode(
                name='',
                source=EpisodeType.text,
                source_description='',
                content='',
                valid_at=edge.valid_at or utc_now(),
                entity_edges=[],
                group_id=edge.group_id,
            ),
            None,
        )

        edges: list[EntityEdge] = [resolved_edge] + invalidated_edges

        await create_entity_edge_embeddings(self.embedder, edges)
        await create_entity_node_embeddings(self.embedder, nodes)

        await add_nodes_and_edges_bulk(self.driver, [], [], nodes, edges, self.embedder)
        return AddTripletResults(edges=edges, nodes=nodes)

    async def remove_episode(self, episode_uuid: str):
        # Find the episode to be deleted
        episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

        # Find edges mentioned by the episode
        edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)

        edges_to_delete: list[EntityEdge] = []

        for edge in edges:

            if episode.uuid not in edge.episodes:
                continue

            removed_primary_provenance = edge.episodes[0] == episode.uuid

            edge.episodes = [
                supporting_episode_uuid
                for supporting_episode_uuid in edge.episodes
                if supporting_episode_uuid != episode.uuid
            ]

            if not edge.episodes:
                edges_to_delete.append(edge)
                continue

            if removed_primary_provenance:
                replacement_episode = await EpisodicNode.get_by_uuid(
                    self.driver,
                    edge.episodes[0],
                )

                edge.created_at = replacement_episode.created_at
                edge.reference_time = replacement_episode.valid_at

            await edge.load_fact_embedding(self.driver)
            await edge.save(self.driver)

        # Find nodes mentioned by the episode
        nodes = await get_mentioned_nodes(self.driver, [episode])
        # We should delete all nodes that are only mentioned in the deleted episode
        nodes_to_delete: list[EntityNode] = []
        for node in nodes:
            query: LiteralString = 'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count'
            records, _, _ = await self.driver.execute_query(query, uuid=node.uuid, routing_='r')

            for record in records:
                if record['episode_count'] == 1:
                    nodes_to_delete.append(node)

        await Edge.delete_by_uuids(self.driver, [edge.uuid for edge in edges_to_delete])
        await Node.delete_by_uuids(self.driver, [node.uuid for node in nodes_to_delete])

        await episode.delete(self.driver)
