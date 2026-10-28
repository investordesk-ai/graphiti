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

import logging

from datetime import datetime
from time import time

from pydantic import BaseModel, ValidationError
from typing_extensions import LiteralString

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    create_entity_edge_embeddings,
)
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeBatchDecisions, EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import (
    BatchEdgeTimestamp,
    BatchEdgeTimestamps,
    EdgeTimestamps,
    ExtractedEdges,
)
from graphiti_core.search.search import search
from graphiti_core.search.search_config import EdgeReranker, SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.attribute_utils import apply_capped_attributes
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.text_utils import concatenate_episodes

logger = logging.getLogger(__name__)

EDGE_TIMESTAMP_BATCH_SIZE = 8
EDGE_RESOLUTION_BATCH_SIZE = 4
EDGE_RESOLUTION_MAX_CHARACTERS = 20000

# rank ingest candidates through the configured cross-encoder without changing retrieval methods
EDGE_INGEST_SEARCH = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
EDGE_INGEST_SEARCH.edge_config.reranker = EdgeReranker.cross_encoder

# clear impossible model-derived temporal bounds before they can become durable graph state
def _clear_impossible_temporal_interval(edge: EntityEdge) -> None:
    """Clear temporal bounds when an edge ends before it begins."""

    valid_at   = ensure_utc(edge.valid_at)
    invalid_at = ensure_utc(edge.invalid_at)

    if valid_at is None or invalid_at is None or invalid_at >= valid_at:
        return

    logger.warning(
        f"[CLEAR_IMPOSSIBLE_TEMPORAL_INTERVAL] Clearing impossible interval for edge "
        f"{edge.uuid}: valid_at={valid_at.isoformat()} invalid_at={invalid_at.isoformat()}"
    )

    edge.valid_at   = None
    edge.invalid_at = None


def build_episodic_edges(
    entity_nodes: list[EntityNode],
    episode_uuid: str | list[str],
    created_at: datetime,
    node_episode_index_map: dict[str, list[int]] | None = None,
) -> list[EpisodicEdge]:
    """Build episodic (MENTIONED_IN) edges between entity nodes and episodes.

    Parameters
    ----------
    entity_nodes : list[EntityNode]
        Nodes to connect to episodes.
    episode_uuid : str | list[str]
        A single episode UUID or a list of episode UUIDs.
    created_at : datetime
        Timestamp for the edges.
    node_episode_index_map : dict[str, list[int]] | None
        Optional mapping from node UUID to 0-indexed episode positions.
        When provided with a list of episode_uuids, each node is connected
        only to its attributed episodes. When None, every node is connected
        to all episodes.
    """
    episode_uuids = [episode_uuid] if isinstance(episode_uuid, str) else episode_uuid

    episodic_edges: list[EpisodicEdge] = []
    for node in entity_nodes:
        if node_episode_index_map and node.uuid in node_episode_index_map:
            indices = node_episode_index_map[node.uuid]
        else:
            indices = list(range(len(episode_uuids)))

        for idx in indices:
            if 0 <= idx < len(episode_uuids):
                episodic_edges.append(
                    EpisodicEdge(
                        source_node_uuid=episode_uuids[idx],
                        target_node_uuid=node.uuid,
                        created_at=created_at,
                        group_id=node.group_id,
                    )
                )

    logger.debug(f'Built {len(episodic_edges)} episodic edges')

    return episodic_edges


def build_community_edges(
    entity_nodes: list[EntityNode],
    community_node: CommunityNode,
    created_at: datetime,
) -> list[CommunityEdge]:
    edges: list[CommunityEdge] = [
        CommunityEdge(
            source_node_uuid=community_node.uuid,
            target_node_uuid=node.uuid,
            created_at=created_at,
            group_id=community_node.group_id,
        )
        for node in entity_nodes
    ]

    return edges


async def extract_edges(
    clients: GraphitiClients,
    episode: EpisodicNode | list[EpisodicNode],
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    edge_type_map: dict[tuple[str, str], list[str]],
    group_id: str = '',
    edge_types: dict[str, type[BaseModel]] | None = None,
    custom_extraction_instructions: str | None = None,
) -> list[EntityEdge]:
    """Extract edges from one or more episodes.

    Parameters
    ----------
    episode : EpisodicNode | list[EpisodicNode]
        A single episode or a list of episodes to extract edges from.
        When a list is provided, their contents are concatenated for extraction
        and edges are linked to all episode UUIDs.
    """
    episodes = episode if isinstance(episode, list) else [episode]
    primary_episode = episodes[0]

    start = time()

    extract_edges_max_tokens = 16384
    llm_client = clients.llm_client

    # Build mapping from edge type name to list of valid signatures
    edge_type_signatures_map: dict[str, list[tuple[str, str]]] = {}
    for signature, edge_type_names in edge_type_map.items():
        for edge_type in edge_type_names:
            if edge_type not in edge_type_signatures_map:
                edge_type_signatures_map[edge_type] = []
            edge_type_signatures_map[edge_type].append(signature)

    edge_types_context = (
        [
            {
                'fact_type_name': type_name,
                'fact_type_signatures': edge_type_signatures_map.get(
                    type_name, [('Entity', 'Entity')]
                ),
                'fact_type_description': type_model.__doc__,
            }
            for type_name, type_model in edge_types.items()
        ]
        if edge_types is not None
        else []
    )

    # Build name-to-node mapping for validation
    name_to_node: dict[str, EntityNode] = {node.name: node for node in nodes}

    # Build episode attribution instructions for multi-episode extraction
    episode_attribution = ''
    if len(episodes) > 1:
        episode_attribution = (
            '\n8. **Episode Attribution**: The CURRENT_MESSAGE contains multiple episodes labeled '
            '[Episode 0], [Episode 1], etc. Each episode header includes a timestamp indicating '
            'when that episode occurred. Use the per-episode timestamp to resolve relative time '
            'mentions within each episode rather than relying solely on REFERENCE_TIME. '
            'For each extracted fact, set `episode_indices` '
            'to the 0-based list of episode numbers that the fact was derived from. '
            'A fact sourced from Episodes 0 and 1 should have `episode_indices: [0, 1]`.'
        )

    # Prepare context for LLM
    # Use the latest episode's timestamp as the primary reference time
    latest_episode = max(episodes, key=lambda ep: ep.valid_at)
    context = {
        'episode_content': concatenate_episodes(episodes),
        'nodes': [{'name': node.name, 'entity_types': node.labels} for node in nodes],
        'previous_episodes': [
            {
                'content': ep.content,
                'timestamp': ep.valid_at.isoformat() if ep.valid_at else None,
            }
            for ep in previous_episodes
        ],
        'reference_time': latest_episode.valid_at,
        'edge_types': edge_types_context,
        'custom_extraction_instructions': (custom_extraction_instructions or '')
        + episode_attribution,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_edges.edge(context),
        response_model=ExtractedEdges,
        max_tokens=extract_edges_max_tokens,
        group_id=group_id or primary_episode.group_id,
        prompt_name='extract_edges.edge',
    )
    all_edges_data = ExtractedEdges(**llm_response).edges

    # Validate entity names
    edges_data: list[ExtractedEdge] = []
    for edge_data in all_edges_data:
        source_name = edge_data.source_entity_name
        target_name = edge_data.target_entity_name

        # Validate LLM-returned names exist in the nodes list
        if source_name not in name_to_node:
            logger.warning(
                'Source entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        if target_name not in name_to_node:
            logger.warning(
                'Target entity not found in nodes for edge relation: %s',
                edge_data.relation_type,
            )
            continue

        # Drop self-edges where source and target resolve to the same node
        source_node = name_to_node[source_name]
        target_node = name_to_node[target_name]
        if source_node.uuid == target_node.uuid:
            logger.info(
                'Dropping self-edge for node %s (source and target resolve to same node)',
                source_node.uuid,
            )
            continue

        edges_data.append(edge_data)

    end = time()
    logger.debug(f'Extracted {len(edges_data)} new edges in {(end - start) * 1000:.0f} ms')

    if len(edges_data) == 0:
        return []

    # Convert the extracted data into EntityEdge objects
    edges = []
    for edge_data in edges_data:
        # Validate Edge Date information
        valid_at = edge_data.valid_at
        invalid_at = edge_data.invalid_at
        valid_at_datetime = None
        invalid_at_datetime = None

        # Filter out empty edges
        if not edge_data.fact.strip():
            continue

        # Names already validated above
        source_node = name_to_node.get(edge_data.source_entity_name)
        target_node = name_to_node.get(edge_data.target_entity_name)

        if source_node is None or target_node is None:
            logger.warning('Could not find source or target node for extracted edge')
            continue

        source_node_uuid = source_node.uuid
        target_node_uuid = target_node.uuid

        if valid_at:
            try:
                valid_at_datetime = ensure_utc(
                    datetime.fromisoformat(valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.warning('Error parsing valid_at date, skipping')

        if invalid_at:
            try:
                invalid_at_datetime = ensure_utc(
                    datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
                )
            except ValueError as e:
                logger.warning(f'WARNING: Error parsing invalid_at date: {e}. Input: {invalid_at}')

        # Map episode_indices (0-indexed) to episode UUIDs.
        # Clamp indices to valid range and fall back to all episodes if empty.
        edge_episode_uuids = []
        for idx in edge_data.episode_indices:
            if 0 <= idx < len(episodes):
                edge_episode_uuids.append(episodes[idx].uuid)
        if not edge_episode_uuids:
            edge_episode_uuids = [ep.uuid for ep in episodes]

        edge = EntityEdge(
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
            name=edge_data.relation_type,
            group_id=group_id or primary_episode.group_id,
            fact=edge_data.fact,
            episodes=edge_episode_uuids,
            created_at=utc_now(),
            valid_at=valid_at_datetime,
            invalid_at=invalid_at_datetime,
            reference_time=(
                episodes[edge_data.episode_indices[0]].valid_at
                if edge_data.episode_indices and 0 <= edge_data.episode_indices[0] < len(episodes)
                else primary_episode.valid_at
            ),
        )
        _clear_impossible_temporal_interval(edge)
        edges.append(edge)
        logger.debug(
            f'Created new edge {edge.uuid} from {edge.source_node_uuid} to {edge.target_node_uuid}'
        )

    logger.debug(f'Extracted edges: {[e.uuid for e in edges]}')

    return edges


async def resolve_extracted_edges(
    clients: GraphitiClients,
    extracted_edges: list[EntityEdge],
    episode: EpisodicNode,
    entities: list[EntityNode],
    edge_types: dict[str, type[BaseModel]],
    edge_type_map: dict[tuple[str, str], list[str]],
    existing_edges_override: list[EntityEdge] | None = None,
    preserve_existing_attributes: bool = False,
    timestamps_prepared: bool = False,
) -> tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]:
    """Resolve extracted edges against existing graph context.

    Returns
    -------
    tuple[list[EntityEdge], list[EntityEdge], list[EntityEdge]]
        A tuple of (resolved_edges, invalidated_edges, new_edges) where:
        - resolved_edges: All edges after resolution (may include existing edges if duplicates found)
        - invalidated_edges: Edges that were invalidated/contradicted by new information
        - new_edges: Only edges that are new to the graph (not duplicates of existing edges)
    """
    # Fast path: deduplicate exact matches within the extracted edges before parallel processing
    seen: dict[tuple[str, str, str, str], EntityEdge] = {}
    deduplicated_edges: list[EntityEdge] = []

    for edge in extracted_edges:
        key = (
            edge.source_node_uuid,
            edge.target_node_uuid,
            edge.name,
            _normalize_string_exact(edge.fact),
        )
        if key not in seen:
            seen[key] = edge
            deduplicated_edges.append(edge)

    extracted_edges = deduplicated_edges

    driver     = clients.driver
    llm_client = clients.llm_client
    embedder   = clients.embedder

    # preparation can run before the group lock; graph-dependent resolution still happens below
    if not timestamps_prepared:
        await _extract_edge_timestamps_batch(
            llm_client,
            extracted_edges,
            episode,
        )

    await create_entity_edge_embeddings(embedder, extracted_edges)

    valid_edges_list: list[list[EntityEdge]] = await semaphore_gather(
        *[
            EntityEdge.get_between_nodes(driver, edge.source_node_uuid, edge.target_node_uuid)
            for edge in extracted_edges
        ]
    )

    # Merge override edges (e.g. from the recent Redis dedup cache) into
    # the per-extracted-edge candidate lists so that recently resolved edges
    # that are not yet visible in the graph-service indexes are still
    # considered during deduplication.
    if existing_edges_override:
        override_by_pair: dict[tuple[str, str], list[EntityEdge]] = {}
        for oe in existing_edges_override:
            key = (oe.source_node_uuid, oe.target_node_uuid)
            override_by_pair.setdefault(key, []).append(oe)

        for i, extracted_edge in enumerate(extracted_edges):
            pair_key = (extracted_edge.source_node_uuid, extracted_edge.target_node_uuid)
            overrides = override_by_pair.get(pair_key, [])
            if overrides:
                existing_uuids = {e.uuid for e in valid_edges_list[i]}
                for oe in overrides:
                    if oe.uuid not in existing_uuids:
                        valid_edges_list[i].append(oe)
                        existing_uuids.add(oe.uuid)

    related_edges_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_INGEST_SEARCH,
                search_filter=SearchFilters(edge_uuids=[edge.uuid for edge in valid_edges]),
            )
            for extracted_edge, valid_edges in zip(extracted_edges, valid_edges_list, strict=True)
        ]
    )

    related_edges_lists: list[list[EntityEdge]] = [result.edges for result in related_edges_results]

    # semantic search may legitimately shortlist only part of the endpoint-matched relationship set;
    # an exact persisted duplicate must never disappear from deterministic resolution because ranking
    # omitted it, otherwise a second active UUID can be created for the same canonical relationship
    for extracted_edge, valid_edges, related_edges in zip(
        extracted_edges,
        valid_edges_list,
        related_edges_lists,
        strict=True,
    ):

        normalised_extracted_fact = _normalize_string_exact(
            extracted_edge.fact
        ) # normalised_extracted_fact

        related_edge_uuids = set()

        for related_edge in related_edges:
            related_edge_uuids.add(related_edge.uuid)

        for valid_edge in valid_edges:

            exact_match = (
                valid_edge.source_node_uuid == extracted_edge.source_node_uuid
                and valid_edge.target_node_uuid == extracted_edge.target_node_uuid
                and valid_edge.name == extracted_edge.name
                and _normalize_string_exact(valid_edge.fact) == normalised_extracted_fact
            ) # exact_match

            if not exact_match:
                continue

            # inject the exact match only when semantic ranking did not already return it
            if valid_edge.uuid not in related_edge_uuids:
                related_edges.insert(0, valid_edge)

            break

    edge_invalidation_candidate_results: list[SearchResults] = await semaphore_gather(
        *[
            search(
                clients,
                extracted_edge.fact,
                group_ids=[extracted_edge.group_id],
                config=EDGE_INGEST_SEARCH,
                search_filter=SearchFilters(),
            )
            for extracted_edge in extracted_edges
        ]
    )

    # Remove duplicates: if an edge appears in both duplicate candidates and invalidation candidates,
    # keep it only in duplicate candidates
    edge_invalidation_candidates: list[list[EntityEdge]] = []
    for related_edges, invalidation_result in zip(
        related_edges_lists, edge_invalidation_candidate_results, strict=True
    ):
        related_uuids = {edge.uuid for edge in related_edges}
        deduplicated = [
            edge for edge in invalidation_result.edges if edge.uuid not in related_uuids
        ]
        edge_invalidation_candidates.append(deduplicated)

    logger.debug(
        f'Related edges: {[e.uuid for edges_lst in related_edges_lists for e in edges_lst]}'
    )

    # Build entity hash table
    uuid_entity_map: dict[str, EntityNode] = {entity.uuid: entity for entity in entities}

    # Collect all node UUIDs referenced by edges that are not in the entities list
    referenced_node_uuids = set()
    for extracted_edge in extracted_edges:
        if extracted_edge.source_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.source_node_uuid)
        if extracted_edge.target_node_uuid not in uuid_entity_map:
            referenced_node_uuids.add(extracted_edge.target_node_uuid)

    # Fetch missing nodes from the database
    if referenced_node_uuids:
        # Pass group_id so graph-service implementations can scope the lookup
        edge_group_id = extracted_edges[0].group_id
        missing_nodes = await EntityNode.get_by_uuids(
            driver, list(referenced_node_uuids), group_id=edge_group_id
        )
        for node in missing_nodes:
            uuid_entity_map[node.uuid] = node

    # Determine which edge types are relevant for each edge based on node signatures.
    # `edge_types_lst` stores the subset of custom edge definitions whose
    # node signature matches each extracted edge.
    edge_types_lst: list[dict[str, type[BaseModel]]] = []
    for extracted_edge in extracted_edges:
        source_node = uuid_entity_map.get(extracted_edge.source_node_uuid)
        target_node = uuid_entity_map.get(extracted_edge.target_node_uuid)
        source_node_labels = (
            source_node.labels + ['Entity'] if source_node is not None else ['Entity']
        )
        target_node_labels = (
            target_node.labels + ['Entity'] if target_node is not None else ['Entity']
        )
        label_tuples = [
            (source_label, target_label)
            for source_label in source_node_labels
            for target_label in target_node_labels
        ]

        extracted_edge_types = {}
        for label_tuple in label_tuples:
            type_names = edge_type_map.get(label_tuple, [])
            for type_name in type_names:
                type_model = edge_types.get(type_name)
                if type_model is None:
                    continue

                extracted_edge_types[type_name] = type_model

        edge_types_lst.append(extracted_edge_types)

    # collect bounded decisions before applying the existing per-edge reconciliation
    batch_decisions = await _batch_edge_decisions(
        llm_client, extracted_edges, related_edges_lists, edge_invalidation_candidates,
    ) # _batch_edge_decisions

    # resolve edges with related edges in the graph and find invalidation candidates
    results: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_edge(
                    llm_client,
                    extracted_edge,
                    related_edges,
                    existing_edges,
                    episode,
                    extracted_edge_types,
                    extract_timestamps=False,
                    preserve_existing_attributes=preserve_existing_attributes,
                    resolution_response=batch_decisions.get(extracted_edge.uuid),
                )
                for extracted_edge, related_edges, existing_edges, extracted_edge_types in zip(
                    extracted_edges,
                    related_edges_lists,
                    edge_invalidation_candidates,
                    edge_types_lst,
                    strict=True,
                )
            ]
        )
    )

    resolved_edges: list[EntityEdge] = []
    invalidated_edges: list[EntityEdge] = []
    new_edges: list[EntityEdge] = []
    for extracted_edge, result in zip(extracted_edges, results, strict=True):
        resolved_edge = result[0]
        invalidated_edge_chunk = result[1]
        # result[2] is duplicate_edges list

        resolved_edges.append(resolved_edge)
        invalidated_edges.extend(invalidated_edge_chunk)

        # Track edges that are new (not duplicates of existing edges)
        # An edge is new if the resolved edge UUID matches the extracted edge UUID
        if resolved_edge.uuid == extracted_edge.uuid:
            new_edges.append(resolved_edge)

    # enforce the temporal invariant again after deduplication and contradiction resolution
    for edge in [*resolved_edges, *invalidated_edges]:
        _clear_impossible_temporal_interval(edge)
        
    logger.debug(f'Resolved edges: {[e.uuid for e in resolved_edges]}')
    logger.debug(f'New edges (non-duplicates): {[e.uuid for e in new_edges]}')

    await semaphore_gather(
        create_entity_edge_embeddings(embedder, resolved_edges),
        create_entity_edge_embeddings(embedder, invalidated_edges),
    )

    return resolved_edges, invalidated_edges, new_edges


def resolve_edge_contradictions(
    resolved_edge: EntityEdge, invalidation_candidates: list[EntityEdge]
) -> list[EntityEdge]:
    if len(invalidation_candidates) == 0:
        return []

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = []
    for edge in invalidation_candidates:
        # (Edge invalid before new edge becomes valid) or (new edge invalid before edge becomes valid)
        edge_invalid_at_utc = ensure_utc(edge.invalid_at)
        resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
        edge_valid_at_utc = ensure_utc(edge.valid_at)
        resolved_edge_invalid_at_utc = ensure_utc(resolved_edge.invalid_at)

        if (
            edge_invalid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_invalid_at_utc <= resolved_edge_valid_at_utc
        ) or (
            edge_valid_at_utc is not None
            and resolved_edge_invalid_at_utc is not None
            and resolved_edge_invalid_at_utc <= edge_valid_at_utc
        ):
            continue
        # New edge invalidates edge
        elif (
            edge_valid_at_utc is not None
            and resolved_edge_valid_at_utc is not None
            and edge_valid_at_utc < resolved_edge_valid_at_utc
        ):
            edge.invalid_at = resolved_edge.valid_at
            edge.expired_at = edge.expired_at if edge.expired_at is not None else utc_now()
            invalidated_edges.append(edge)

    return invalidated_edges


# validate a single timestamp response before allowing it to update graph state
def _validate_edge_timestamp_response(
    timestamps: EdgeTimestamps,
) -> list[str]:

    validation_errors = []

    valid_at   = None
    invalid_at = None

    # validate each supplied bound independently so a malformed single bound also triggers repair
    if timestamps.valid_at:

        try:

            valid_at = ensure_utc(
                datetime.fromisoformat(
                    timestamps.valid_at.replace("Z", "+00:00")
                )
            )

        except ValueError:

            validation_errors.append(
                f"valid_at is not parseable ISO 8601; value={timestamps.valid_at!r}"
            )

    if timestamps.invalid_at:

        try:

            invalid_at = ensure_utc(
                datetime.fromisoformat(
                    timestamps.invalid_at.replace("Z", "+00:00")
                )
            )

        except ValueError:

            validation_errors.append(
                f"invalid_at is not parseable ISO 8601; value={timestamps.invalid_at!r}"
            )

    # only test ordering when both supplied bounds parsed successfully
    if (
        valid_at is not None
        and invalid_at is not None
        and invalid_at < valid_at
    ):

        validation_errors.append(
            f"impossible temporal interval; valid_at={timestamps.valid_at} "
            f"invalid_at={timestamps.invalid_at}"
        )

    return validation_errors


async def _extract_edge_timestamps(
    llm_client: LLMClient,
    edge: EntityEdge,
    episode: EpisodicNode | None,
    strict: bool = False,
) -> None:
    """Extract valid_at / invalid_at timestamps for an edge via a lightweight LLM call.

    Modifies the edge in place. Skips if the edge already has timestamps set
    (e.g., from the extraction prompt in the separate-extraction path) or if
    no reference time is available.
    """
    if edge.valid_at is not None or edge.invalid_at is not None:
        return

    if episode is None or episode.valid_at is None:
        return

    context = {
        'fact': edge.fact,
        'reference_time': episode.valid_at.isoformat(),
    }
    try:
        llm_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_timestamps(context),
            response_model=EdgeTimestamps,
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_timestamps',
        )
        timestamps = EdgeTimestamps(**llm_response)

        validation_errors = _validate_edge_timestamp_response(
            timestamps
        ) # validation_errors

        if validation_errors:

            logger.info(
                f"[EXTRACT_EDGE_TIMESTAMPS] Timestamp response failed semantic validation; "
                f"retrying once with explicit feedback: {validation_errors}"
            )

            repair_context = {
                **context,
                "previous_response" : timestamps.model_dump(),
                "validation_errors" : validation_errors,
            } # repair_context

            repaired_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_timestamps_repair(
                    repair_context
                ),
                response_model=EdgeTimestamps,
                model_size=ModelSize.small,
                prompt_name="extract_edges.extract_timestamps_repair",
            ) # repaired_response

            repaired_timestamps = EdgeTimestamps(**repaired_response)

            repair_validation_errors = _validate_edge_timestamp_response(
                repaired_timestamps
            ) # repair_validation_errors

            if repair_validation_errors:

                logger.warning(
                    f"[EXTRACT_EDGE_TIMESTAMPS] Repaired timestamp response remains invalid; "
                    f"leaving temporal bounds unset: {repair_validation_errors}"
                )

                if strict:
                    raise ValueError("Graph preparation received invalid temporal evidence")
                return

            logger.info(
                "[EXTRACT_EDGE_TIMESTAMPS] Repaired invalid timestamp response"
            )

            timestamps = repaired_timestamps

        if timestamps.valid_at:
            try:
                edge.valid_at = ensure_utc(
                    datetime.fromisoformat(timestamps.valid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.debug(f'Error parsing valid_at: {timestamps.valid_at}')
        if timestamps.invalid_at:
            try:
                edge.invalid_at = ensure_utc(
                    datetime.fromisoformat(timestamps.invalid_at.replace('Z', '+00:00'))
                )
            except ValueError:
                logger.debug(f'Error parsing invalid_at: {timestamps.invalid_at}')

        # enforce the same temporal invariant used by the batch extraction path
        _clear_impossible_temporal_interval(edge)

    except Exception:
        logger.warning('Failed to extract timestamps for edge %s', edge.uuid, exc_info=True)
        raise


def _validate_batch_timestamp_response(
    response_object: BatchEdgeTimestamps,
    batch_edges: list[EntityEdge],
) -> tuple[dict[str, BatchEdgeTimestamp], list[str]]:

    expected_edge_ids = {
        edge.uuid
        for edge in batch_edges
    }

    timestamps_by_edge_id: dict[str, BatchEdgeTimestamp] = {}
    validation_errors = []
    seen_edge_ids = set()

    for temporal_bounds in response_object.timestamps:

        edge_id = temporal_bounds.edge_id

        if edge_id not in expected_edge_ids:

            validation_errors.append(
                f"unexpected edge_id={edge_id}"
            )

            continue

        if edge_id in seen_edge_ids:

            validation_errors.append(
                f"duplicate edge_id={edge_id}"
            )

            continue

        seen_edge_ids.add(edge_id)

        # reject malformed or impossible temporal bounds before allowing them to update graph state
        valid_at                    = None
        invalid_at                  = None
        timestamp_validation_failed = False

        if temporal_bounds.valid_at:

            try:

                valid_at = ensure_utc(
                    datetime.fromisoformat(
                        temporal_bounds.valid_at.replace("Z", "+00:00")
                    )
                )

            except ValueError:

                validation_errors.append(
                    f"edge_id={edge_id} has unparseable valid_at={temporal_bounds.valid_at!r}"
                )

                timestamp_validation_failed = True

        if temporal_bounds.invalid_at:

            try:

                invalid_at = ensure_utc(
                    datetime.fromisoformat(
                        temporal_bounds.invalid_at.replace("Z", "+00:00")
                    )
                )

            except ValueError:

                validation_errors.append(
                    f"edge_id={edge_id} has unparseable invalid_at={temporal_bounds.invalid_at!r}"
                )

                timestamp_validation_failed = True

        if timestamp_validation_failed:
            continue

        if (
            valid_at is not None
            and invalid_at is not None
            and invalid_at < valid_at
        ):

            validation_errors.append(
                f"edge_id={edge_id} has impossible temporal interval; "
                f"valid_at={temporal_bounds.valid_at} "
                f"invalid_at={temporal_bounds.invalid_at}"
            )

            continue

        timestamps_by_edge_id[edge_id] = temporal_bounds

    missing_edge_ids = expected_edge_ids - seen_edge_ids

    for edge_id in sorted(missing_edge_ids):

        validation_errors.append(
            f"missing edge_id={edge_id}"
        )

    return timestamps_by_edge_id, validation_errors


# prepare temporal evidence without reading or mutating the graph
async def prepare_edge_timestamps(
    llm_client: LLMClient,
    edges: list[EntityEdge],
    episode: EpisodicNode,
) -> None:

    if episode.valid_at is None:
        raise ValueError("Graph preparation requires an explicit reference time")

    await _extract_edge_timestamps_batch(llm_client, edges, episode, strict = True)


async def _extract_edge_timestamps_batch(
    llm_client: LLMClient,
    edges: list[EntityEdge],
    episode: EpisodicNode | None,
    strict: bool = False,
) -> None:
    """Extract missing temporal bounds for multiple edges in one LLM call."""

    if episode is None or episode.valid_at is None:
        return

    missing_edges = [
        edge
        for edge in edges
        if edge.valid_at is None and edge.invalid_at is None
    ]

    if not missing_edges:
        return

    try:
        
        for batch_start in range(0, len(missing_edges), EDGE_TIMESTAMP_BATCH_SIZE):
            batch_edges = missing_edges[
                batch_start : batch_start + EDGE_TIMESTAMP_BATCH_SIZE
            ]

            context = {
                "facts": [
                    {
                        "edge_id"        : edge.uuid,
                        "fact"           : edge.fact,
                        "reference_time" : episode.valid_at.isoformat(),
                    }
                    for edge in batch_edges
                ],
            } # context

            # ask the small model for the normal batch timestamp extraction
            llm_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_timestamps_batch(context),
                response_model=BatchEdgeTimestamps,
                model_size=ModelSize.small,
                prompt_name="extract_edges.extract_timestamps_batch",
            ) # llm_response

            response_object = BatchEdgeTimestamps(**llm_response)

            timestamps_by_edge_id, validation_errors = _validate_batch_timestamp_response(
                response_object,
                batch_edges,
            ) # timestamps_by_edge_id, validation_errors

            if validation_errors:

                logger.info(
                    f"[EXTRACT_EDGE_TIMESTAMPS_BATCH] Batch timestamp response failed semantic "
                    f"validation; retrying once with explicit feedback: {validation_errors}"
                )

                repair_context = {
                    **context,
                    "expected_edge_ids" : [
                        edge.uuid
                        for edge in batch_edges
                    ],
                    "previous_response" : response_object.model_dump(),
                    "validation_errors" : validation_errors,
                } # repair_context

                repaired_response = await llm_client.generate_response(
                    prompt_library.extract_edges.extract_timestamps_batch_repair(
                        repair_context
                    ),
                    response_model=BatchEdgeTimestamps,
                    model_size=ModelSize.small,
                    prompt_name="extract_edges.extract_timestamps_batch_repair",
                ) # repaired_response

                repaired_object = BatchEdgeTimestamps(**repaired_response)

                repaired_by_edge_id, repair_validation_errors = (
                    _validate_batch_timestamp_response(
                        repaired_object,
                        batch_edges,
                    )
                ) # repaired_by_edge_id, repair_validation_errors

                if repair_validation_errors:

                    logger.warning(
                        f"[EXTRACT_EDGE_TIMESTAMPS_BATCH] Repaired batch timestamp response "
                        f"remains invalid; falling back only for unresolved edges: "
                        f"{repair_validation_errors}"
                    )

                else:

                    logger.info(
                        "[EXTRACT_EDGE_TIMESTAMPS_BATCH] Repaired invalid batch timestamp response"
                    )

                # prefer every valid result returned by the repair while retaining valid first-attempt
                # results for any edge the repair still failed to return
                timestamps_by_edge_id.update(
                    repaired_by_edge_id
                )

            # apply every valid timestamp result by stable edge ID rather than list position
            unresolved_edges = []

            for edge in batch_edges:

                temporal_bounds = timestamps_by_edge_id.get(
                    edge.uuid
                )

                if temporal_bounds is None:

                    unresolved_edges.append(edge)
                    continue

                if temporal_bounds.valid_at:

                    try:

                        edge.valid_at = ensure_utc(
                            datetime.fromisoformat(
                                temporal_bounds.valid_at.replace("Z", "+00:00")
                            )
                        )

                    except ValueError:

                        logger.debug(
                            f"Error parsing valid_at: {temporal_bounds.valid_at}"
                        )

                if temporal_bounds.invalid_at:

                    try:

                        edge.invalid_at = ensure_utc(
                            datetime.fromisoformat(
                                temporal_bounds.invalid_at.replace("Z", "+00:00")
                            )
                        )

                    except ValueError:

                        logger.debug(
                            f"Error parsing invalid_at: {temporal_bounds.invalid_at}"
                        )

                # never allow probabilistic timestamp extraction to create an impossible graph interval
                _clear_impossible_temporal_interval(edge)

            # if the repair still omitted an edge, fall back only for that unresolved edge
            for unresolved_edge in unresolved_edges:

                logger.warning(
                    f"[EXTRACT_EDGE_TIMESTAMPS_BATCH] Falling back to single-edge timestamp "
                    f"extraction for unresolved edge {unresolved_edge.uuid}"
                )

                await _extract_edge_timestamps(
                    llm_client,
                    unresolved_edge,
                    episode,
                    strict = strict,
                )

    except Exception:

        logger.warning(
            'Failed to extract timestamps for edge batch',
            exc_info=True,
        )

        raise


# validate model-returned edge-resolution indexes against the runtime candidate collections
def _validate_edge_resolution_response(
    response_object: EdgeDuplicate,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
) -> list[str]:

    validation_errors = []

    invalid_duplicate_ids = [
        duplicate_fact_id
        for duplicate_fact_id in response_object.duplicate_facts
        if not 0 <= duplicate_fact_id < len(related_edges)
    ]

    if invalid_duplicate_ids:

        validation_errors.append(
            f"duplicate_facts idx values {invalid_duplicate_ids} are invalid; "
            f"EXISTING FACTS count={len(related_edges)}"
        )

    invalid_contradicted_existing_ids = [
        contradicted_fact_id
        for contradicted_fact_id in response_object.contradicted_existing_facts
        if not 0 <= contradicted_fact_id < len(related_edges)
    ]

    if invalid_contradicted_existing_ids:

        validation_errors.append(
            f"contradicted_existing_facts idx values {invalid_contradicted_existing_ids} are invalid; "
            f"EXISTING FACTS count={len(related_edges)}"
        )

    invalid_contradicted_invalidation_ids = [
        contradicted_fact_id
        for contradicted_fact_id in response_object.contradicted_invalidation_candidates
        if not 0 <= contradicted_fact_id < len(existing_edges)
    ]

    if invalid_contradicted_invalidation_ids:

        validation_errors.append(
            f"contradicted_invalidation_candidates idx values "
            f"{invalid_contradicted_invalidation_ids} are invalid; "
            f"FACT INVALIDATION CANDIDATES count={len(existing_edges)}"
        )

    return validation_errors


# build the same independent candidate index spaces for single and batched resolution
def _edge_resolution_context(extracted_edge, related_edges, existing_edges) -> dict:

    # build independent 0-based index spaces for duplicate candidates and invalidation candidates;
    # the model must never infer which collection owns one numeric index from a shared offset
    related_edges_context = []

    for edge_index, edge in enumerate(related_edges):

        related_edges_context.append({
            "idx"  : edge_index,
            "fact" : edge.fact,
        })

    invalidation_edge_candidates_context = []

    for edge_index, existing_edge in enumerate(existing_edges):

        invalidation_edge_candidates_context.append({
            "idx"  : edge_index,
            "fact" : existing_edge.fact,
        })

    context = {
        "existing_edges"               : related_edges_context,
        "new_edge"                     : extracted_edge.fact,
        "edge_invalidation_candidates" : invalidation_edge_candidates_context,
    } # context

    return context


# batch inference while retaining the existing fast paths and per-edge validation
async def _batch_edge_decisions(llm_client, extracted_edges, related_lists, existing_lists) -> dict:

    decisions = {}
    pending = []

    for edge, related, existing in zip(extracted_edges, related_lists, existing_lists, strict = True):
        if not related and not existing:
            continue

        exact = False
        for candidate in related:
            if (
                candidate.source_node_uuid == edge.source_node_uuid
                and candidate.target_node_uuid == edge.target_node_uuid
                and candidate.name == edge.name
                and _normalize_string_exact(candidate.fact) == _normalize_string_exact(edge.fact)
            ):
                exact = True
                break

        if not exact:
            pending.append({
                "edge_id": edge.uuid,
                "context": _edge_resolution_context(edge, related, existing),
            }) # pending.append

    position = 0

    while position < len(pending):
        items = []
        messages = None

        while position < len(pending) and len(items) < EDGE_RESOLUTION_BATCH_SIZE:
            candidate = items + [pending[position]]
            candidate_messages = prompt_library.dedupe_edges.resolve_edge_batch({"items": candidate})
            characters = 0

            for message in candidate_messages:
                characters += len(message.content)

            if characters > EDGE_RESOLUTION_MAX_CHARACTERS:
                if not items:
                    # preserve oversized evidence by using the existing single-edge path
                    position += 1
                break

            items = candidate
            messages = candidate_messages
            position += 1

        if len(items) < 2:
            continue

        try:
            response = await llm_client.generate_response(
                messages,
                response_model = EdgeBatchDecisions,
                model_size = ModelSize.small,
                prompt_name = "dedupe_edges.resolve_edge_batch",
                max_tokens = 4096,
            ) # generate_response
            parsed = EdgeBatchDecisions.model_validate(response)
        except ValidationError:
            logger.info("[BATCH_EDGE_DECISIONS] Invalid batch shape; resolving these edges individually")
            continue

        requested = set()
        for item in items:
            requested.add(item["edge_id"])

        counts = {}
        for decision in parsed.decisions:
            counts[decision.edge_id] = counts.get(decision.edge_id, 0) + 1

        for decision in parsed.decisions:
            if decision.edge_id in requested and counts[decision.edge_id] == 1:
                decisions[decision.edge_id] = decision.model_dump(exclude = {"edge_id"})

        logger.info(
            f"[BATCH_EDGE_DECISIONS] Received {len(parsed.decisions)} decisions for {len(items)} edges"
        ) # logger.info

    return decisions

async def resolve_extracted_edge(
    llm_client: LLMClient,
    extracted_edge: EntityEdge,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
    episode: EpisodicNode,
    edge_type_candidates: dict[str, type[BaseModel]] | None = None,
    *,
    extract_timestamps: bool = True,
    preserve_existing_attributes: bool = False,
    resolution_response: dict | None = None,
) -> tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]:
    """Resolve an extracted edge against existing graph context.

    Parameters
    ----------
    llm_client : LLMClient
        Client used to invoke the LLM for deduplication and attribute extraction.
    extracted_edge : EntityEdge
        Newly extracted edge whose canonical representation is being resolved.
    related_edges : list[EntityEdge]
        Candidate edges with identical endpoints used for duplicate detection.
    existing_edges : list[EntityEdge]
        Broader set of edges evaluated for contradiction / invalidation.
    episode : EpisodicNode
        Episode providing content context when extracting edge attributes.
    edge_type_candidates : dict[str, type[BaseModel]] | None
        Custom edge types permitted for the current source/target signature.

    Returns
    -------
    tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]
        The resolved edge, any duplicates, and edges to invalidate.
    """
    if len(related_edges) == 0 and len(existing_edges) == 0:
        # Still extract custom attributes and timestamps even when no dedup needed
        edge_model = edge_type_candidates.get(extracted_edge.name) if edge_type_candidates else None
        if edge_model is not None and len(edge_model.model_fields) != 0:
            edge_attributes_context = {
                'fact': extracted_edge.fact,
                'reference_time': episode.valid_at if episode is not None else None,
                'existing_attributes': extracted_edge.attributes,
            }
            edge_attributes_response = await llm_client.generate_response(
                prompt_library.extract_edges.extract_attributes(edge_attributes_context),
                response_model=edge_model,  # type: ignore
                model_size=ModelSize.small,
                prompt_name='extract_edges.extract_attributes',
                attribute_extraction=True,
            )
            merged, _ = apply_capped_attributes(
                edge_attributes_response,
                edge_model,
                extracted_edge.attributes,
                merge_mode='replace',
                prompt_name='extract_edges.extract_attributes',
                entity_uuid=extracted_edge.uuid,
                group_id=extracted_edge.group_id,
            )
            extracted_edge.attributes = merged

        if extract_timestamps:
            await _extract_edge_timestamps(llm_client, extracted_edge, episode)

        return extracted_edge, [], []

    # Fast path: if the relation type, fact text and endpoints already exist verbatim, reuse the matching edge.
    normalized_fact = _normalize_string_exact(extracted_edge.fact)
    for edge in related_edges:
        if (
            edge.source_node_uuid == extracted_edge.source_node_uuid
            and edge.target_node_uuid == extracted_edge.target_node_uuid
            and edge.name == extracted_edge.name
            and _normalize_string_exact(edge.fact) == normalized_fact
        ):
            resolved = edge

            if preserve_existing_attributes:

                resolved.attributes = {
                    **dict(resolved.attributes or {}),
                    **dict(extracted_edge.attributes or {}),
                }

            if episode is not None and episode.uuid not in resolved.episodes:
                resolved.episodes.append(episode.uuid)

            return resolved, [], []

    start = time()

    context = _edge_resolution_context(extracted_edge, related_edges, existing_edges)

    existing_fact_range = "none"

    if related_edges:
        existing_fact_range = f"0-{len(related_edges) - 1}"

    invalidation_candidate_range = "none"

    if existing_edges:
        invalidation_candidate_range = f"0-{len(existing_edges) - 1}"

    if related_edges or existing_edges:

        logger.debug(
            f"Resolving edge: sent {len(related_edges)} EXISTING FACTS "
            f"(idx {existing_fact_range}) and {len(existing_edges)} INVALIDATION CANDIDATES "
            f"(idx {invalidation_candidate_range})"
        )

    # batched decisions enter the existing per-edge validation and repair path
    llm_response = resolution_response

    if llm_response is None:
        llm_response = await llm_client.generate_response(
            prompt_library.dedupe_edges.resolve_edge(context),
            response_model = EdgeDuplicate,
            model_size = ModelSize.small,
            prompt_name = "dedupe_edges.resolve_edge",
        ) # llm_response

    response_object = EdgeDuplicate(**llm_response)

    validation_errors = _validate_edge_resolution_response(
        response_object,
        related_edges,
        existing_edges,
    ) # validation_errors

    if validation_errors:

        logger.info(
            f"[RESOLVE_EXTRACTED_EDGE] Edge-resolution response failed semantic validation; "
            f"retrying once with explicit feedback: {validation_errors}"
        )

        repair_context = {
            **context,
            "previous_response"            : response_object.model_dump(),
            "validation_errors"            : validation_errors,
            "existing_fact_range"          : existing_fact_range,
            "invalidation_candidate_range" : invalidation_candidate_range,
        } # repair_context

        repaired_response = await llm_client.generate_response(
            prompt_library.dedupe_edges.resolve_edge_repair(
                repair_context
            ),
            response_model=EdgeDuplicate,
            model_size=ModelSize.small,
            prompt_name="dedupe_edges.resolve_edge_repair",
        ) # repaired_response

        repaired_object = EdgeDuplicate(**repaired_response)

        repair_validation_errors = _validate_edge_resolution_response(
            repaired_object,
            related_edges,
            existing_edges,
        ) # repair_validation_errors

        if repair_validation_errors:

            logger.warning(
                f"[RESOLVE_EXTRACTED_EDGE] Repaired edge-resolution response remains invalid; "
                f"falling back to deterministic index filtering: {repair_validation_errors}"
            )

        else:

            logger.info(
                "[RESOLVE_EXTRACTED_EDGE] Repaired invalid edge-resolution response"
            )

        # whether valid or not, the repair response supersedes the first response;
        # the existing deterministic filtering below remains the final safety boundary
        response_object = repaired_object

    duplicate_facts                    = response_object.duplicate_facts
    contradicted_existing_facts        = response_object.contradicted_existing_facts
    contradicted_invalidation_facts    = response_object.contradicted_invalidation_candidates

    duplicate_fact_ids = []
    invalid_duplicates = []

    # duplicate indexes may only address the independently indexed EXISTING FACTS collection
    for duplicate_fact_id in duplicate_facts:

        if 0 <= duplicate_fact_id < len(related_edges):
            duplicate_fact_ids.append(duplicate_fact_id)

        else:
            invalid_duplicates.append(duplicate_fact_id)

    if invalid_duplicates:

        logger.warning(
            f"[RESOLVE_EXTRACTED_EDGE] Rejecting invalid duplicate_facts idx values "
            f"{invalid_duplicates} after repair; EXISTING FACTS count={len(related_edges)}"
        )

    resolved_edge = extracted_edge

    for duplicate_fact_id in duplicate_fact_ids:

        resolved_edge = related_edges[duplicate_fact_id]
        break

    if duplicate_fact_ids and episode is not None:

        if episode.uuid not in resolved_edge.episodes:
            resolved_edge.episodes.append(episode.uuid)

    if (
        preserve_existing_attributes
        and resolved_edge.uuid != extracted_edge.uuid
    ):

        resolved_edge.attributes = {
            **dict(resolved_edge.attributes or {}),
            **dict(extracted_edge.attributes or {}),
        }

    contradicted_existing_fact_ids     = []
    invalid_contradicted_existing_ids  = []
    contradicted_invalidation_fact_ids = []
    invalid_contradicted_invalidation  = []

    # contradiction indexes from EXISTING FACTS are validated only against that collection
    for contradicted_fact_id in contradicted_existing_facts:

        if 0 <= contradicted_fact_id < len(related_edges):
            contradicted_existing_fact_ids.append(contradicted_fact_id)

        else:
            invalid_contradicted_existing_ids.append(contradicted_fact_id)

    if invalid_contradicted_existing_ids:

        logger.warning(
            f"[RESOLVE_EXTRACTED_EDGE] Rejecting invalid contradicted_existing_facts idx values "
            f"{invalid_contradicted_existing_ids} after repair; "
            f"EXISTING FACTS count={len(related_edges)}"
        )

    # invalidation indexes have their own independent range starting at zero
    for contradicted_fact_id in contradicted_invalidation_facts:

        if 0 <= contradicted_fact_id < len(existing_edges):
            contradicted_invalidation_fact_ids.append(contradicted_fact_id)

        else:
            invalid_contradicted_invalidation.append(contradicted_fact_id)

    if invalid_contradicted_invalidation:

        logger.warning(
            f"[RESOLVE_EXTRACTED_EDGE] Rejecting invalid contradicted_invalidation_candidates "
            f"idx values {invalid_contradicted_invalidation} after repair; "
            f"FACT INVALIDATION CANDIDATES count={len(existing_edges)}"
        )

    invalidation_candidates = []

    for contradicted_fact_id in contradicted_existing_fact_ids:
        invalidation_candidates.append(related_edges[contradicted_fact_id])

    for contradicted_fact_id in contradicted_invalidation_fact_ids:
        invalidation_candidates.append(existing_edges[contradicted_fact_id])

    # Only extract structured attributes if the edge's relation_type matches an allowed custom type
    # AND the edge model exists for this node pair signature
    edge_model = edge_type_candidates.get(resolved_edge.name) if edge_type_candidates else None
    if edge_model is not None and len(edge_model.model_fields) != 0:
        edge_attributes_context = {
            'fact': resolved_edge.fact,
            'reference_time': episode.valid_at if episode is not None else None,
            'existing_attributes': resolved_edge.attributes,
        }

        edge_attributes_response = await llm_client.generate_response(
            prompt_library.extract_edges.extract_attributes(edge_attributes_context),
            response_model=edge_model,  # type: ignore
            model_size=ModelSize.small,
            prompt_name='extract_edges.extract_attributes',
            attribute_extraction=True,
        )

        merged, _ = apply_capped_attributes(
            edge_attributes_response,
            edge_model,
            resolved_edge.attributes,
            merge_mode='replace',
            prompt_name='extract_edges.extract_attributes',
            entity_uuid=resolved_edge.uuid,
            group_id=resolved_edge.group_id,
        )
        resolved_edge.attributes = merged
    else:

        # the standard Graphiti path retains its existing stale-attribute behaviour;
        # externally extracted edges keep the validated attributes supplied by their extractor
        if not preserve_existing_attributes:
            resolved_edge.attributes = {}

    # direct callers retain the single-edge fallback; resolve_extracted_edges()
    # already performed the batched temporal pass
    if extract_timestamps and resolved_edge.uuid == extracted_edge.uuid:
        await _extract_edge_timestamps(llm_client, resolved_edge, episode)

    end = time()
    logger.debug(
        f'Resolved Edge: {extracted_edge.uuid} -> {resolved_edge.uuid}, in {(end - start) * 1000} ms'
    )

    now = utc_now()

    if resolved_edge.invalid_at and not resolved_edge.expired_at:
        resolved_edge.expired_at = now

    # Determine if the new_edge needs to be expired
    if resolved_edge.expired_at is None:
        invalidation_candidates.sort(key=lambda c: (c.valid_at is None, ensure_utc(c.valid_at)))
        for candidate in invalidation_candidates:
            candidate_valid_at_utc = ensure_utc(candidate.valid_at)
            resolved_edge_valid_at_utc = ensure_utc(resolved_edge.valid_at)
            if (
                candidate_valid_at_utc is not None
                and resolved_edge_valid_at_utc is not None
                and candidate_valid_at_utc > resolved_edge_valid_at_utc
            ):
                # Expire new edge since we have information about more recent events
                resolved_edge.invalid_at = candidate.valid_at
                resolved_edge.expired_at = now
                break

    # Determine which contradictory edges need to be expired
    invalidated_edges: list[EntityEdge] = resolve_edge_contradictions(
        resolved_edge, invalidation_candidates
    )
    duplicate_edges: list[EntityEdge] = [related_edges[idx] for idx in duplicate_fact_ids]

    return resolved_edge, invalidated_edges, duplicate_edges


async def filter_existing_duplicate_of_edges(
    driver: GraphDriver, duplicates_node_tuples: list[tuple[EntityNode, EntityNode]]
) -> list[tuple[EntityNode, EntityNode]]:
    if not duplicates_node_tuples:
        return []

    duplicate_nodes_map = {
        (source.uuid, target.uuid): (source, target) for source, target in duplicates_node_tuples
    }

    if driver.provider == GraphProvider.NEPTUNE:
        query: LiteralString = """
            UNWIND $duplicate_node_uuids AS duplicate_tuple
            MATCH (n:Entity {uuid: duplicate_tuple.source})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple.target})
            RETURN DISTINCT
                n.uuid AS source_uuid,
                m.uuid AS target_uuid
        """

        duplicate_nodes = [
            {'source': source.uuid, 'target': target.uuid}
            for source, target in duplicates_node_tuples
        ]

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_nodes,
            routing_='r',
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            query = """
                UNWIND $duplicate_node_uuids AS duplicate
                MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = [{'src': src, 'dst': dst} for src, dst in duplicate_nodes_map]
        else:
            query: LiteralString = """
                UNWIND $duplicate_node_uuids AS duplicate_tuple
                MATCH (n:Entity {uuid: duplicate_tuple[0]})-[r:RELATES_TO {name: 'IS_DUPLICATE_OF'}]->(m:Entity {uuid: duplicate_tuple[1]})
                RETURN DISTINCT
                    n.uuid AS source_uuid,
                    m.uuid AS target_uuid
            """
            duplicate_node_uuids = list(duplicate_nodes_map.keys())

        records, _, _ = await driver.execute_query(
            query,
            duplicate_node_uuids=duplicate_node_uuids,
            routing_='r',
        )

    # Remove duplicates that already have the IS_DUPLICATE_OF edge
    for record in records:
        duplicate_tuple = (record.get('source_uuid'), record.get('target_uuid'))
        if duplicate_nodes_map.get(duplicate_tuple):
            duplicate_nodes_map.pop(duplicate_tuple)

    return list(duplicate_nodes_map.values())
