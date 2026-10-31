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
from typing import Any

from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations
from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EpisodicEdge
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.models.edges.edge_db_queries import (
    EPISODIC_EDGE_RETURN,
)

logger = logging.getLogger(__name__)


def _episodic_edge_from_record(record: Any) -> EpisodicEdge:
    return EpisodicEdge(
        uuid=record['uuid'],
        group_id=record['group_id'],
        source_node_uuid=record['source_node_uuid'],
        target_node_uuid=record['target_node_uuid'],
        created_at=parse_db_date(record['created_at']),  # type: ignore[arg-type]
    )


class Neo4jEpisodicEdgeOperations(EpisodicEdgeOperations):
    async def save(
        self,
        executor: QueryExecutor,
        edge: EpisodicEdge,
        tx: Transaction | None = None,
    ) -> None:
        await self.save_bulk(executor, [edge], tx = tx)

        logger.debug(f'Saved Edge to Graph: {edge.uuid}')

    # replay the same episode/entity association while retaining its stored identity and creation time
    async def save_bulk(
        self,
        executor: QueryExecutor,
        edges: list[EpisodicEdge],
        tx: Transaction | None = None,
        batch_size: int = 100,
    ) -> None:
        if not edges:
            return

        requested = {}
        for edge in edges:
            identity = (edge.group_id, edge.source_node_uuid, edge.target_node_uuid)
            if edge.uuid in requested and requested[edge.uuid] != identity:
                raise ValueError(f'Conflicting episode/entity bindings for mention UUID {edge.uuid}')
            requested[edge.uuid] = identity

        # hold the episode write lock before checking its links; MERGE alone permits concurrent duplicates
        query = """
            UNWIND $episodic_edges AS edge
            MATCH (episode:Episodic {uuid: edge.source_node_uuid})
            SET episode.uuid = episode.uuid
            WITH edge, episode
            MATCH (node:Entity {uuid: edge.target_node_uuid})
            MERGE (episode)-[mention:MENTIONS {group_id: edge.group_id}]->(node)
            ON CREATE SET mention.uuid = edge.uuid, mention.created_at = edge.created_at
            RETURN edge.uuid AS requested_uuid, mention.uuid AS uuid, mention.created_at AS created_at
        """
        edge_dicts = [edge.model_dump() for edge in edges]
        if tx is not None:
            result = await tx.run(query, episodic_edges = edge_dicts)
            records = await result.data()
        else:
            records, _, _ = await executor.execute_query(query, episodic_edges = edge_dicts)

        # validate every returned identity before updating any caller-owned object
        saved = {}
        for record in records:
            requested_uuid = record['requested_uuid']
            if requested_uuid not in requested:
                raise ValueError(f'Unexpected mention result for UUID {requested_uuid}')
            uuid = record['uuid']
            created_at = parse_db_date(record['created_at'])
            if not isinstance(uuid, str) or not uuid.strip() or created_at is None:
                raise ValueError(f'Incomplete stored mention identity for UUID {requested_uuid}')
            identity = (uuid, created_at)
            if requested_uuid in saved and saved[requested_uuid] != identity:
                raise ValueError(f'Ambiguous stored mentions for UUID {requested_uuid}')
            saved[requested_uuid] = identity
        if set(saved) != set(requested):
            raise ValueError('Some mention endpoints are missing from the graph')

        for edge in edges:
            edge.uuid, edge.created_at = saved[edge.uuid]

    async def delete(
        self,
        executor: QueryExecutor,
        edge: EpisodicEdge,
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuid=edge.uuid)
        else:
            await executor.execute_query(query, uuid=edge.uuid)

        logger.debug(f'Deleted Edge: {edge.uuid}')

    async def delete_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
        tx: Transaction | None = None,
    ) -> None:
        query = """
            MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)
            WHERE e.uuid IN $uuids
            DELETE e
        """
        if tx is not None:
            await tx.run(query, uuids=uuids)
        else:
            await executor.execute_query(query, uuids=uuids)

    async def get_by_uuid(
        self,
        executor: QueryExecutor,
        uuid: str,
    ) -> EpisodicEdge:
        query = (
            """
            MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
            RETURN
            """
            + EPISODIC_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuid=uuid, routing_='r')
        edges = [_episodic_edge_from_record(r) for r in records]
        if len(edges) == 0:
            raise EdgeNotFoundError(uuid)
        return edges[0]

    async def get_by_uuids(
        self,
        executor: QueryExecutor,
        uuids: list[str],
    ) -> list[EpisodicEdge]:
        query = (
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.uuid IN $uuids
            RETURN
            """
            + EPISODIC_EDGE_RETURN
        )
        records, _, _ = await executor.execute_query(query, uuids=uuids, routing_='r')
        return [_episodic_edge_from_record(r) for r in records]

    async def get_by_group_ids(
        self,
        executor: QueryExecutor,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicEdge]:
        cursor_clause = 'AND e.uuid < $uuid' if uuid_cursor else ''
        limit_clause = 'LIMIT $limit' if limit is not None else ''
        query = (
            """
            MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
            WHERE e.group_id IN $group_ids
            """
            + cursor_clause
            + """
            RETURN
            """
            + EPISODIC_EDGE_RETURN
            + """
            ORDER BY e.uuid DESC
            """
            + limit_clause
        )
        records, _, _ = await executor.execute_query(
            query,
            group_ids=group_ids,
            uuid=uuid_cursor,
            limit=limit,
            routing_='r',
        )
        return [_episodic_edge_from_record(r) for r in records]
