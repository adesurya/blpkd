"""
core/abstractions/vector_store.py

Vector database abstraction layer.
Swap pgvector → Milvus by changing VECTOR_BACKEND env var.

Usage:
    from core.abstractions.vector_store import get_vector_store
    store = get_vector_store()
    await store.upsert(collection="faces", id="face_001", vector=[...], metadata={...})
    results = await store.search(collection="faces", vector=[...], top_k=5)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)

FACE_EMBEDDING_DIM = 512   # InsightFace buffalo_l output dimension


@dataclass
class SearchResult:
    id: str
    score: float          # cosine similarity: 0.0 - 1.0 (higher = more similar)
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Abstract vector store. All backends must implement this."""

    @abstractmethod
    async def create_collection(self, name: str, dim: int) -> None:
        """Create a vector collection/index if not exists."""
        ...

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a vector with metadata."""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> list[SearchResult]:
        """Find top-k most similar vectors above threshold."""
        ...

    @abstractmethod
    async def delete(self, collection: str, id: str) -> None:
        """Delete a vector by ID."""
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Count vectors in a collection."""
        ...


# ─────────────────────────────────────────────────────────────
# MVP: pgvector Backend (PostgreSQL extension)
# ─────────────────────────────────────────────────────────────
class PgVectorStore(VectorStore):
    """
    PostgreSQL pgvector backend.
    Excellent for MVP — up to ~1M vectors with good performance.
    Uses cosine similarity via <=> operator.
    """

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            from pgvector.asyncpg import register_vector
            self._pool = await asyncpg.create_pool(
                self._db_url.replace("+asyncpg", ""),
                init=register_vector,
            )
        return self._pool

    async def create_collection(self, name: str, dim: int = FACE_EMBEDDING_DIM) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {name}_vectors (
                    id          TEXT PRIMARY KEY,
                    vector      vector({dim}),
                    metadata    JSONB,
                    created_at  TIMESTAMP DEFAULT NOW(),
                    updated_at  TIMESTAMP DEFAULT NOW()
                )
            """)
            # IVFFlat index for approximate search (faster than exact at scale)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {name}_vectors_idx
                ON {name}_vectors
                USING ivfflat (vector vector_cosine_ops)
                WITH (lists = 100)
            """)
        log.info("vector_store.collection_created", collection=name, dim=dim)

    async def upsert(
        self,
        collection: str,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        import json
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                INSERT INTO {collection}_vectors (id, vector, metadata, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (id) DO UPDATE
                    SET vector = EXCLUDED.vector,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
            """, id, vector, json.dumps(metadata))

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int = 5,
        threshold: float = 0.45,
    ) -> list[SearchResult]:
        import json
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    id,
                    metadata,
                    1 - (vector <=> $1::vector) AS similarity
                FROM {collection}_vectors
                WHERE 1 - (vector <=> $1::vector) >= $2
                ORDER BY vector <=> $1::vector
                LIMIT $3
            """, vector, threshold, top_k)

        return [
            SearchResult(
                id=row["id"],
                score=float(row["similarity"]),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    async def delete(self, collection: str, id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {collection}_vectors WHERE id = $1", id
            )

    async def count(self, collection: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(f"SELECT COUNT(*) FROM {collection}_vectors")


# ─────────────────────────────────────────────────────────────
# ENTERPRISE: Milvus Backend
# ─────────────────────────────────────────────────────────────
class MilvusVectorStore(VectorStore):
    """
    Milvus vector database backend.
    Best for: 10M+ vectors, distributed deployment, GPU index.
    Swap in Enterprise phase by setting VECTOR_BACKEND=milvus.
    """

    def __init__(self, host: str, port: int) -> None:
        from pymilvus import connections
        connections.connect(host=host, port=port)
        log.info("vector_store.milvus_connected", host=host, port=port)

    async def create_collection(self, name: str, dim: int = FACE_EMBEDDING_DIM) -> None:
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
        if utility.has_collection(name):
            return

        fields = [
            FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("metadata", DataType.JSON),
        ]
        schema = CollectionSchema(fields)
        collection = Collection(name, schema)
        # IVF_FLAT index — good balance of speed/accuracy
        collection.create_index(
            "vector",
            {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}},
        )
        log.info("vector_store.milvus_collection_created", collection=name)

    async def upsert(
        self, collection: str, id: str,
        vector: list[float], metadata: dict[str, Any],
    ) -> None:
        from pymilvus import Collection
        col = Collection(collection)
        col.upsert([[id], [vector], [metadata]])
        col.flush()

    async def search(
        self, collection: str, vector: list[float],
        top_k: int = 5, threshold: float = 0.45,
    ) -> list[SearchResult]:
        from pymilvus import Collection
        col = Collection(collection)
        col.load()
        results = col.search(
            [vector], "vector", {"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k, output_fields=["id", "metadata"],
        )
        return [
            SearchResult(id=hit.entity.get("id"), score=hit.score, metadata=hit.entity.get("metadata", {}))
            for hit in results[0]
            if hit.score >= threshold
        ]

    async def delete(self, collection: str, id: str) -> None:
        from pymilvus import Collection
        Collection(collection).delete(f'id in ["{id}"]')

    async def count(self, collection: str) -> int:
        from pymilvus import Collection
        return Collection(collection).num_entities


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Factory. Returns singleton vector store.

    MVP:        VECTOR_BACKEND=pgvector → PgVectorStore
    Enterprise: VECTOR_BACKEND=milvus   → MilvusVectorStore
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    from core.config.settings import settings

    backend = settings.VECTOR_BACKEND
    if backend == "pgvector":
        _store_instance = PgVectorStore(settings.DATABASE_URL_SYNC)
    elif backend == "milvus":
        _store_instance = MilvusVectorStore(settings.MILVUS_HOST, settings.MILVUS_PORT)
    else:
        raise ValueError(f"Unknown vector backend: {backend}")

    log.info("vector_store.initialized", backend=backend)
    return _store_instance
