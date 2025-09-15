"""
Multi-tenant document ingestion pipeline.
Reuses single-tenant chunker and embedder for each tenant's dedicated database.
"""

import asyncio
import logging
import uuid
import json
from typing import List, Dict, Any, Optional
import asyncpg
from dataclasses import dataclass

# Import single-tenant modules
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.chunker import ChunkingConfig, create_chunker, DocumentChunk
from ingestion.embedder import create_embedder
from tenant_graphiti_client import TenantGraphitiClient

logger = logging.getLogger(__name__)


@dataclass
class MultiTenantDocument:
    """Document for multi-tenant ingestion"""

    title: str
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class MultiTenantIngestionPipeline:
    """Multi-tenant document ingestion pipeline using single-tenant components"""

    def __init__(self):
        """Initialize the pipeline with chunker and embedder"""
        self.chunker_config = ChunkingConfig(
            chunk_size=800,  # Gemini optimized
            chunk_overlap=150,
            use_semantic_splitting=True,
        )
        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()

        logger.info("Multi-tenant ingestion pipeline initialized")

    async def ingest_document_for_tenant(
        self,
        tenant_database_url: str,
        document: MultiTenantDocument,
        graphiti_client: Optional[TenantGraphitiClient] = None,
        tenant_namespace: Optional[str] = None,
    ) -> str:
        """
        Ingest a document into a specific tenant's database and graph.

        Args:
            tenant_database_url: Database URL for the tenant's dedicated DB
            document: Document to ingest
            graphiti_client: Optional Graphiti client for knowledge graph
            tenant_namespace: Tenant namespace for graph isolation

        Returns:
            Document ID
        """
        logger.info(f"Starting document ingestion for tenant: {document.title}")

        try:
            # Step 1: Create chunks using single-tenant chunker
            chunks = await self.chunker.chunk_document(
                content=document.content,
                title=document.title,
                source=document.source,
                metadata=document.metadata or {},
            )

            logger.info(f"Created {len(chunks)} chunks for document: {document.title}")

            # Step 2: Generate embeddings using single-tenant embedder
            embedded_chunks = await self.embedder.embed_chunks(chunks)

            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

            # Step 3: Store in tenant's dedicated database
            document_id = await self._store_document_in_tenant_db(
                tenant_database_url, document, embedded_chunks
            )

            logger.info(f"Stored document in tenant DB with ID: {document_id}")

            # Step 4: Add to knowledge graph if client provided
            if graphiti_client and tenant_namespace:
                await self._add_to_knowledge_graph(
                    graphiti_client, document, document_id, tenant_namespace
                )
                logger.info(
                    f"Added document to knowledge graph for namespace: {tenant_namespace}"
                )

            return document_id

        except Exception as e:
            logger.error(f"Failed to ingest document for tenant: {e}")
            raise

    async def _store_document_in_tenant_db(
        self,
        tenant_database_url: str,
        document: MultiTenantDocument,
        chunks: List[DocumentChunk],
    ) -> str:
        """Store document and chunks in tenant's dedicated database"""

        conn = await asyncpg.connect(tenant_database_url)

        try:
            async with conn.transaction():
                # Insert document
                document_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO documents (id, title, source, content, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                    """,
                    document_id,
                    document.title,
                    document.source,
                    document.content,
                    json.dumps(document.metadata or {}),
                )

                # Insert chunks with embeddings
                for chunk in chunks:
                    if hasattr(chunk, "embedding") and chunk.embedding:
                        # Convert embedding to PostgreSQL vector format
                        embedding_str = "[" + ",".join(map(str, chunk.embedding)) + "]"

                        await conn.execute(
                            """
                            INSERT INTO chunks (
                                id, document_id, content, embedding, chunk_index, 
                                token_count, metadata, created_at
                            )
                            VALUES ($1, $2, $3, $4::vector, $5, $6, $7, NOW())
                            """,
                            str(uuid.uuid4()),
                            document_id,
                            chunk.content,
                            embedding_str,
                            chunk.index,
                            chunk.token_count,
                            json.dumps(chunk.metadata),
                        )
                    else:
                        # Insert chunk without embedding (fallback)
                        await conn.execute(
                            """
                            INSERT INTO chunks (
                                id, document_id, content, chunk_index, 
                                token_count, metadata, created_at
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, NOW())
                            """,
                            str(uuid.uuid4()),
                            document_id,
                            chunk.content,
                            chunk.index,
                            chunk.token_count,
                            json.dumps(chunk.metadata),
                        )

                # Update processing status
                await conn.execute(
                    """
                    INSERT INTO document_processing (
                        document_id, status, chunks_created, processing_started_at, 
                        processing_completed_at, metadata
                    )
                    VALUES ($1, 'completed', $2, NOW(), NOW(), $3)
                    ON CONFLICT (document_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        chunks_created = EXCLUDED.chunks_created,
                        processing_completed_at = EXCLUDED.processing_completed_at,
                        metadata = EXCLUDED.metadata
                    """,
                    document_id,
                    len(chunks),
                    json.dumps(
                        {
                            "embeddings_generated": len(
                                [c for c in chunks if hasattr(c, "embedding")]
                            )
                        }
                    ),
                )

                return document_id

        finally:
            await conn.close()

    async def _add_to_knowledge_graph(
        self,
        graphiti_client: TenantGraphitiClient,
        document: MultiTenantDocument,
        document_id: str,
        tenant_namespace: str,
    ):
        """Add document to knowledge graph with tenant namespace isolation"""

        from tenant_graphiti_client import GraphEpisode

        # Extract tenant_id from namespace (tenant_namespace is "tenant_{tenant_id}")
        tenant_id = (
            tenant_namespace.replace("tenant_", "")
            if tenant_namespace.startswith("tenant_")
            else tenant_namespace
        )

        episode = GraphEpisode(
            tenant_id=tenant_id,  # Use actual tenant ID, not namespace
            name=f"Document: {document.title}",
            content=document.content,
            source_description=f"Document from {document.source}",
            metadata={
                "document_id": document_id,
                "source": document.source,
                "title": document.title,
                "tenant_namespace": tenant_namespace,  # Keep namespace in metadata
                **(document.metadata or {}),
            },
        )

        result = await graphiti_client.add_episode_for_tenant(episode)

        if result:
            logger.info(
                f"✅ Successfully added episode for document {document.title} to tenant {tenant_id}"
            )
        else:
            logger.error(
                f"❌ Failed to add episode for document {document.title} to tenant {tenant_id}"
            )
            # Even if episode addition fails, continue with the workflow
            logger.warning(
                "⚠️  Document ingestion will continue despite episode addition failure"
            )

        return result

    async def vector_search_for_tenant(
        self, tenant_database_url: str, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search in a specific tenant's database.

        Args:
            tenant_database_url: Database URL for the tenant's dedicated DB
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.embed_query(query)

            # Connect to tenant database
            conn = await asyncpg.connect(tenant_database_url)

            try:
                # Try vector search using the match_chunks function
                results = await conn.fetch(
                    "SELECT * FROM match_chunks($1::vector, 0.7, $2)",
                    "[" + ",".join(map(str, query_embedding)) + "]",
                    limit,
                )

                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "content": row["content"],
                        "similarity": float(row["similarity"]),
                        "chunk_index": row["chunk_index"],
                        "token_count": row["token_count"],
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else {},
                        "document_title": row["document_title"],
                        "document_source": row["document_source"],
                    }
                    for row in results
                ]

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Vector search failed for tenant: {e}")
            # Fallback to text search
            return await self._text_search_fallback(tenant_database_url, query, limit)

    async def _text_search_fallback(
        self, tenant_database_url: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback text search if vector search fails"""

        conn = await asyncpg.connect(tenant_database_url)

        try:
            results = await conn.fetch(
                """
                SELECT c.id as chunk_id, c.document_id, c.content, 
                       c.chunk_index, c.token_count, c.metadata,
                       d.title as document_title, d.source as document_source,
                       0.5 as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.content ILIKE '%' || $1 || '%'
                ORDER BY c.chunk_index
                LIMIT $2
                """,
                query,
                limit,
            )

            return [
                {
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "similarity": float(row["similarity"]),
                    "chunk_index": row["chunk_index"],
                    "token_count": row["token_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "document_title": row["document_title"],
                    "document_source": row["document_source"],
                }
                for row in results
            ]

        finally:
            await conn.close()

    async def hybrid_search_for_tenant(
        self,
        tenant_database_url: str,
        query: str,
        limit: int = 10,
        text_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + text) in a specific tenant's database.

        Args:
            tenant_database_url: Database URL for the tenant's dedicated DB
            query: Search query
            limit: Maximum number of results
            text_weight: Weight for text similarity
            vector_weight: Weight for vector similarity

        Returns:
            List of search results with combined scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedder.embed_query(query)

            # Connect to tenant database
            conn = await asyncpg.connect(tenant_database_url)

            try:
                # Use hybrid search function
                results = await conn.fetch(
                    "SELECT * FROM hybrid_search($1, $2::vector, $3, $4, 0.7, $5)",
                    query,
                    "[" + ",".join(map(str, query_embedding)) + "]",
                    text_weight,
                    vector_weight,
                    limit,
                )

                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "content": row["content"],
                        "combined_score": float(row["combined_score"]),
                        "text_score": float(row["text_score"]),
                        "vector_score": float(row["vector_score"]),
                        "metadata": json.loads(row["metadata"])
                        if row["metadata"]
                        else {},
                        "document_title": row["document_title"],
                        "document_source": row["document_source"],
                    }
                    for row in results
                ]

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Hybrid search failed for tenant: {e}")
            # Fallback to vector search only
            return await self.vector_search_for_tenant(
                tenant_database_url, query, limit
            )


# Example usage
async def main():
    """Example usage of multi-tenant ingestion"""

    pipeline = MultiTenantIngestionPipeline()

    # Sample document
    document = MultiTenantDocument(
        title="AI Technology Overview",
        source="example.md",
        content="""
        Artificial Intelligence (AI) has revolutionized many industries.
        Machine learning algorithms are now used in various applications.
        Neural networks and deep learning are key technologies in AI.
        
        Google's AI initiatives include advanced language models and computer vision.
        The company has invested heavily in transformer architectures.
        """,
        metadata={"category": "technology", "author": "Example Author"},
    )

    # Tenant database URL (example)
    tenant_db_url = "postgresql://user:pass@host:5432/tenant_db"

    try:
        # Ingest document
        doc_id = await pipeline.ingest_document_for_tenant(tenant_db_url, document)
        print(f"Ingested document with ID: {doc_id}")

        # Test vector search
        results = await pipeline.vector_search_for_tenant(
            tenant_db_url, "AI technology", limit=5
        )
        print(f"Vector search results: {len(results)}")

        # Test hybrid search
        hybrid_results = await pipeline.hybrid_search_for_tenant(
            tenant_db_url, "Google AI", limit=5
        )
        print(f"Hybrid search results: {len(hybrid_results)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
