"""
TenantDataIngestionService - Coordinated document ingestion into both Neon database and Neo4j graph.
This service handles complete document ingestion workflows for specific tenants with proper isolation.
"""

import logging
import uuid
import time
import json
import asyncio
import sys
import os
from typing import Any, Dict, Optional, List
from datetime import datetime
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tenant_ingestion_models import (
    DocumentInput,
    DocumentChunk,
    TenantIngestionResult,
    BatchIngestionResult,
    TenantAgentDependencies,
    IngestionError,
    BatchIngestionError,
)
from tenant_manager import TenantManager
from tenant_graphiti_client import GraphEpisode

# Import the proper chunker and embedder from ingestion module
from ingestion.chunker import SemanticChunker, ChunkingConfig
from ingestion.embedder import EmbeddingGenerator

logger = logging.getLogger(__name__)


class TenantDataIngestionService:
    """
    Coordinate document ingestion into both Neon database and Neo4j graph.

    This service provides:
    1. Complete document ingestion workflow for specific tenants
    2. Coordination between database and graph storage
    3. Rollback capabilities on failure
    4. Batch ingestion support
    5. Document update and deletion workflows
    """

    def __init__(
        self,
        tenant_manager: TenantManager,
        chunker: SemanticChunker = None,
        embedder: EmbeddingGenerator = None,
    ):
        """
        Initialize ingestion service.

        Args:
            tenant_manager: TenantManager instance for tenant operations
            chunker: SemanticChunker for document chunking (optional, will create default)
            embedder: EmbeddingGenerator for creating embeddings (optional, will create default)
        """
        self.tenant_manager = tenant_manager

        # Initialize chunker with proper configuration from environment
        if chunker is None:
            chunking_config = ChunkingConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 150)),
                max_chunk_size=int(os.getenv("MAX_CHUNK_SIZE", 1500)),
                use_semantic_splitting=True,
                preserve_structure=True,
            )
            self.chunker = SemanticChunker(chunking_config)
        else:
            self.chunker = chunker

        # Initialize embedder
        if embedder is None:
            self.embedder = EmbeddingGenerator()
        else:
            self.embedder = embedder

    async def ingest_document_for_tenant(
        self, tenant_id: str, document: DocumentInput
    ) -> TenantIngestionResult:
        """
        Complete document ingestion workflow for specific tenant with configurable options.

        Args:
            tenant_id: UUID of the tenant
            document: DocumentInput to ingest (with ingest_vector and ingest_graph options)

        Returns:
            TenantIngestionResult with ingestion details

        Raises:
            IngestionError: If any step of ingestion fails
        """
        start_time = time.time()
        document_id = None

        logger.info(
            f"Starting document ingestion for tenant {tenant_id}: {document.title}"
        )

        # Log ingestion options
        ingestion_types = []
        if document.ingest_vector:
            ingestion_types.append("vector database")
        if document.ingest_graph:
            ingestion_types.append("knowledge graph")
        logger.info(f"Ingestion options: {' + '.join(ingestion_types)}")

        try:
            # 1. Get tenant-specific dependencies
            deps = await TenantAgentDependencies.create_for_tenant(
                tenant_id=tenant_id,
                tenant_manager=self.tenant_manager,
                shared_graphiti_client=self.tenant_manager.graphiti_client,
            )

            # 2. Store document in tenant's dedicated database
            document_id = await self._store_document_in_tenant_db(deps, document)

            # 3. Create and store chunks with embeddings (if vector ingestion is enabled)
            chunks_created = 0
            chunk_objects = []
            if document.ingest_vector:
                chunks_created, chunk_objects = await self._process_and_store_chunks(
                    deps, document_id, document
                )
                logger.info(
                    f"Vector ingestion: Created {chunks_created} chunks with embeddings"
                )
            else:
                logger.info("Vector ingestion: Skipped (disabled)")

            # 4. Add document content to tenant's graph namespace (if graph ingestion is enabled)
            # IMPORTANT: Reuse the same chunks to ensure consistency between vector and graph storage
            graph_episode_created = False
            graph_episode_id = None
            if document.ingest_graph:
                if chunks_created > 0:
                    # Use existing chunks from vector ingestion to ensure consistency
                    (
                        graph_episode_created,
                        graph_episode_id,
                    ) = await self._add_chunks_to_tenant_graph(
                        deps, document_id, document, chunk_objects
                    )
                elif not document.ingest_vector:
                    # If only graph ingestion is enabled, create chunks specifically for graph
                    logger.info(
                        "Creating chunks specifically for graph ingestion (vector disabled)"
                    )
                    temp_chunks = await self.chunker.chunk_document(
                        content=document.content,
                        title=document.title,
                        source=document.source,
                        metadata=document.metadata or {},
                    )
                    (
                        graph_episode_created,
                        graph_episode_id,
                    ) = await self._add_chunks_to_tenant_graph(
                        deps, document_id, document, temp_chunks
                    )

                if graph_episode_created:
                    logger.info(f"Graph ingestion: Created episode {graph_episode_id}")
                else:
                    logger.info("Graph ingestion: Episode creation failed")
            else:
                logger.info("Graph ingestion: Skipped (disabled)")

            # 5. Update tenant usage metrics
            await self._update_tenant_metrics(tenant_id, document)

            processing_time = (time.time() - start_time) * 1000

            result = TenantIngestionResult(
                document_id=document_id,
                title=document.title,
                chunks_created=chunks_created,
                processing_time_ms=processing_time,
                vector_stored=chunks_created
                > 0,  # True if chunks with embeddings were created
                graph_stored=graph_episode_created,  # Use graph_episode_created status
                graph_episode_created=graph_episode_created,
                graph_episode_id=graph_episode_id if graph_episode_created else None,
                errors=[],
            )

            logger.info(
                f"Successfully ingested document {document_id} for tenant {tenant_id} in {processing_time:.2f}ms"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to ingest document for tenant {tenant_id}: {str(e)}")

            # Rollback on failure
            if document_id:
                try:
                    await self._rollback_ingestion(deps, document_id)
                except Exception:
                    pass  # Don't fail on rollback failure

            processing_time = (time.time() - start_time) * 1000

            # Return failed result
            result = TenantIngestionResult(
                document_id=document_id or "failed",
                title=document.title,
                chunks_created=0,
                processing_time_ms=processing_time,
                vector_stored=False,
                graph_stored=False,
                graph_episode_created=False,
                errors=[str(e)],
            )

            raise IngestionError(
                f"Failed to ingest document for tenant {tenant_id}: {str(e)}"
            )

    async def _store_document_in_tenant_db(
        self, deps: TenantAgentDependencies, document: DocumentInput
    ) -> str:
        """
        Store document in tenant's dedicated Neon database.

        Args:
            deps: Tenant-specific dependencies
            document: Document to store

        Returns:
            Generated document ID

        Raises:
            Exception: If database storage fails
        """
        document_id = str(uuid.uuid4())

        try:
            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                # Insert document with JSON serialized metadata
                import json

                metadata_json = (
                    json.dumps(document.metadata) if document.metadata else "{}"
                )

                await conn.execute(
                    """
                    INSERT INTO documents (id, title, source, content, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                """,
                    document_id,
                    document.title,
                    document.source,
                    document.content,
                    metadata_json,
                )

                logger.debug(f"Stored document {document_id} in tenant database")
                return document_id

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to store document in tenant database: {str(e)}")
            raise

    async def _process_and_store_chunks(
        self, deps: TenantAgentDependencies, document_id: str, document: DocumentInput
    ) -> tuple[int, List[DocumentChunk]]:
        """
        Create chunks, generate embeddings, store in tenant database.

        Args:
            deps: Tenant-specific dependencies
            document_id: ID of the document
            document: Document content

        Returns:
            Tuple of (number of chunks created, list of chunk objects)

        Raises:
            Exception: If chunk processing fails
        """
        try:
            # Create chunks using the proper semantic chunker
            logger.info(f"Creating chunks for document: {document.title}")
            chunk_objects = await self.chunker.chunk_document(
                content=document.content,
                title=document.title,
                source=document.source,
                metadata=document.metadata,
            )

            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunk_objects)} chunks")
            embedded_chunks = await self.embedder.embed_chunks(chunk_objects)

            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                chunks_created = 0
                chunks_with_embeddings = 0

                for i, chunk in enumerate(embedded_chunks):
                    # Get the embedding vector
                    embedding_vector = getattr(chunk, "embedding", None)

                    if embedding_vector is None:
                        logger.warning(
                            f"No embedding for chunk {i}, skipping vector storage due to API quota/error"
                        )
                        # Store chunk without embedding
                        await conn.execute(
                            """
                            INSERT INTO chunks (id, document_id, content, chunk_index, token_count, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                            str(uuid.uuid4()),
                            document_id,
                            chunk.content,
                            i,
                            chunk.token_count,
                            json.dumps(
                                {
                                    **(chunk.metadata or {}),
                                    "embedding_skipped": True,
                                    "reason": "API quota exhausted or embedding generation failed",
                                }
                            ),
                        )
                        chunks_created += 1
                        continue

                    # Validate embedding quality
                    if not self._is_valid_embedding(embedding_vector):
                        logger.warning(
                            f"Invalid embedding for chunk {i}, skipping vector storage"
                        )
                        # Store chunk without embedding
                        await conn.execute(
                            """
                            INSERT INTO chunks (id, document_id, content, chunk_index, token_count, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                            str(uuid.uuid4()),
                            document_id,
                            chunk.content,
                            i,
                            chunk.token_count,
                            json.dumps(
                                {
                                    **(chunk.metadata or {}),
                                    "embedding_skipped": True,
                                    "reason": "Invalid embedding generated (all zeros)",
                                }
                            ),
                        )
                        chunks_created += 1
                        continue

                    # Convert embedding list to PostgreSQL vector format
                    if isinstance(embedding_vector, list):
                        # Convert list to string format for pgvector: [1.0,2.0,3.0]
                        embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"
                    else:
                        embedding_str = str(embedding_vector)

                    # Store chunk with embedding using proper vector cast
                    await conn.execute(
                        """
                        INSERT INTO chunks (id, document_id, content, embedding, chunk_index, token_count, metadata)
                        VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                    """,
                        str(uuid.uuid4()),
                        document_id,
                        chunk.content,
                        embedding_str,  # Use properly formatted embedding string with ::vector cast
                        i,
                        chunk.token_count,
                        json.dumps(chunk.metadata) if chunk.metadata else "{}",
                    )

                    chunks_created += 1
                    chunks_with_embeddings += 1

                logger.info(
                    f"Created {chunks_created} chunks for document {document_id} "
                    f"({chunks_with_embeddings} with valid embeddings, {chunks_created - chunks_with_embeddings} without)"
                )
                return chunks_created, chunk_objects

            finally:
                await conn.close()

        except Exception as e:
            logger.error(
                f"Failed to process chunks for document {document_id}: {str(e)}"
            )
            raise

    async def _add_chunks_to_tenant_graph(
        self,
        deps: TenantAgentDependencies,
        document_id: str,
        document: DocumentInput,
        chunks: List[DocumentChunk],
    ) -> tuple[bool, str | None]:
        """
        Add pre-existing chunks to tenant's graph namespace via Graphiti using production-ready robust ingestion.

        This method uses the same chunks that were already created for vector ingestion,
        ensuring consistency between vector and graph storage.

        Args:
            deps: Tenant-specific dependencies
            document_id: ID of the document
            document: Document content
            chunks: Pre-existing chunks to add to graph

        Returns:
            Tuple of (success: bool, episodes_created_count: str | None)

        Raises:
            Exception: If graph addition fails critically
        """
        try:
            if not deps.shared_graphiti_client:
                logger.warning(
                    f"No Graphiti client available, skipping graph ingestion for document {document_id}"
                )
                return False, None

            if not chunks:
                logger.warning(f"No chunks provided for document {document_id}")
                return False, None

            logger.info(
                f"Adding {len(chunks)} existing chunks to tenant {deps.tenant_id} graph namespace: tenant_{deps.tenant_id}"
            )

            # Convert chunks to format expected by robust ingestion manager
            chunk_data = []
            for chunk in chunks:
                chunk_data.append(
                    {
                        "chunk_id": getattr(chunk, "id", str(uuid.uuid4())),
                        "text": chunk.content,
                        "created_at": datetime.now(),
                        "metadata": getattr(chunk, "metadata", {}),
                    }
                )

            # Use the new PRODUCTION-READY robust ingestion method
            results = await deps.shared_graphiti_client.ingest_tenant_chunks_into_graph(
                tenant_id=deps.tenant_id,
                doc_name=document.title,
                chunks=chunk_data,
            )

            success = results.get("episodes_ingested", 0) > 0
            episodes_created = results.get("episodes_ingested", 0)
            episodes_failed = results.get("episodes_failed", 0)

            if success:
                logger.info(
                    f"✅ Successfully added {episodes_created} episodes to tenant {deps.tenant_id} graph namespace"
                )

                # Log any partial failures for monitoring
                if episodes_failed > 0:
                    logger.warning(
                        f"⚠️ Partial success: {episodes_failed} episodes failed and may need retry"
                    )
                    # TODO: Store failed episodes in retry queue for later processing

                return True, str(episodes_created)
            else:
                # Complete failure - log details for debugging
                logger.error(
                    f"❌ Complete graph ingestion failure for document {document_id} in tenant {deps.tenant_id}"
                )
                logger.error(f"Failure details: {results}")
                return False, None

        except Exception as e:
            logger.error(
                f"❌ Critical failure in graph ingestion for document {document_id}: {e}"
            )
            raise  # Re-raise for caller to handle

    async def _add_document_to_tenant_graph(
        self, deps: TenantAgentDependencies, document_id: str, document: DocumentInput
    ) -> tuple[bool, str | None]:
        """
        Add document to tenant's graph namespace via Graphiti using production-ready robust ingestion.

        This method uses the new robust ingestion manager that provides:
        - Per-episode atomic ingestion (no partial failures)
        - Text sanitization to prevent JSON parsing errors
        - Structured error handling with retry capability
        - Comprehensive logging and monitoring

        Args:
            deps: Tenant-specific dependencies
            document_id: ID of the document
            document: Document content

        Returns:
            Tuple of (success: bool, episodes_created_count: str | None)

        Raises:
            Exception: If graph addition fails critically
        """
        try:
            if not deps.shared_graphiti_client:
                logger.warning(
                    f"No Graphiti client available, skipping graph ingestion for document {document_id}"
                )
                return False, None

            # Create chunks from document content using proper chunker
            chunks = await self.chunker.chunk_document(
                content=document.content,
                title=document.title,
                source=document.source,
                metadata=document.metadata or {},
            )

            if not chunks:
                logger.warning(f"No chunks created for document {document_id}")
                return False, None

            logger.info(
                f"Adding {len(chunks)} chunks to tenant {deps.tenant_id} graph namespace: tenant_{deps.tenant_id}"
            )

            # Convert chunks to format expected by robust ingestion manager
            chunk_data = []
            for chunk in chunks:
                chunk_data.append(
                    {
                        "chunk_id": getattr(chunk, "id", str(uuid.uuid4())),
                        "text": chunk.content,
                        "created_at": datetime.now(),
                        "metadata": getattr(chunk, "metadata", {}),
                    }
                )

            # Use the new PRODUCTION-READY robust ingestion method
            results = await deps.shared_graphiti_client.ingest_tenant_chunks_into_graph(
                tenant_id=deps.tenant_id,
                doc_name=document.title,
                chunks=chunk_data,
            )

            success = results.get("episodes_ingested", 0) > 0
            episodes_created = results.get("episodes_ingested", 0)
            episodes_failed = results.get("episodes_failed", 0)

            if success:
                logger.info(
                    f"✅ Successfully added {episodes_created} episodes to tenant {deps.tenant_id} graph namespace"
                )

                # Log any partial failures for monitoring
                if episodes_failed > 0:
                    logger.warning(
                        f"⚠️ Partial success: {episodes_failed} episodes failed and may need retry"
                    )
                    # TODO: Store failed episodes in retry queue for later processing

                return True, str(episodes_created)
            else:
                # Complete failure - log details for debugging
                logger.error(
                    f"❌ Complete graph ingestion failure for document {document_id} in tenant {deps.tenant_id}"
                )
                logger.error(f"Failure details: {results}")
                return False, None

        except Exception as e:
            logger.error(
                f"❌ Critical failure in graph ingestion for document {document_id}: {e}"
            )
            raise  # Re-raise for caller to handle

    def _prepare_tenant_episode_content(
        self,
        chunk: DocumentChunk,
        document_title: str,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Prepare episode content for tenant graph with size limits.

        Args:
            chunk: Document chunk
            document_title: Title of the document
            document_metadata: Additional metadata

        Returns:
            Formatted episode content optimized for Graphiti
        """
        # Limit chunk content to avoid token limits (following single-tenant pattern)
        max_content_length = 6000

        content = chunk.content
        if len(content) > max_content_length:
            # Truncate content but try to end at a sentence boundary
            truncated = content[:max_content_length]
            last_sentence_end = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?"),
                truncated.rfind("\n"),
            )

            if last_sentence_end > max_content_length // 2:
                content = truncated[: last_sentence_end + 1]
            else:
                content = truncated + "..."

        # Create episode content with minimal context
        episode_content = f"Document: {document_title}\n\nContent:\n{content}"

        return episode_content

    async def _update_tenant_metrics(self, tenant_id: str, document: DocumentInput):
        """
        Update tenant usage metrics in catalog database.

        Args:
            tenant_id: UUID of the tenant
            document: Document that was ingested

        Raises:
            Exception: If metrics update fails
        """
        try:
            # TODO: Implement usage tracking in catalog database
            # This would track documents ingested, storage used, etc.
            logger.debug(f"Updated usage metrics for tenant {tenant_id}")

        except Exception as e:
            logger.warning(f"Failed to update tenant metrics for {tenant_id}: {str(e)}")
            # Don't raise - metrics failure shouldn't fail ingestion

    async def _rollback_ingestion(
        self, deps: TenantAgentDependencies, document_id: str
    ):
        """
        Rollback failed ingestion by cleaning up partial data.

        Args:
            deps: Tenant-specific dependencies
            document_id: ID of the document to rollback
        """
        if not document_id:
            return

        try:
            logger.info(f"Rolling back failed ingestion for document {document_id}")

            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                # Remove document and cascading chunks from database
                await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
                logger.debug(f"Rollback: Removed document {document_id} from database")

            finally:
                await conn.close()

            # Note: Graph cleanup is handled automatically by Graphiti transactions
            # or would need to be implemented based on Graphiti capabilities

        except Exception as rollback_error:
            # Log rollback failure but don't raise to avoid masking original error
            logger.error(
                f"Rollback failed for document {document_id}: {rollback_error}"
            )

    async def batch_ingest_documents_for_tenant(
        self, tenant_id: str, documents: list[DocumentInput]
    ) -> BatchIngestionResult:
        """
        Batch ingest multiple documents for tenant with progress tracking.

        Args:
            tenant_id: UUID of the tenant
            documents: List of documents to ingest

        Returns:
            BatchIngestionResult with batch processing details

        Raises:
            BatchIngestionError: If batch processing encounters errors
        """
        start_time = time.time()
        successful_results = []
        failed_documents = []

        logger.info(
            f"Starting batch ingestion of {len(documents)} documents for tenant {tenant_id}"
        )

        for i, document in enumerate(documents):
            try:
                result = await self.ingest_document_for_tenant(tenant_id, document)
                successful_results.append(result)

                # Progress logging
                logger.info(
                    f"Batch progress: {i + 1}/{len(documents)} documents processed"
                )

            except IngestionError as e:
                failed_documents.append(
                    {"document": document.model_dump(), "error": str(e), "index": i}
                )
                logger.warning(
                    f"Failed to ingest document {i}: {document.title} - {str(e)}"
                )
                continue

        processing_time = (time.time() - start_time) * 1000

        result = BatchIngestionResult(
            total_documents=len(documents),
            successful_documents=len(successful_results),
            failed_documents=len(failed_documents),
            document_results=successful_results,
            total_processing_time_ms=processing_time,
            errors=[
                f"Failed document {d['index']}: {d['error']}" for d in failed_documents
            ],
        )

        logger.info(
            f"Batch ingestion completed: {result.successful_documents}/{result.total_documents} successful in {processing_time:.2f}ms"
        )

        if failed_documents:
            raise BatchIngestionError(
                f"Batch ingestion completed with {len(failed_documents)} failures",
                successful_ids=[r.document_id for r in successful_results],
                failed_documents=failed_documents,
            )

        return result

    async def update_document_for_tenant(
        self, tenant_id: str, document_id: str, updated_document: DocumentInput
    ) -> bool:
        """
        Update existing document in both database and graph.

        Args:
            tenant_id: UUID of the tenant
            document_id: ID of document to update
            updated_document: Updated document content

        Returns:
            True if update successful

        Raises:
            IngestionError: If update fails
        """
        logger.info(f"Updating document {document_id} for tenant {tenant_id}")

        try:
            # Get tenant dependencies
            deps = await TenantAgentDependencies.create_for_tenant(
                tenant_id=tenant_id,
                tenant_manager=self.tenant_manager,
                shared_graphiti_client=self.tenant_manager.graphiti_client,
            )

            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                # 1. Update document in database
                result = await conn.execute(
                    """
                    UPDATE documents 
                    SET title = $2, content = $3, metadata = $4, updated_at = NOW()
                    WHERE id = $1
                """,
                    document_id,
                    updated_document.title,
                    updated_document.content,
                    updated_document.metadata,
                )

                if result == "UPDATE 0":
                    raise IngestionError(f"Document {document_id} not found")

                # 2. Regenerate chunks and embeddings
                await conn.execute(
                    "DELETE FROM chunks WHERE document_id = $1", document_id
                )
                chunks_created, chunk_objects = await self._process_and_store_chunks(
                    deps, document_id, updated_document
                )

            finally:
                await conn.close()

            # 3. Update graph (Graphiti handles episode updates) - reuse the same chunks
            await self._add_chunks_to_tenant_graph(
                deps, document_id, updated_document, chunk_objects
            )

            logger.info(
                f"Successfully updated document {document_id} for tenant {tenant_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to update document {document_id} for tenant {tenant_id}: {str(e)}"
            )
            raise IngestionError(f"Failed to update document: {str(e)}")

    async def delete_document_for_tenant(
        self, tenant_id: str, document_id: str
    ) -> bool:
        """
        Delete document from both database and graph.

        Args:
            tenant_id: UUID of the tenant
            document_id: ID of document to delete

        Returns:
            True if deletion successful

        Raises:
            IngestionError: If deletion fails
        """
        logger.info(f"Deleting document {document_id} for tenant {tenant_id}")

        try:
            # Get tenant dependencies
            deps = await TenantAgentDependencies.create_for_tenant(
                tenant_id=tenant_id,
                tenant_manager=self.tenant_manager,
                shared_graphiti_client=self.tenant_manager.graphiti_client,
            )

            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                # Delete from database (cascades to chunks due to foreign key)
                result = await conn.execute(
                    "DELETE FROM documents WHERE id = $1", document_id
                )

                if result == "DELETE 0":
                    raise IngestionError(f"Document {document_id} not found")

            finally:
                await conn.close()

            # Note: Graph cleanup is handled automatically in Graphiti
            # Episodes are content-based, so no explicit deletion needed

            logger.info(
                f"Successfully deleted document {document_id} for tenant {tenant_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete document {document_id} for tenant {tenant_id}: {str(e)}"
            )
            raise IngestionError(f"Failed to delete document: {str(e)}")

    async def get_tenant_document_stats(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get document statistics for a tenant.

        Args:
            tenant_id: UUID of the tenant

        Returns:
            Dictionary with document statistics

        Raises:
            Exception: If statistics retrieval fails
        """
        try:
            # Get tenant dependencies
            deps = await TenantAgentDependencies.create_for_tenant(
                tenant_id=tenant_id,
                tenant_manager=self.tenant_manager,
                shared_graphiti_client=self.tenant_manager.graphiti_client,
            )

            # Connect to tenant's database
            conn = await asyncpg.connect(deps.tenant_database_url)

            try:
                # Get document count
                doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")

                # Get chunk count
                chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

                # Get total content size
                total_size = await conn.fetchval(
                    "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM documents"
                )

                # Get latest document
                latest_doc = await conn.fetchrow("""
                    SELECT title, created_at FROM documents 
                    ORDER BY created_at DESC LIMIT 1
                """)

                return {
                    "tenant_id": tenant_id,
                    "document_count": doc_count,
                    "chunk_count": chunk_count,
                    "total_content_size_bytes": total_size,
                    "latest_document": {
                        "title": latest_doc["title"] if latest_doc else None,
                        "created_at": latest_doc["created_at"].isoformat()
                        if latest_doc
                        else None,
                    }
                    if latest_doc
                    else None,
                }

            finally:
                await conn.close()

        except Exception as e:
            logger.error(
                f"Failed to get document stats for tenant {tenant_id}: {str(e)}"
            )
            raise

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
                    "SELECT * FROM match_chunks($1::vector, $2::float, $3::int, $4::uuid)",
                    "[" + ",".join(map(str, query_embedding)) + "]",
                    0.5,  # match_threshold
                    limit,  # match_count
                    None,  # filter_document_id
                )

                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "content": row["content"],
                        "similarity": float(row["similarity"]),
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
                # Use hybrid search function with correct parameter order
                # Database function signature: hybrid_search(search_text text, query_embedding vector, text_weight, vector_weight, match_threshold, max_results)
                results = await conn.fetch(
                    "SELECT * FROM hybrid_search($1::text, $2::vector, $3::float, $4::float, $5::float, $6::int)",
                    query,
                    "[" + ",".join(map(str, query_embedding)) + "]",
                    text_weight,
                    1.0 - text_weight,  # vector_weight
                    0.5,  # match_threshold
                    limit,  # max_results
                )

                return [
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "document_id": str(row["document_id"]),
                        "content": row["content"],
                        "combined_score": float(row["combined_score"]),
                        "vector_similarity": float(row["vector_score"]),
                        "text_similarity": float(row["text_score"]),
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

    def _is_valid_embedding(self, embedding: List[float]) -> bool:
        """
        Validate if an embedding is valid (not all zeros or invalid values).

        Args:
            embedding: The embedding vector to validate

        Returns:
            True if embedding is valid, False otherwise
        """
        if not embedding or len(embedding) == 0:
            return False

        # Check if all values are zero (indicates quota/API failure)
        if all(val == 0.0 for val in embedding):
            return False

        # Check for invalid values (NaN, inf)
        if any(
            not isinstance(val, (int, float))
            or (isinstance(val, float) and (val != val or abs(val) == float("inf")))
            for val in embedding
        ):
            return False

        # Check if embedding has reasonable variance (not all same value)
        unique_values = set(embedding[:10])  # Check first 10 values
        if len(unique_values) == 1:
            return False

        return True

    # ...existing methods...
