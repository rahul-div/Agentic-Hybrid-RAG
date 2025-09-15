"""
Tenant Database Schema Initializer
Handles initialization of identical database schemas in each tenant's dedicated Neon project.

This module implements Task 1.2.3 from the corrected task breakdown:
- Initialize schema in new tenant databases
- Ensure all tenants have identical schema structure
- Handle database connections per tenant
- Provide schema migration utilities
"""

import logging
import asyncpg
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TenantSchemaInitializer:
    """
    Initialize identical schema in each tenant's dedicated database.

    Each tenant gets their own Neon project/database, so no tenant_id columns needed.
    The entire database belongs to one tenant.
    """

    # Complete tenant database schema - NO tenant_id columns needed
    TENANT_SCHEMA_SQL = """
    -- Enable required extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    
    -- Documents table (NO tenant_id - entire DB is for one tenant)
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        title TEXT NOT NULL,
        source TEXT NOT NULL,
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Chunks with vector embeddings (768 dimensions for Gemini)
    CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        content TEXT NOT NULL,
        embedding vector(768),
        chunk_index INTEGER NOT NULL,
        token_count INTEGER,
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(document_id, chunk_index)
    );
    
    -- Sessions for conversation tracking
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        user_id VARCHAR(50),
        title VARCHAR(255),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW(),
        last_activity TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Messages for conversation history
    CREATE TABLE IF NOT EXISTS messages (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Knowledge graph entities (local cache for this tenant's Neo4j data)
    CREATE TABLE IF NOT EXISTS entities (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        name TEXT NOT NULL,
        entity_type VARCHAR(100) NOT NULL,
        properties JSONB DEFAULT '{}',
        source_document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        updated_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(name, entity_type)
    );
    
    -- Knowledge graph relationships (local cache)
    CREATE TABLE IF NOT EXISTS relationships (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        relationship_type VARCHAR(100) NOT NULL,
        properties JSONB DEFAULT '{}',
        source_document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE(source_entity_id, target_entity_id, relationship_type)
    );
    
    -- Document processing status and metadata
    CREATE TABLE IF NOT EXISTS document_processing (
        document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
        status VARCHAR(50) NOT NULL DEFAULT 'pending',
        chunks_created INTEGER DEFAULT 0,
        entities_extracted INTEGER DEFAULT 0,
        relationships_created INTEGER DEFAULT 0,
        processing_started_at TIMESTAMPTZ,
        processing_completed_at TIMESTAMPTZ,
        error_message TEXT,
        metadata JSONB DEFAULT '{}'
    );
    
    -- Performance indexes for vector similarity search
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON chunks(token_count);
    
    -- Performance indexes for text search and filtering
    CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
    CREATE INDEX IF NOT EXISTS idx_documents_title_fts ON documents USING gin(to_tsvector('english', title));
    CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents USING gin(to_tsvector('english', content));
    CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);
    
    -- Session and message indexes
    CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
    CREATE INDEX IF NOT EXISTS idx_sessions_activity ON sessions(last_activity DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
    
    -- Entity and relationship indexes
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_entities_document ON entities(source_document_id);
    CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
    CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
    CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type);
    
    -- Document processing indexes
    CREATE INDEX IF NOT EXISTS idx_doc_processing_status ON document_processing(status);
    CREATE INDEX IF NOT EXISTS idx_doc_processing_started ON document_processing(processing_started_at);
    
    -- Update triggers for timestamp management
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
    CREATE TRIGGER update_entities_updated_at BEFORE UPDATE ON entities 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
    CREATE TRIGGER update_sessions_activity BEFORE UPDATE ON sessions 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
    -- Vector search functions
    
    -- Vector search function for semantic similarity
    CREATE OR REPLACE FUNCTION match_chunks(
        query_embedding VECTOR(768),
        match_threshold FLOAT DEFAULT 0.5,
        match_count INTEGER DEFAULT 10,
        filter_document_id UUID DEFAULT NULL
    )
    RETURNS TABLE (
        chunk_id UUID,
        document_id UUID,
        content TEXT,
        similarity FLOAT,
        chunk_index INTEGER,
        token_count INTEGER,
        metadata JSONB,
        document_title TEXT,
        document_source TEXT
    )
    LANGUAGE SQL STABLE
    AS $$
        SELECT 
            c.id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> query_embedding) AS similarity,
            c.chunk_index,
            c.token_count,
            c.metadata,
            d.title AS document_title,
            d.source AS document_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 
            c.embedding IS NOT NULL
            AND 1 - (c.embedding <=> query_embedding) > match_threshold
            AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
        ORDER BY (1 - (c.embedding <=> query_embedding)) DESC
        LIMIT match_count;
    $$;
    
    -- Simple text-based search function (fallback without vector embeddings)
    CREATE OR REPLACE FUNCTION search_chunks_text(
        query_text TEXT,
        match_count INTEGER DEFAULT 10
    )
    RETURNS TABLE (
        chunk_id UUID,
        document_id UUID,
        content TEXT,
        similarity FLOAT,
        chunk_index INTEGER,
        token_count INTEGER
    )
    LANGUAGE SQL STABLE
    AS $$
        SELECT 
            chunks.id,
            chunks.document_id,
            chunks.content,
            CASE 
                WHEN chunks.content ILIKE '%' || query_text || '%' THEN 0.9
                ELSE 0.1
            END AS similarity,
            chunks.chunk_index,
            chunks.token_count
        FROM chunks
        WHERE chunks.content ILIKE '%' || query_text || '%'
        ORDER BY 
            CASE WHEN chunks.content ILIKE '%' || query_text || '%' THEN 1 ELSE 0 END DESC,
            chunks.chunk_index
        LIMIT match_count;
    $$;
    
    -- Full-text search function
    CREATE OR REPLACE FUNCTION search_documents(
        search_query TEXT,
        max_results INTEGER DEFAULT 10
    )
    RETURNS TABLE (
        document_id UUID,
        title TEXT,
        source TEXT,
        rank REAL
    )
    LANGUAGE SQL STABLE
    AS $$
        SELECT 
            d.id,
            d.title,
            d.source,
            ts_rank(to_tsvector('english', d.title || ' ' || d.content), plainto_tsquery('english', search_query)) AS rank
        FROM documents d
        WHERE to_tsvector('english', d.title || ' ' || d.content) @@ plainto_tsquery('english', search_query)
        ORDER BY rank DESC
        LIMIT max_results;
    $$;
    
    -- Hybrid search function (combines vector and text search)
    CREATE OR REPLACE FUNCTION hybrid_search(
        search_text TEXT,
        query_embedding VECTOR(768),
        text_weight FLOAT DEFAULT 0.3,
        vector_weight FLOAT DEFAULT 0.7,
        match_threshold FLOAT DEFAULT 0.5,
        max_results INTEGER DEFAULT 10
    )
    RETURNS TABLE (
        chunk_id UUID,
        document_id UUID,
        content TEXT,
        combined_score FLOAT,
        text_score FLOAT,
        vector_score FLOAT,
        metadata JSONB,
        document_title TEXT,
        document_source TEXT
    )
    LANGUAGE SQL STABLE
    AS $$
        WITH text_results AS (
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content,
                c.metadata,
                d.title as document_title,
                d.source as document_source,
                ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', search_text)) AS text_rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', search_text)
        ),
        vector_results AS (
            SELECT 
                c.id as chunk_id,
                c.document_id,
                c.content,
                c.metadata,
                d.title as document_title,
                d.source as document_source,
                1 - (c.embedding <=> query_embedding) AS vector_rank
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL
                AND 1 - (c.embedding <=> query_embedding) > match_threshold
        )
        SELECT 
            COALESCE(t.chunk_id, v.chunk_id) as chunk_id,
            COALESCE(t.document_id, v.document_id) as document_id,
            COALESCE(t.content, v.content) as content,
            (COALESCE(t.text_rank, 0) * text_weight + COALESCE(v.vector_rank, 0) * vector_weight) as combined_score,
            COALESCE(t.text_rank, 0) as text_score,
            COALESCE(v.vector_rank, 0) as vector_score,
            COALESCE(t.metadata, v.metadata) as metadata,
            COALESCE(t.document_title, v.document_title) as document_title,
            COALESCE(t.document_source, v.document_source) as document_source
        FROM text_results t
        FULL OUTER JOIN vector_results v ON t.chunk_id = v.chunk_id
        ORDER BY combined_score DESC
        LIMIT max_results;
    $$;
    
    -- Comments for documentation
    COMMENT ON TABLE documents IS 'Core documents table for tenant-specific storage';
    COMMENT ON TABLE chunks IS 'Document chunks with vector embeddings for semantic search';
    COMMENT ON TABLE sessions IS 'User conversation sessions';
    COMMENT ON TABLE messages IS 'Chat messages within sessions';
    COMMENT ON TABLE entities IS 'Local cache of knowledge graph entities';
    COMMENT ON TABLE relationships IS 'Local cache of knowledge graph relationships';
    COMMENT ON TABLE document_processing IS 'Processing status and metadata for documents';
    
    -- Function comments
    COMMENT ON FUNCTION match_chunks IS 'Vector similarity search function';
    COMMENT ON FUNCTION search_documents IS 'Full-text search across documents';
    COMMENT ON FUNCTION hybrid_search IS 'Combined vector and text search with weighted scoring';
    """

    # Schema version for migration tracking
    SCHEMA_VERSION = "2.0.0"

    def __init__(self):
        """Initialize the schema initializer"""
        pass

    async def initialize_tenant_database(self, database_url: str) -> bool:
        """
        Initialize complete schema in new tenant database.

        Args:
            database_url: PostgreSQL connection URL for the tenant's database

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing tenant database schema")

        try:
            # Connect to tenant database
            conn = await asyncpg.connect(database_url)

            try:
                # Execute schema SQL in a transaction
                async with conn.transaction():
                    await conn.execute(self.TENANT_SCHEMA_SQL)

                    # Store schema version
                    await conn.execute("""
                        CREATE TABLE IF NOT EXISTS schema_metadata (
                            key VARCHAR(50) PRIMARY KEY,
                            value TEXT NOT NULL,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)

                    await conn.execute(
                        """
                        INSERT INTO schema_metadata (key, value)
                        VALUES ('schema_version', $1)
                        ON CONFLICT (key) DO UPDATE SET 
                            value = EXCLUDED.value,
                            updated_at = NOW()
                    """,
                        self.SCHEMA_VERSION,
                    )

                logger.info("Successfully initialized tenant database schema")
                return True

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize tenant database schema: {e}")
            return False

    async def validate_tenant_schema(self, database_url: str) -> Dict[str, Any]:
        """
        Validate that tenant database has correct schema.

        Args:
            database_url: PostgreSQL connection URL for the tenant's database

        Returns:
            Dict with validation results
        """
        logger.info("Validating tenant database schema")

        try:
            conn = await asyncpg.connect(database_url)

            try:
                # Check required tables exist
                required_tables = [
                    "documents",
                    "chunks",
                    "sessions",
                    "messages",
                    "entities",
                    "relationships",
                    "document_processing",
                    "schema_metadata",
                ]

                existing_tables = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """)

                existing_table_names = {row["table_name"] for row in existing_tables}
                missing_tables = set(required_tables) - existing_table_names

                # Check required extensions
                extensions = await conn.fetch("""
                    SELECT extname FROM pg_extension
                """)
                existing_extensions = {row["extname"] for row in extensions}
                required_extensions = {"vector", "uuid-ossp", "pg_trgm"}
                missing_extensions = required_extensions - existing_extensions

                # Check schema version
                try:
                    version_row = await conn.fetchrow("""
                        SELECT value FROM schema_metadata WHERE key = 'schema_version'
                    """)
                    current_version = version_row["value"] if version_row else None
                except Exception:
                    current_version = None

                # Check required functions
                functions = await conn.fetch("""
                    SELECT routine_name 
                    FROM information_schema.routines 
                    WHERE routine_schema = 'public' AND routine_type = 'FUNCTION'
                """)
                existing_functions = {row["routine_name"] for row in functions}
                required_functions = {
                    "match_chunks",
                    "search_documents",
                    "hybrid_search",
                    "update_updated_at_column",
                }
                missing_functions = required_functions - existing_functions

                validation_result = {
                    "valid": len(missing_tables) == 0
                    and len(missing_extensions) == 0
                    and len(missing_functions) == 0,
                    "schema_version": current_version,
                    "expected_version": self.SCHEMA_VERSION,
                    "missing_tables": list(missing_tables),
                    "missing_extensions": list(missing_extensions),
                    "missing_functions": list(missing_functions),
                    "existing_tables": list(existing_table_names),
                    "existing_extensions": list(existing_extensions),
                    "existing_functions": list(existing_functions),
                }

                if validation_result["valid"]:
                    logger.info("Tenant database schema validation passed")
                else:
                    logger.warning(
                        f"Tenant database schema validation failed: {validation_result}"
                    )

                return validation_result

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to validate tenant database schema: {e}")
            return {
                "valid": False,
                "error": str(e),
                "schema_version": None,
                "expected_version": self.SCHEMA_VERSION,
            }

    async def get_schema_info(self, database_url: str) -> Dict[str, Any]:
        """
        Get detailed schema information for a tenant database.

        Args:
            database_url: PostgreSQL connection URL for the tenant's database

        Returns:
            Dict with detailed schema information
        """
        try:
            conn = await asyncpg.connect(database_url)

            try:
                # Get table information
                tables_info = await conn.fetch("""
                    SELECT 
                        t.table_name,
                        t.table_type,
                        pg_size_pretty(pg_total_relation_size(c.oid)) as size
                    FROM information_schema.tables t
                    LEFT JOIN pg_class c ON c.relname = t.table_name
                    WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
                    ORDER BY t.table_name
                """)

                # Get row counts for each table
                table_stats = {}
                for table_info in tables_info:
                    table_name = table_info["table_name"]
                    try:
                        count_result = await conn.fetchrow(
                            f'SELECT COUNT(*) as count FROM "{table_name}"'
                        )
                        table_stats[table_name] = {
                            "row_count": count_result["count"],
                            "size": table_info["size"],
                        }
                    except Exception:
                        table_stats[table_name] = {
                            "row_count": "error",
                            "size": table_info["size"],
                        }

                # Get index information
                indexes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename, indexname
                """)

                return {
                    "tables": [dict(row) for row in tables_info],
                    "table_stats": table_stats,
                    "indexes": [dict(row) for row in indexes],
                    "total_tables": len(tables_info),
                }

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {"error": str(e)}


# Global schema initializer instance
_schema_initializer: Optional[TenantSchemaInitializer] = None


def get_schema_initializer() -> TenantSchemaInitializer:
    """Get the global schema initializer instance"""
    global _schema_initializer

    if _schema_initializer is None:
        _schema_initializer = TenantSchemaInitializer()

    return _schema_initializer
