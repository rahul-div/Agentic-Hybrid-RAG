-- Per-Tenant Database Schema Template
-- Each tenant gets their own dedicated Neon project with this schema
-- NO tenant_id columns needed - entire database belongs to one tenant
-- Based on official Neon project-per-tenant patterns

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table (NO tenant_id - entire DB is for one tenant)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks with vector embeddings (NO tenant_id needed)
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding VECTOR(768), -- OpenAI ada-002 dimension
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sessions for conversation tracking
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW()
);

-- Messages for conversation history
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    content TEXT NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX idx_documents_created ON documents(created_at);
CREATE INDEX idx_documents_title ON documents USING GIN(to_tsvector('english', title));
CREATE INDEX idx_documents_content ON documents USING GIN(to_tsvector('english', content));

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_created ON chunks(created_at);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_activity ON sessions(last_activity);

CREATE INDEX idx_messages_session ON messages(session_id, created_at);
CREATE INDEX idx_messages_user ON messages(user_id, created_at);

-- Vector search function (tenant-isolated by design)
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding VECTOR(768),
    match_count INTEGER DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.7
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        chunks.id,
        chunks.document_id,
        chunks.content,
        1 - (chunks.embedding <=> query_embedding) AS similarity
    FROM chunks
    WHERE 1 - (chunks.embedding <=> query_embedding) > similarity_threshold
    ORDER BY chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Full text search function
CREATE OR REPLACE FUNCTION search_documents(
    search_query TEXT,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    document_id UUID,
    title TEXT,
    content TEXT,
    rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.title,
        documents.content,
        ts_rank(
            to_tsvector('english', documents.title || ' ' || documents.content),
            plainto_tsquery('english', search_query)
        ) AS rank
    FROM documents
    WHERE to_tsvector('english', documents.title || ' ' || documents.content) 
          @@ plainto_tsquery('english', search_query)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$;

-- Update trigger for documents
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_modtime 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Update last_activity for sessions
CREATE OR REPLACE FUNCTION update_session_activity()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE sessions 
    SET last_activity = NOW() 
    WHERE id = NEW.session_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_session_activity_trigger
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION update_session_activity();

-- Comments for documentation
COMMENT ON TABLE documents IS 'Documents stored in this tenant database (isolated)';
COMMENT ON TABLE chunks IS 'Document chunks with vector embeddings for semantic search';
COMMENT ON TABLE sessions IS 'User conversation sessions';
COMMENT ON TABLE messages IS 'Messages within conversation sessions';

COMMENT ON FUNCTION match_chunks IS 'Vector similarity search for document chunks';
COMMENT ON FUNCTION search_documents IS 'Full-text search across documents';
