-- Multi-Tenant Catalog Database Schema
-- Project-per-Tenant Architecture following Official Neon Best Practices
-- https://neon.tech/docs/guides/multi-tenancy

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tenant projects mapping table (core control plane)
CREATE TABLE tenant_projects (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_name VARCHAR(255) NOT NULL,
    tenant_email VARCHAR(255) UNIQUE NOT NULL,
    neon_project_id VARCHAR(100) NOT NULL UNIQUE,
    neon_database_url TEXT NOT NULL,
    region VARCHAR(50) NOT NULL DEFAULT 'aws-us-east-1',
    status VARCHAR(20) DEFAULT 'active',
    plan VARCHAR(50) DEFAULT 'basic',
    max_documents INTEGER DEFAULT 1000,
    max_storage_mb INTEGER DEFAULT 500,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tenant configuration store
CREATE TABLE tenant_configs (
    tenant_id UUID PRIMARY KEY REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
    settings JSONB DEFAULT '{}',
    feature_flags JSONB DEFAULT '{}',
    api_limits JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tenant usage tracking for billing and monitoring
CREATE TABLE tenant_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
    metric_name VARCHAR(50) NOT NULL,
    metric_value BIGINT NOT NULL,
    period_date DATE NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, metric_name, period_date)
);

-- Tenant API keys and authentication
CREATE TABLE tenant_api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSONB DEFAULT '[]',
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(tenant_id, key_name)
);

-- Performance indexes
CREATE INDEX idx_tenant_projects_status ON tenant_projects(status);
CREATE INDEX idx_tenant_projects_created ON tenant_projects(created_at);
CREATE INDEX idx_tenant_projects_email ON tenant_projects(tenant_email);
CREATE INDEX idx_tenant_usage_period ON tenant_usage(tenant_id, period_date);
CREATE INDEX idx_tenant_usage_metric ON tenant_usage(metric_name, period_date);
CREATE INDEX idx_tenant_api_keys_tenant ON tenant_api_keys(tenant_id);
CREATE INDEX idx_tenant_api_keys_hash ON tenant_api_keys(key_hash);

-- Tenant status constraint
ALTER TABLE tenant_projects ADD CONSTRAINT valid_status 
    CHECK (status IN ('active', 'suspended', 'pending', 'terminated', 'deleting'));

-- Plan constraint
ALTER TABLE tenant_projects ADD CONSTRAINT valid_plan 
    CHECK (plan IN ('basic', 'pro', 'premium', 'enterprise', 'trial'));

-- Update trigger for tenant_projects
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenant_projects_modtime 
    BEFORE UPDATE ON tenant_projects 
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_tenant_configs_modtime 
    BEFORE UPDATE ON tenant_configs 
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Tenant operation logs for audit trail
CREATE TABLE tenant_operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
    operation_type VARCHAR(50) NOT NULL,
    operation_details JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    performed_by VARCHAR(255)
);

-- Index for operation logs
CREATE INDEX idx_tenant_operations_tenant ON tenant_operations(tenant_id, started_at);
CREATE INDEX idx_tenant_operations_type ON tenant_operations(operation_type, status);

-- Operation status constraint
ALTER TABLE tenant_operations ADD CONSTRAINT valid_operation_status 
    CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled'));

-- Trigger for operation completion timestamp
CREATE OR REPLACE FUNCTION update_operation_completion()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('completed', 'failed', 'cancelled') AND OLD.completed_at IS NULL THEN
        NEW.completed_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_operation_completion_time 
    BEFORE UPDATE ON tenant_operations 
    FOR EACH ROW EXECUTE FUNCTION update_operation_completion();

-- Views for common queries
CREATE VIEW active_tenants AS
SELECT 
    tenant_id,
    tenant_name,
    tenant_email,
    plan,
    created_at,
    (SELECT COUNT(*) FROM tenant_usage tu WHERE tu.tenant_id = tp.tenant_id AND metric_name = 'documents_ingested') as total_documents
FROM tenant_projects tp
WHERE status = 'active';

CREATE VIEW tenant_usage_summary AS
SELECT 
    tp.tenant_id,
    tp.tenant_name,
    tp.plan,
    tp.max_documents,
    tp.max_storage_mb,
    COALESCE(SUM(CASE WHEN tu.metric_name = 'documents_ingested' THEN tu.metric_value ELSE 0 END), 0) as total_documents,
    COALESCE(SUM(CASE WHEN tu.metric_name = 'storage_used_mb' THEN tu.metric_value ELSE 0 END), 0) as total_storage_mb,
    COALESCE(SUM(CASE WHEN tu.metric_name = 'api_requests' THEN tu.metric_value ELSE 0 END), 0) as total_api_requests
FROM tenant_projects tp
LEFT JOIN tenant_usage tu ON tp.tenant_id = tu.tenant_id
WHERE tp.status = 'active'
GROUP BY tp.tenant_id, tp.tenant_name, tp.plan, tp.max_documents, tp.max_storage_mb;

-- Comments for documentation
COMMENT ON TABLE tenant_projects IS 'Maps tenants to their dedicated Neon projects (project-per-tenant)';
COMMENT ON TABLE tenant_configs IS 'Per-tenant configuration and feature flags';
COMMENT ON TABLE tenant_usage IS 'Tracks tenant usage metrics for billing and monitoring';
COMMENT ON TABLE tenant_api_keys IS 'Tenant-specific API keys for authentication';
COMMENT ON TABLE tenant_operations IS 'Audit log for all tenant operations';

COMMENT ON COLUMN tenant_projects.neon_project_id IS 'Neon project ID for this tenant (unique database)';
COMMENT ON COLUMN tenant_projects.neon_database_url IS 'Direct connection URL to tenant database';
COMMENT ON COLUMN tenant_projects.region IS 'Neon region where tenant project is deployed';
