"""
Catalog Database Connection Pool Manager
Handles connections to the catalog database for tenant management operations.
"""

import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import asyncpg
from asyncpg.pool import Pool
import os
from dataclasses import dataclass
from datetime import datetime, date
from uuid import UUID

logger = logging.getLogger(__name__)


@dataclass
class TenantProjectInfo:
    """Tenant project information model"""

    tenant_id: UUID
    tenant_name: str
    tenant_email: str
    neon_project_id: str
    neon_database_url: str
    region: str
    status: str
    plan: str
    max_documents: int
    max_storage_mb: int
    created_at: datetime
    updated_at: datetime


@dataclass
class TenantUsageRecord:
    """Tenant usage record model"""

    id: UUID
    tenant_id: UUID
    metric_name: str
    metric_value: int
    period_date: date
    recorded_at: datetime


@dataclass
class TenantApiKey:
    """Tenant API key model"""

    id: UUID
    tenant_id: UUID
    key_name: str
    key_hash: str
    permissions: List[str]
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime


class CatalogDatabase:
    """
    Manages connections and operations for the catalog database.
    Handles tenant project mappings, configurations, and usage tracking.
    """

    def __init__(
        self, database_url: str, min_connections: int = 5, max_connections: int = 20
    ):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Optional[Pool] = None

    async def initialize(self):
        """Initialize the connection pool"""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                command_timeout=60,
                server_settings={
                    "jit": "off",  # Disable JIT for connection pool
                    "application_name": "multi_tenant_catalog",
                },
            )
            logger.info(
                f"Catalog database pool initialized with {self.min_connections}-{self.max_connections} connections"
            )

            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.info("Catalog database connection test successful")

        except Exception as e:
            logger.error(f"Failed to initialize catalog database pool: {e}")
            raise

    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            logger.info("Catalog database pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool"""
        if not self._pool:
            raise RuntimeError("Database pool not initialized")

        async with self._pool.acquire() as conn:
            yield conn

    # ===== TENANT PROJECT OPERATIONS =====

    async def store_tenant_project(
        self,
        name: str,
        email: str,
        neon_project_id: str,
        database_url: str,
        region: str,
        plan: str = "basic",
        max_documents: int = 1000,
        max_storage_mb: int = 500,
    ) -> UUID:
        """Store new tenant-project mapping"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Insert tenant project
                result = await conn.fetchrow(
                    """
                    INSERT INTO tenant_projects 
                    (tenant_name, tenant_email, neon_project_id, neon_database_url, 
                     region, plan, max_documents, max_storage_mb)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING tenant_id
                """,
                    name,
                    email,
                    neon_project_id,
                    database_url,
                    region,
                    plan,
                    max_documents,
                    max_storage_mb,
                )

                tenant_id = result["tenant_id"]

                # Initialize tenant configuration
                await conn.execute(
                    """
                    INSERT INTO tenant_configs (tenant_id, settings, feature_flags, api_limits)
                    VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
                """,
                    tenant_id,
                    "{}",  # JSON string for JSONB columns
                    "{}",
                    "{}",
                )

                # Log operation
                await self._log_operation(
                    conn,
                    tenant_id,
                    "tenant_creation",
                    {"neon_project_id": neon_project_id, "region": region},
                )

                logger.info(
                    f"Created tenant project mapping: {tenant_id} -> {neon_project_id}"
                )
                return tenant_id

    async def get_tenant_project(self, tenant_id: UUID) -> Optional[TenantProjectInfo]:
        """Get tenant project information"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM tenant_projects WHERE tenant_id = $1
            """,
                tenant_id,
            )

            if row:
                return TenantProjectInfo(**dict(row))
            return None

    async def get_tenant_by_email(self, email: str) -> Optional[TenantProjectInfo]:
        """Get tenant by email"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM tenant_projects WHERE tenant_email = $1
            """,
                email,
            )

            if row:
                return TenantProjectInfo(**dict(row))
            return None

    async def list_tenant_projects(
        self, limit: int = 100, offset: int = 0, status_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List tenant projects with pagination and optional status filter"""
        async with self.get_connection() as conn:
            if status_filter:
                rows = await conn.fetch(
                    """
                    SELECT * FROM tenant_projects 
                    WHERE status = $1
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                """,
                    status_filter,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM tenant_projects 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                """,
                    limit,
                    offset,
                )

            return [dict(row) for row in rows]

    async def update_tenant_status(
        self, tenant_id: UUID, status: str, reason: str = None
    ):
        """Update tenant status"""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                UPDATE tenant_projects 
                SET status = $2, updated_at = NOW()
                WHERE tenant_id = $1
            """,
                tenant_id,
                status,
            )

    async def update_tenant_limits(
        self,
        tenant_id: UUID,
        plan: Optional[str] = None,
        max_documents: Optional[int] = None,
        max_storage_mb: Optional[int] = None,
    ):
        """Update tenant plan and limits"""
        async with self.get_connection() as conn:
            updates = []
            params = []
            param_count = 1

            if plan is not None:
                updates.append(f"plan = ${param_count + 1}")
                params.append(plan)
                param_count += 1

            if max_documents is not None:
                updates.append(f"max_documents = ${param_count + 1}")
                params.append(max_documents)
                param_count += 1

            if max_storage_mb is not None:
                updates.append(f"max_storage_mb = ${param_count + 1}")
                params.append(max_storage_mb)
                param_count += 1

            if updates:
                updates.append("updated_at = NOW()")
                query = f"""
                    UPDATE tenant_projects 
                    SET {", ".join(updates)}
                    WHERE tenant_id = $1
                """
                await conn.execute(query, tenant_id, *params)

    async def delete_tenant_project(self, tenant_id: UUID):
        """Delete tenant project mapping (cascades to related data)"""
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Log operation before deletion
                await self._log_operation(conn, tenant_id, "tenant_deletion", {})

                # Delete tenant (cascades to configs, usage, api_keys)
                result = await conn.execute(
                    """
                    DELETE FROM tenant_projects WHERE tenant_id = $1
                """,
                    tenant_id,
                )

                logger.info(f"Deleted tenant project: {tenant_id}")
                return result

    # ===== TENANT CONFIGURATION OPERATIONS =====

    async def get_tenant_config(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get tenant configuration"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT settings, feature_flags, api_limits 
                FROM tenant_configs 
                WHERE tenant_id = $1
            """,
                tenant_id,
            )

            if row:
                return {
                    "settings": dict(row["settings"]),
                    "feature_flags": dict(row["feature_flags"]),
                    "api_limits": dict(row["api_limits"]),
                }
            return {}

    async def update_tenant_config(
        self, tenant_id: UUID, config_type: str, config_data: Dict[str, Any]
    ):
        """Update tenant configuration"""
        async with self.get_connection() as conn:
            await conn.execute(
                f"""
                UPDATE tenant_configs 
                SET {config_type} = $1, updated_at = NOW()
                WHERE tenant_id = $2
            """,
                config_data,
                tenant_id,
            )

    # ===== TENANT USAGE OPERATIONS =====

    async def record_tenant_usage(
        self,
        tenant_id: UUID,
        metric_name: str,
        metric_value: int,
        period_date: date = None,
    ):
        """Record tenant usage metric"""
        if period_date is None:
            period_date = date.today()

        async with self.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO tenant_usage (tenant_id, metric_name, metric_value, period_date)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (tenant_id, metric_name, period_date)
                DO UPDATE SET 
                    metric_value = tenant_usage.metric_value + EXCLUDED.metric_value,
                    recorded_at = NOW()
            """,
                tenant_id,
                metric_name,
                metric_value,
                period_date,
            )

    async def get_tenant_usage_stats(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get aggregated usage statistics for tenant"""
        async with self.get_connection() as conn:
            # Get latest usage metrics
            usage_rows = await conn.fetch(
                """
                SELECT metric_name, SUM(metric_value) as total_value
                FROM tenant_usage 
                WHERE tenant_id = $1 
                GROUP BY metric_name
            """,
                tenant_id,
            )

            # Convert to dictionary
            usage_stats = {row["metric_name"]: row["total_value"] for row in usage_rows}

            # Get last activity from operations
            last_activity = await conn.fetchval(
                """
                SELECT MAX(started_at) 
                FROM tenant_operations 
                WHERE tenant_id = $1
            """,
                tenant_id,
            )

            return {
                "documents_count": usage_stats.get("documents_ingested", 0),
                "chunks_count": usage_stats.get("chunks_created", 0),
                "storage_used_mb": usage_stats.get("storage_used_mb", 0.0),
                "queries_count": usage_stats.get("queries_executed", 0),
                "last_activity": last_activity,
            }

    # ===== API KEY OPERATIONS =====

    async def store_api_key(
        self,
        tenant_id: UUID,
        key_name: str,
        key_hash: str,
        permissions: List[str] = None,
        expires_at: datetime = None,
    ) -> UUID:
        """Store tenant API key"""
        if permissions is None:
            permissions = []

        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO tenant_api_keys 
                (tenant_id, key_name, key_hash, permissions, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """,
                tenant_id,
                key_name,
                key_hash,
                permissions,
                expires_at,
            )

            return result["id"]

    async def get_tenant_by_api_key_hash(self, key_hash: str) -> Optional[UUID]:
        """Get tenant ID by API key hash"""
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT tak.tenant_id 
                FROM tenant_api_keys tak
                JOIN tenant_projects tp ON tak.tenant_id = tp.tenant_id
                WHERE tak.key_hash = $1 
                AND tp.status = 'active'
                AND (tak.expires_at IS NULL OR tak.expires_at > NOW())
            """,
                key_hash,
            )

            if row:
                # Update last used timestamp
                await conn.execute(
                    """
                    UPDATE tenant_api_keys 
                    SET last_used_at = NOW() 
                    WHERE key_hash = $1
                """,
                    key_hash,
                )

                return row["tenant_id"]
            return None

    async def list_tenant_api_keys(self, tenant_id: UUID) -> List[TenantApiKey]:
        """List API keys for tenant"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM tenant_api_keys 
                WHERE tenant_id = $1 
                ORDER BY created_at DESC
            """,
                tenant_id,
            )

            return [TenantApiKey(**dict(row)) for row in rows]

    async def revoke_api_key(self, tenant_id: UUID, key_name: str):
        """Revoke (delete) API key"""
        async with self.get_connection() as conn:
            await conn.execute(
                """
                DELETE FROM tenant_api_keys 
                WHERE tenant_id = $1 AND key_name = $2
            """,
                tenant_id,
                key_name,
            )

    # ===== AUDIT LOG OPERATIONS =====

    async def _log_operation(
        self,
        conn,
        tenant_id: UUID,
        operation_type: str,
        operation_details: Dict[str, Any],
        performed_by: str = "system",
    ):
        """Log tenant operation for audit trail"""
        import json

        # Convert dict to JSON string for PostgreSQL JSONB column
        details_json = json.dumps(operation_details) if operation_details else "{}"

        await conn.execute(
            """
            INSERT INTO tenant_operations 
            (tenant_id, operation_type, operation_details, performed_by)
            VALUES ($1, $2, $3::jsonb, $4)
        """,
            tenant_id,
            operation_type,
            details_json,
            performed_by,
        )

    async def get_tenant_operations(
        self, tenant_id: UUID, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get tenant operation history"""
        async with self.get_connection() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM tenant_operations 
                WHERE tenant_id = $1 
                ORDER BY started_at DESC 
                LIMIT $2 OFFSET $3
            """,
                tenant_id,
                limit,
                offset,
            )

            return [dict(row) for row in rows]

    async def update_tenant_project(
        self,
        tenant_id: UUID,
        tenant_name: Optional[str] = None,
        status: Optional[str] = None,
        plan: Optional[str] = None,
        max_documents: Optional[int] = None,
        max_storage_mb: Optional[int] = None,
    ):
        """Update tenant project information with flexible field updates"""
        async with self.get_connection() as conn:
            updates = []
            params = []
            param_count = 1

            if tenant_name is not None:
                updates.append(f"tenant_name = ${param_count + 1}")
                params.append(tenant_name)
                param_count += 1

            if status is not None:
                updates.append(f"status = ${param_count + 1}")
                params.append(status)
                param_count += 1

            if plan is not None:
                updates.append(f"plan = ${param_count + 1}")
                params.append(plan)
                param_count += 1

            if max_documents is not None:
                updates.append(f"max_documents = ${param_count + 1}")
                params.append(max_documents)
                param_count += 1

            if max_storage_mb is not None:
                updates.append(f"max_storage_mb = ${param_count + 1}")
                params.append(max_storage_mb)
                param_count += 1

            if updates:
                updates.append("updated_at = NOW()")
                query = f"""
                    UPDATE tenant_projects 
                    SET {", ".join(updates)}
                    WHERE tenant_id = $1
                """
                await conn.execute(query, tenant_id, *params)


# Global catalog database instance
_catalog_db: Optional[CatalogDatabase] = None


async def get_catalog_database() -> CatalogDatabase:
    """Get the global catalog database instance"""
    global _catalog_db

    if _catalog_db is None:
        database_url = os.getenv("CATALOG_DATABASE_URL")
        if not database_url:
            raise ValueError("CATALOG_DATABASE_URL environment variable not set")

        _catalog_db = CatalogDatabase(database_url)
        await _catalog_db.initialize()

    return _catalog_db


async def close_catalog_database():
    """Close the global catalog database instance"""
    global _catalog_db

    if _catalog_db:
        await _catalog_db.close()
        _catalog_db = None
