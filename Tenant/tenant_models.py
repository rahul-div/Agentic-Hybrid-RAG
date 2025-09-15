"""
Production-Ready Data Models for Multi-Tenant RAG System
Integrates with existing tenant_manager and provides proper database schema.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Integer,
    Boolean,
    Text,
    ForeignKey,
    JSON,
    ARRAY,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Tenant(Base):
    """Tenant model with Neon project integration."""

    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    admin_email = Column(String(255), nullable=False)
    neon_project_id = Column(String(255), nullable=True)
    neon_database_url = Column(Text, nullable=True)
    status = Column(String(50), default="active")
    plan = Column(String(50), default="basic")
    max_documents = Column(Integer, default=1000)
    max_storage_mb = Column(Integer, default=500)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    users = relationship(
        "TenantUser", back_populates="tenant", cascade="all, delete-orphan"
    )
    api_keys = relationship(
        "APIKey", back_populates="tenant", cascade="all, delete-orphan"
    )
    audit_logs = relationship("AuditLog", back_populates="tenant")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "admin_email": self.admin_email,
            "neon_project_id": self.neon_project_id,
            "status": self.status,
            "plan": self.plan,
            "max_documents": self.max_documents,
            "max_storage_mb": self.max_storage_mb,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class TenantUser(Base):
    """User model within tenant context."""

    __tablename__ = "tenant_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    email = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)
    role = Column(String(50), default="user")  # admin, user, readonly
    status = Column(String(50), default="active")
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Unique constraint
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="unique_tenant_user_email"),
    )

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "last_login_at": self.last_login_at.isoformat()
            if self.last_login_at
            else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class APIKey(Base):
    """API Key model with proper security."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_id = Column(String(100), unique=True, nullable=False)
    tenant_id = Column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant_users.id", ondelete="CASCADE"),
        nullable=True,
    )
    name = Column(String(255), nullable=False)
    key_prefix = Column(String(20), nullable=False)  # First 12 chars for display
    key_hash = Column(Text, nullable=False)  # PBKDF2 hash
    scopes = Column(ARRAY(String), nullable=False)
    status = Column(String(50), default="active")
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="api_keys")
    user = relationship("TenantUser", back_populates="api_keys")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (excluding sensitive data)."""
        return {
            "id": str(self.id),
            "key_id": self.key_id,
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "name": self.name,
            "key_prefix": self.key_prefix,
            "scopes": self.scopes,
            "status": self.status,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat()
            if self.last_used_at
            else None,
            "usage_count": self.usage_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class AuditLog(Base):
    """Audit log for security and compliance."""

    __tablename__ = "audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("tenant_users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSON, default={})
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="audit_logs")
    user = relationship("TenantUser", back_populates="audit_logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": str(self.ip_address) if self.ip_address else None,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat(),
        }


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, database_url: str):
        """Initialize database connection."""
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self):
        """Get database session."""
        return self.SessionLocal()

    def close(self):
        """Close database connection."""
        self.engine.dispose()


# Database session dependency for FastAPI
def get_db_session(database_manager: DatabaseManager):
    """Dependency to get database session."""
    session = database_manager.get_session()
    try:
        yield session
    finally:
        session.close()


# Repository classes for clean data access
class TenantRepository:
    """Repository for tenant operations."""

    def __init__(self, session):
        self.session = session

    def create(self, tenant_data: Dict[str, Any]) -> Tenant:
        """Create new tenant."""
        tenant = Tenant(**tenant_data)
        self.session.add(tenant)
        self.session.commit()
        self.session.refresh(tenant)
        return tenant

    def get_by_id(self, tenant_id: uuid.UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.session.query(Tenant).filter(Tenant.id == tenant_id).first()

    def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        return self.session.query(Tenant).filter(Tenant.slug == slug).first()

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Tenant]:
        """List all tenants."""
        return self.session.query(Tenant).offset(offset).limit(limit).all()

    def update(self, tenant_id: uuid.UUID, updates: Dict[str, Any]) -> Optional[Tenant]:
        """Update tenant."""
        tenant = self.get_by_id(tenant_id)
        if tenant:
            for key, value in updates.items():
                setattr(tenant, key, value)
            tenant.updated_at = datetime.utcnow()
            self.session.commit()
            self.session.refresh(tenant)
        return tenant

    def delete(self, tenant_id: uuid.UUID) -> bool:
        """Delete tenant."""
        tenant = self.get_by_id(tenant_id)
        if tenant:
            self.session.delete(tenant)
            self.session.commit()
            return True
        return False


class APIKeyRepository:
    """Repository for API key operations."""

    def __init__(self, session):
        self.session = session

    def create(self, api_key_data: Dict[str, Any]) -> APIKey:
        """Create new API key."""
        api_key = APIKey(**api_key_data)
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)
        return api_key

    def get_by_key_id(self, key_id: str) -> Optional[APIKey]:
        """Get API key by key_id."""
        return self.session.query(APIKey).filter(APIKey.key_id == key_id).first()

    def list_tenant_keys(self, tenant_id: uuid.UUID) -> List[APIKey]:
        """List all API keys for a tenant."""
        return self.session.query(APIKey).filter(APIKey.tenant_id == tenant_id).all()

    def update_usage(self, key_id: str) -> bool:
        """Update API key usage statistics."""
        api_key = self.get_by_key_id(key_id)
        if api_key:
            api_key.last_used_at = datetime.utcnow()
            api_key.usage_count += 1
            self.session.commit()
            return True
        return False

    def revoke(self, key_id: str, reason: str = None) -> bool:
        """Revoke API key."""
        api_key = self.get_by_key_id(key_id)
        if api_key:
            api_key.status = "revoked"
            if reason:
                api_key.metadata["revocation_reason"] = reason
            self.session.commit()
            return True
        return False


class AuditRepository:
    """Repository for audit log operations."""

    def __init__(self, session):
        self.session = session

    def log_action(
        self,
        action: str,
        tenant_id: uuid.UUID = None,
        user_id: uuid.UUID = None,
        resource_type: str = None,
        resource_id: str = None,
        details: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
    ) -> AuditLog:
        """Log an action to audit trail."""
        audit_entry = AuditLog(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.session.add(audit_entry)
        self.session.commit()
        return audit_entry

    def get_tenant_logs(
        self, tenant_id: uuid.UUID, limit: int = 100, offset: int = 0
    ) -> List[AuditLog]:
        """Get audit logs for a tenant."""
        return (
            self.session.query(AuditLog)
            .filter(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


# Integration with existing tenant_api_key_manager.py
def integrate_with_existing_system():
    """
    Integration notes:

    1. Replace the in-memory storage in TenantAPIKeyManager with DatabaseManager
    2. Use APIKeyRepository for all API key operations
    3. Add audit logging to all sensitive operations
    4. Integrate with existing tenant_manager.py for Neon project creation

    Example integration:

    ```python
    # In tenant_api_key_manager.py, replace:
    self.api_keys: Dict[str, APIKey] = {}

    # With:
    self.db_manager = DatabaseManager(database_url)
    self.api_key_repo = APIKeyRepository(self.db_manager.get_session())
    ```
    """
    pass


if __name__ == "__main__":
    # Example usage
    import os

    # Initialize database
    db_url = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost/tenant_catalog"
    )
    db_manager = DatabaseManager(db_url)

    # Create tables
    db_manager.create_tables()

    # Create repositories
    session = db_manager.get_session()
    tenant_repo = TenantRepository(session)
    api_key_repo = APIKeyRepository(session)
    audit_repo = AuditRepository(session)

    print("Database models and repositories initialized successfully!")

    session.close()
    db_manager.close()
