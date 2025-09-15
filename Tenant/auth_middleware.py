"""
Authentication and Authorization Middleware for Multi-Tenant RAG System
Provides JWT-based authentication with tenant context extraction.
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta

try:
    from fastapi import HTTPException, status
    from fastapi.security import HTTPBearer
    from jose import jwt, JWTError
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
except ImportError:
    print(
        "Warning: FastAPI/Starlette dependencies not installed. Install with: pip install fastapi python-jose[cryptography]"
    )

    # Mock classes for development
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail

    class HTTPBearer:
        pass

    class BaseHTTPMiddleware:
        def __init__(self, app):
            self.app = app

    class Request:
        def __init__(self):
            self.headers = {}
            self.url = None

    class Response:
        def __init__(self, content=None, status_code=200):
            self.content = content
            self.status_code = status_code

    class JWTError(Exception):
        pass

    status = type(
        "Status",
        (),
        {
            "HTTP_401_UNAUTHORIZED": 401,
            "HTTP_403_FORBIDDEN": 403,
            "HTTP_429_TOO_MANY_REQUESTS": 429,
        },
    )()

    jwt = type(
        "JWT",
        (),
        {
            "encode": lambda payload, key, algorithm: "mock_token",
            "decode": lambda token, key, algorithms: {"tenant_id": "mock"},
        },
    )()

logger = logging.getLogger(__name__)


class TenantContext:
    """Tenant context extracted from JWT token."""

    def __init__(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        permissions: List[str] = None,
        metadata: Dict[str, Any] = None,
        session_id: Optional[str] = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.permissions = permissions or ["read"]
        self.metadata = metadata or {}
        self.session_id = session_id
        self.authenticated_at = datetime.utcnow()

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def is_admin(self) -> bool:
        """Check if user has admin permissions."""
        return "admin" in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "authenticated_at": self.authenticated_at.isoformat(),
        }


class JWTManager:
    """JWT token management with tenant support."""

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        token_expiry_hours: int = 24,
        refresh_expiry_days: int = 7,
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry_hours = token_expiry_hours
        self.refresh_expiry_days = refresh_expiry_days

        # Rate limiting and security
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 15

    def create_access_token(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        permissions: List[str] = None,
        metadata: Dict[str, Any] = None,
        custom_expiry: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token."""
        try:
            expire_delta = custom_expiry or timedelta(hours=self.token_expiry_hours)
            expire = datetime.utcnow() + expire_delta

            payload = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "permissions": permissions or ["read"],
                "metadata": metadata or {},
                "iat": datetime.utcnow(),
                "exp": expire,
                "type": "access",
            }

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created access token for tenant {tenant_id}")
            return token

        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise

    def create_refresh_token(
        self, tenant_id: str, user_id: Optional[str] = None
    ) -> str:
        """Create JWT refresh token."""
        try:
            expire = datetime.utcnow() + timedelta(days=self.refresh_expiry_days)

            payload = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "iat": datetime.utcnow(),
                "exp": expire,
                "type": "refresh",
            }

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"Created refresh token for tenant {tenant_id}")
            return token

        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check token type
            if payload.get("type") != "access":
                raise JWTError("Invalid token type")

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token has expired")

            return payload

        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise JWTError("Invalid token")

    def extract_tenant_context(self, token: str) -> TenantContext:
        """Extract tenant context from token."""
        try:
            payload = self.verify_token(token)

            return TenantContext(
                tenant_id=payload["tenant_id"],
                user_id=payload.get("user_id"),
                permissions=payload.get("permissions", ["read"]),
                metadata=payload.get("metadata", {}),
                session_id=payload.get("session_id"),
            )

        except Exception as e:
            logger.error(f"Failed to extract tenant context: {e}")
            raise

    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        try:
            payload = jwt.decode(
                refresh_token, self.secret_key, algorithms=[self.algorithm]
            )

            # Verify it's a refresh token
            if payload.get("type") != "refresh":
                raise JWTError("Invalid refresh token")

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Refresh token has expired")

            # Create new access token
            return self.create_access_token(
                tenant_id=payload["tenant_id"], user_id=payload.get("user_id")
            )

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            raise

    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited."""
        now = datetime.utcnow()

        # Clean old attempts
        if identifier in self.failed_attempts:
            cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
            self.failed_attempts[identifier] = [
                attempt
                for attempt in self.failed_attempts[identifier]
                if attempt > cutoff
            ]

        # Check rate limit
        attempts = self.failed_attempts.get(identifier, [])
        return len(attempts) >= self.max_failed_attempts

    def record_failed_attempt(self, identifier: str):
        """Record a failed authentication attempt."""
        now = datetime.utcnow()

        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []

        self.failed_attempts[identifier].append(now)
        logger.warning(f"Failed authentication attempt for {identifier}")

    def clear_failed_attempts(self, identifier: str):
        """Clear failed attempts for identifier."""
        if identifier in self.failed_attempts:
            del self.failed_attempts[identifier]


class TenantAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for tenant authentication and context injection."""

    def __init__(
        self,
        app,
        jwt_manager: JWTManager,
        excluded_paths: List[str] = None,
        require_auth: bool = True,
    ):
        super().__init__(app)
        self.jwt_manager = jwt_manager
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/auth/token",
        ]
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tenant authentication."""
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        try:
            # Extract authorization header
            auth_header = request.headers.get("Authorization")

            if not auth_header:
                if self.require_auth:
                    return self._create_error_response(
                        "Missing authorization header", status.HTTP_401_UNAUTHORIZED
                    )
                else:
                    # Continue without auth context
                    return await call_next(request)

            # Parse bearer token
            if not auth_header.startswith("Bearer "):
                return self._create_error_response(
                    "Invalid authorization header format", status.HTTP_401_UNAUTHORIZED
                )

            token = auth_header.split(" ")[1]

            # Check rate limiting
            if self.jwt_manager.is_rate_limited(
                token[:20]
            ):  # Use token prefix as identifier
                return self._create_error_response(
                    "Too many failed attempts. Please try again later.",
                    status.HTTP_429_TOO_MANY_REQUESTS,
                )

            # Extract tenant context
            try:
                tenant_context = self.jwt_manager.extract_tenant_context(token)

                # Add tenant context to request state
                request.state.tenant_context = tenant_context
                request.state.authenticated = True

                # Log successful authentication
                logger.info(
                    f"Authenticated request for tenant {tenant_context.tenant_id}"
                )

                # Clear any failed attempts for this token
                self.jwt_manager.clear_failed_attempts(token[:20])

            except JWTError as e:
                # Record failed attempt
                self.jwt_manager.record_failed_attempt(token[:20])

                return self._create_error_response(
                    f"Authentication failed: {str(e)}", status.HTTP_401_UNAUTHORIZED
                )

            # Continue with the request
            response = await call_next(request)

            # Add tenant context to response headers (optional)
            if hasattr(request.state, "tenant_context"):
                response.headers["X-Tenant-ID"] = tenant_context.tenant_id
                response.headers["X-Auth-Context"] = "authenticated"

            return response

        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return self._create_error_response(
                "Internal authentication error", status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _create_error_response(self, detail: str, status_code: int) -> Response:
        """Create error response."""
        try:
            from starlette.responses import JSONResponse

            return JSONResponse(
                status_code=status_code,
                content={"error": detail, "timestamp": datetime.utcnow().isoformat()},
            )
        except ImportError:
            # Fallback for mock environment
            return Response(
                content=f'{{"error": "{detail}", "timestamp": "{datetime.utcnow().isoformat()}"}}',
                status_code=status_code,
                media_type="application/json",
            )


class TenantSecurityManager:
    """Advanced security manager for multi-tenant operations."""

    def __init__(self, jwt_manager: JWTManager):
        self.jwt_manager = jwt_manager
        self.tenant_permissions: Dict[str, List[str]] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def set_tenant_permissions(self, tenant_id: str, permissions: List[str]):
        """Set default permissions for a tenant."""
        self.tenant_permissions[tenant_id] = permissions
        self._audit_log("permission_update", tenant_id, {"permissions": permissions})

    def validate_tenant_access(
        self,
        tenant_context: TenantContext,
        required_permission: str,
        resource_id: Optional[str] = None,
    ) -> bool:
        """Validate if tenant has access to perform operation."""
        try:
            # Check if user has required permission
            if not tenant_context.has_permission(required_permission):
                self._audit_log(
                    "access_denied",
                    tenant_context.tenant_id,
                    {
                        "required_permission": required_permission,
                        "user_permissions": tenant_context.permissions,
                        "resource_id": resource_id,
                    },
                )
                return False

            # Additional tenant-level validation
            tenant_perms = self.tenant_permissions.get(tenant_context.tenant_id, [])
            if tenant_perms and required_permission not in tenant_perms:
                self._audit_log(
                    "tenant_permission_denied",
                    tenant_context.tenant_id,
                    {
                        "required_permission": required_permission,
                        "tenant_permissions": tenant_perms,
                        "resource_id": resource_id,
                    },
                )
                return False

            self._audit_log(
                "access_granted",
                tenant_context.tenant_id,
                {"permission": required_permission, "resource_id": resource_id},
            )
            return True
        except Exception as e:
            logger.error(f"Access validation error: {e}")
            return False

    def validate_cross_tenant_access(
        self, source_tenant: str, target_tenant: str, operation: str
    ) -> bool:
        """Validate cross-tenant operations (should typically be denied)."""
        # Log attempted cross-tenant access
        self._audit_log(
            "cross_tenant_attempt",
            source_tenant,
            {
                "target_tenant": target_tenant,
                "operation": operation,
                "result": "denied",
            },
        )

        # Cross-tenant access is denied by default
        return False

    def _audit_log(self, action: str, tenant_id: str, details: Dict[str, Any]):
        """Log security-related events."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "tenant_id": tenant_id,
            "details": details,
        }

        self.audit_log.append(log_entry)
        logger.info(f"Security audit: {action} for tenant {tenant_id}")

        # Keep only last 1000 entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

    def get_audit_logs(
        self,
        tenant_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit logs with optional filtering."""
        logs = self.audit_log

        if tenant_id:
            logs = [log for log in logs if log["tenant_id"] == tenant_id]

        if action:
            logs = [log for log in logs if log["action"] == action]

        return logs[-limit:]


class JWTAuthenticator:
    """
    Simple JWT authenticator for multi-tenant systems.
    This is a wrapper around JWTManager for easier integration.
    """

    def __init__(self, secret_key: str = None):
        """Initialize JWT authenticator with secret key."""
        import os

        # Use provided secret key or get from environment
        self.secret_key = secret_key or os.getenv(
            "JWT_SECRET_KEY", "your-secret-key-change-in-production"
        )

        # Initialize JWT manager
        self.jwt_manager = JWTManager(
            secret_key=self.secret_key,
            algorithm="HS256",
            token_expiry_hours=24,
            refresh_expiry_days=7,
        )

        # Initialize security manager
        self.security_manager = TenantSecurityManager(self.jwt_manager)

        logger.info("JWT Authenticator initialized")

    def create_token(self, token_data: Dict[str, Any]) -> str:
        """
        Create a JWT token from token data.

        Args:
            token_data: Dictionary containing tenant_id, user_id, permissions, etc.

        Returns:
            JWT token string
        """
        try:
            return self.jwt_manager.create_access_token(
                tenant_id=token_data.get("tenant_id"),
                user_id=token_data.get("user_id"),
                permissions=token_data.get("permissions", ["read"]),
                metadata=token_data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Failed to create token: {e}")
            raise

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            return self.jwt_manager.verify_token(token)
        except Exception as e:
            logger.error(f"Failed to decode token: {e}")
            raise

    def get_tenant_context(self, token: str) -> TenantContext:
        """
        Extract tenant context from JWT token.

        Args:
            token: JWT token string

        Returns:
            TenantContext object
        """
        try:
            return self.jwt_manager.extract_tenant_context(token)
        except Exception as e:
            logger.error(f"Failed to extract tenant context: {e}")
            raise

    def validate_api_key(self, tenant_id: str, api_key: str) -> bool:
        """
        Validate API key for a tenant.
        This is a simple implementation - in production, use proper key management.

        Args:
            tenant_id: Tenant identifier
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        # Simple validation: api_key should be "api_key_{tenant_id}"
        expected_key = f"api_key_{tenant_id}"
        is_valid = api_key == expected_key

        if is_valid:
            logger.info(f"API key validated successfully for tenant {tenant_id}")
        else:
            logger.warning(f"Invalid API key for tenant {tenant_id}")

        return is_valid

    def authenticate_tenant(
        self, tenant_id: str, user_id: str, api_key: str
    ) -> Optional[str]:
        """
        Authenticate a tenant and return JWT token.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            api_key: API key for authentication

        Returns:
            JWT token if authentication successful, None otherwise
        """
        try:
            # Validate API key
            if not self.validate_api_key(tenant_id, api_key):
                return None

            # Create token data
            token_data = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "permissions": ["read", "write"],  # Default permissions
                "metadata": {
                    "auth_method": "api_key",
                    "authenticated_at": datetime.utcnow().isoformat(),
                },
            }

            # Create and return token
            return self.create_token(token_data)

        except Exception as e:
            logger.error(f"Authentication failed for tenant {tenant_id}: {e}")
            return None

    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Refresh token string

        Returns:
            New access token if successful, None otherwise
        """
        try:
            return self.jwt_manager.refresh_access_token(refresh_token)
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    def is_token_valid(self, token: str) -> bool:
        """
        Check if a token is valid without decoding it fully.

        Args:
            token: JWT token string

        Returns:
            True if valid, False otherwise
        """
        try:
            self.jwt_manager.verify_token(token)
            return True
        except Exception:
            return False

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a token without throwing exceptions.

        Args:
            token: JWT token string

        Returns:
            Token information dict or None if invalid
        """
        try:
            payload = self.decode_token(token)
            return {
                "tenant_id": payload.get("tenant_id"),
                "user_id": payload.get("user_id"),
                "permissions": payload.get("permissions", []),
                "expires_at": (
                    datetime.fromtimestamp(payload.get("exp", 0)).isoformat()
                    if payload.get("exp")
                    else None
                ),
                "issued_at": (
                    datetime.fromtimestamp(payload.get("iat", 0)).isoformat()
                    if payload.get("iat")
                    else None
                ),
            }
        except Exception:
            return None


# Convenience functions and decorators


def require_permission(permission: str):
    """Decorator to require specific permission for endpoint."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented based on your framework
            # For FastAPI, you'd use Depends() with a permission checker
            pass

        return wrapper

    return decorator


def get_tenant_context_from_request(request) -> Optional[TenantContext]:
    """Extract tenant context from request state."""
    return getattr(request.state, "tenant_context", None)


# Example usage
if __name__ == "__main__":
    # Example JWT manager usage
    jwt_manager = JWTManager("your-secret-key")

    # Create access token
    token = jwt_manager.create_access_token(
        tenant_id="demo_corp", user_id="user_123", permissions=["read", "write"]
    )
    print(f"Created token: {token}")

    # Extract tenant context
    context = jwt_manager.extract_tenant_context(token)
    print(f"Tenant context: {context.to_dict()}")

    # Security manager example
    security_manager = TenantSecurityManager(jwt_manager)
    security_manager.set_tenant_permissions("demo_corp", ["read", "write", "admin"])

    # Validate access
    has_access = security_manager.validate_tenant_access(
        context, "write", "document_123"
    )
    print(f"Has write access: {has_access}")
