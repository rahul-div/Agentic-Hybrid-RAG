"""
Multi-Tenant Project Manager for Neon PostgreSQL
Implements official Neon project-per-tenant architecture for complete isolation.
Based on Neon's primary recommendation for multi-tenant applications.

This module handles:
1. Creating dedicated Neon projects for each tenant
2. Managing project lifecycle (create, monitor, delete)
3. Initializing tenant database schemas
4. Handling Neon API operations with proper error handling and retry logic
"""

import logging
import httpx
import os
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class NeonProjectInfo:
    """Information about a Neon project"""

    project_id: str
    name: str
    database_url: str
    region: str
    status: str
    created_at: datetime
    default_branch_id: str
    default_endpoint_id: str


@dataclass
class TenantCreateRequest:
    """Request model for creating a new tenant"""

    name: str
    email: str
    region: str = "aws-us-east-1"
    plan: str = "basic"
    max_documents: int = 1000
    max_storage_mb: int = 500


class NeonAPIError(Exception):
    """Custom exception for Neon API errors"""

    def __init__(
        self, message: str, status_code: int = None, response_data: Dict = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class NeonProjectManager:
    """
    Manages Neon projects for tenant isolation using official Neon API.

    Each tenant gets a dedicated Neon project (database) for complete isolation.
    This follows Neon's recommended project-per-tenant architecture.
    """

    def __init__(self, api_key: str, default_region: str = "aws-us-east-1"):
        """
        Initialize Neon project manager.

        Args:
            api_key: Neon API key for project management
            default_region: Default region for new projects
        """
        self.api_key = api_key
        self.default_region = default_region
        self.base_url = "https://console.neon.tech/api/v2"
        self.session = None

        # Rate limiting and retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.timeout = 30.0  # seconds

    async def _ensure_session(self):
        """Ensure HTTP session is initialized"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
                verify=False,  # Disable SSL verification for macOS compatibility
            )

    async def _make_request(
        self, method: str, endpoint: str, data: Dict = None, params: Dict = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Neon API with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            API response data

        Raises:
            NeonAPIError: If API request fails after retries
        """
        await self._ensure_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = await self.session.get(url, params=params)
                elif method.upper() == "POST":
                    response = await self.session.post(url, json=data, params=params)
                elif method.upper() == "DELETE":
                    response = await self.session.delete(url, params=params)
                elif method.upper() == "PATCH":
                    response = await self.session.patch(url, json=data, params=params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Handle response
                if response.status_code == 200 or response.status_code == 201:
                    return response.json()
                elif response.status_code == 204:  # No content (successful delete)
                    return {}
                elif response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (
                            2**attempt
                        )  # Exponential backoff
                        logger.warning(
                            f"Rate limited, waiting {wait_time}s before retry {attempt + 1}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise NeonAPIError(
                            f"Rate limited after {self.max_retries} attempts",
                            response.status_code,
                            response.json() if response.content else {},
                        )
                else:
                    error_data = response.json() if response.content else {}
                    raise NeonAPIError(
                        f"Neon API error: {response.status_code} - {error_data.get('message', 'Unknown error')}",
                        response.status_code,
                        error_data,
                    )

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Request error: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise NeonAPIError(
                        f"Network error after {self.max_retries} attempts: {e}"
                    )

        # Should never reach here
        raise NeonAPIError("Unexpected error in request handling")

    async def create_tenant_project(
        self, tenant_name: str, region: str = None
    ) -> NeonProjectInfo:
        """
        Create a dedicated Neon project for a tenant.

        Args:
            tenant_name: Name of the tenant (will be sanitized for project name)
            region: Neon region for the project

        Returns:
            NeonProjectInfo with project details

        Raises:
            NeonAPIError: If project creation fails
        """
        if region is None:
            region = self.default_region

        # Sanitize tenant name for project name
        project_name = (
            f"tenant-{tenant_name.lower().replace(' ', '-').replace('_', '-')}"
        )
        project_name = "".join(c for c in project_name if c.isalnum() or c in "-")

        # Ensure project name is not too long (Neon has limits)
        if len(project_name) > 50:
            project_name = project_name[:50].rstrip("-")

        logger.info(
            f"Creating Neon project for tenant: {tenant_name} -> {project_name}"
        )

        try:
            # Get organization ID first
            org_id = await self.get_organization_id()

            # Create project via Neon API
            project_data = {
                "project": {
                    "name": project_name,
                    "region_id": region,
                    "org_id": org_id,  # Required field
                    "settings": {
                        "quota": {
                            "active_time_sec": 3600,  # 1 hour active time
                            "compute_time_sec": 3600,  # 1 hour compute time
                            "written_data_bytes": 1073741824,  # 1GB written data
                            "data_transfer_bytes": 10737418240,  # 10GB data transfer
                        }
                    },
                }
            }

            response = await self._make_request("POST", "/projects", data=project_data)

            project = response.get("project", {})
            if not project:
                raise NeonAPIError(
                    "Invalid response from Neon API - missing project data"
                )

            project_id = project.get("id")
            if not project_id:
                raise NeonAPIError("Project creation failed - no project ID returned")

            # Wait a moment for project to be initialized, then get connection details
            logger.info(f"Getting connection details for project {project_id}...")
            await asyncio.sleep(2)  # Brief wait for project initialization

            # Get connection details
            connection_info = await self._get_project_connection_string(project_id)

            project_info = NeonProjectInfo(
                project_id=project_id,
                name=project.get("name", project_name),
                database_url=connection_info["connection_string"],
                region=region,
                status=project.get("status", "active"),
                created_at=datetime.fromisoformat(
                    project.get("created_at", datetime.now().isoformat())
                ),
                default_branch_id=project.get("default_branch_id", ""),
                default_endpoint_id=project.get("default_endpoint_id", ""),
            )

            logger.info(
                f"Successfully created Neon project: {project_id} for tenant: {tenant_name}"
            )
            return project_info

        except Exception as e:
            logger.error(f"Failed to create Neon project for tenant {tenant_name}: {e}")
            raise

    async def _get_project_connection_string(self, project_id: str) -> Dict[str, str]:
        """
        Get connection string for a Neon project.

        Args:
            project_id: Neon project ID

        Returns:
            Dict with connection details
        """
        try:
            # Get project branches to find the default branch
            branches_response = await self._make_request(
                "GET", f"/projects/{project_id}/branches"
            )
            branches = branches_response.get("branches", [])

            # Find the default branch
            default_branch = None
            for branch in branches:
                if branch.get("default", False):
                    default_branch = branch
                    break

            if not default_branch:
                raise NeonAPIError(f"No default branch found for project {project_id}")

            default_branch_id = default_branch["id"]

            # Validate branch ID format (must match Neon's regex: ^[a-z0-9-]{1,60}$)
            import re

            if not re.match(r"^[a-z0-9-]{1,60}$", default_branch_id):
                raise NeonAPIError(f"Invalid branch ID format: {default_branch_id}")

            # Get connection details for the default branch
            conn_response = await self._make_request(
                "GET",
                f"/projects/{project_id}/connection_uri",
                params={
                    "branch_id": default_branch_id,
                    "role_name": "neondb_owner",
                    "database_name": "neondb",
                },
            )

            connection_string = conn_response.get("uri")
            if not connection_string:
                raise NeonAPIError("Failed to get connection string from Neon API")

            return {
                "connection_string": connection_string,
                "project_id": project_id,
                "branch_id": default_branch_id,
            }

        except Exception as e:
            logger.error(
                f"Failed to get connection string for project {project_id}: {e}"
            )
            raise

    async def get_project_info(self, project_id: str) -> Optional[NeonProjectInfo]:
        """
        Get information about a Neon project.

        Args:
            project_id: Neon project ID

        Returns:
            NeonProjectInfo if project exists, None otherwise
        """
        try:
            response = await self._make_request("GET", f"/projects/{project_id}")
            project = response.get("project", {})

            if not project:
                return None

            # Get connection string
            connection_info = await self._get_project_connection_string(project_id)

            return NeonProjectInfo(
                project_id=project_id,
                name=project.get("name", ""),
                database_url=connection_info["connection_string"],
                region=project.get("region_id", ""),
                status=project.get("status", "unknown"),
                created_at=datetime.fromisoformat(
                    project.get("created_at", datetime.now().isoformat())
                ),
                default_branch_id=project.get("default_branch_id", ""),
                default_endpoint_id=project.get("default_endpoint_id", ""),
            )

        except NeonAPIError as e:
            if e.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get project info for {project_id}: {e}")
            return None

    async def delete_tenant_project(self, project_id: str) -> bool:
        """
        Delete a tenant's Neon project (destroys all data).

        Args:
            project_id: Neon project ID to delete

        Returns:
            True if deletion successful, False otherwise

        Note:
            This permanently destroys all tenant data. Use with caution.
        """
        logger.warning(
            f"Deleting Neon project: {project_id} - This will destroy all data!"
        )

        try:
            await self._make_request("DELETE", f"/projects/{project_id}")
            logger.info(f"Successfully deleted Neon project: {project_id}")
            return True

        except NeonAPIError as e:
            if e.status_code == 404:
                logger.warning(
                    f"Project {project_id} not found - may already be deleted"
                )
                return True  # Consider this successful
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting project {project_id}: {e}")
            return False

    async def list_projects(self) -> List[NeonProjectInfo]:
        """
        List all Neon projects for this account.

        Returns:
            List of NeonProjectInfo objects
        """
        try:
            # Get organization ID for the request
            org_id = await self.get_organization_id()

            response = await self._make_request(
                "GET", "/projects", params={"org_id": org_id}
            )
            projects = response.get("projects", [])

            project_infos = []
            for project in projects:
                try:
                    connection_info = await self._get_project_connection_string(
                        project["id"]
                    )

                    project_info = NeonProjectInfo(
                        project_id=project["id"],
                        name=project.get("name", ""),
                        database_url=connection_info["connection_string"],
                        region=project.get("region_id", ""),
                        status=project.get("status", "unknown"),
                        created_at=datetime.fromisoformat(
                            project.get("created_at", datetime.now().isoformat())
                        ),
                        default_branch_id=project.get("default_branch_id", ""),
                        default_endpoint_id=project.get("default_endpoint_id", ""),
                    )
                    project_infos.append(project_info)

                except Exception as e:
                    logger.warning(
                        f"Failed to get connection info for project {project['id']}: {e}"
                    )
                    continue

            return project_infos

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    async def get_organization_id(self) -> str:
        """Get the organization ID for the current user"""
        # Use the confirmed organization ID from Neon Console
        org_id = "org-divine-leaf-04179575"
        logger.info(f"Using confirmed organization ID: {org_id}")
        return org_id

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None


# Global Neon project manager instance
_neon_manager: Optional[NeonProjectManager] = None


async def get_neon_project_manager() -> NeonProjectManager:
    """Get the global Neon project manager instance"""
    global _neon_manager

    if _neon_manager is None:
        api_key = os.getenv("NEON_API_KEY")
        if not api_key:
            raise ValueError("NEON_API_KEY environment variable not set")

        default_region = os.getenv("NEON_DEFAULT_REGION", "aws-us-east-1")
        _neon_manager = NeonProjectManager(api_key, default_region)

    return _neon_manager


async def close_neon_project_manager():
    """Close the global Neon project manager instance"""
    global _neon_manager

    if _neon_manager:
        await _neon_manager.close()
        _neon_manager = None
