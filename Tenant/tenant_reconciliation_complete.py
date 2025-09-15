#!/usr/bin/env python3
"""
Complete Tenant Reconciliation Service
======================================

Handles catalog-project drift when tenants are deleted outside the API.
This is the ONLY script you need for reconciliation.

Usage:
------
# Check for drift (analysis only)
python3 tenant_reconciliation_complete.py --mode=analysis

# Simulate fixes (dry-run)
python3 tenant_reconciliation_complete.py --mode=dry-run

# Apply fixes (reconcile)
python3 tenant_reconciliation_complete.py --mode=reconcile

Example for your scenario:
-------------------------
# You deleted projects via Neon UI, catalog still has records
python3 tenant_reconciliation_complete.py --mode=reconcile

This will automatically remove orphaned catalog entries for deleted projects.
"""

import os
import sys
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
import asyncpg
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tenant_reconciliation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TenantStatus(Enum):
    """Tenant status enum"""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class DriftType(Enum):
    """Types of catalog-project drift"""

    ORPHANED_CATALOG_ENTRY = "orphaned_catalog_entry"  # In catalog, not in Neon
    ORPHANED_NEON_PROJECT = "orphaned_neon_project"  # In Neon, not in catalog
    STATUS_MISMATCH = "status_mismatch"  # Different status between catalog and Neon


@dataclass
class DriftIssue:
    """Represents a drift issue between catalog and Neon projects"""

    drift_type: DriftType
    tenant_id: Optional[UUID]
    tenant_name: Optional[str]
    neon_project_id: Optional[str]
    neon_project_name: Optional[str]
    catalog_status: Optional[str]
    neon_status: Optional[str]
    description: str
    severity: str  # "critical", "warning", "info"
    recommended_action: str


@dataclass
class ReconciliationReport:
    """Report of reconciliation analysis and actions"""

    timestamp: datetime
    total_catalog_tenants: int
    total_neon_projects: int
    drift_issues: List[DriftIssue]
    actions_taken: List[str]
    summary: Dict[str, int]


@dataclass
class NeonProject:
    """Neon project information"""

    project_id: str
    name: str
    region_id: str
    created_at: datetime
    owner_id: str


# =============================================================================
# Neon API Client (Simplified)
# =============================================================================


class NeonAPIClient:
    """Simplified Neon API client for reconciliation"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://console.neon.tech/api/v2"

    async def list_projects(self) -> List[NeonProject]:
        """List all projects in the organization"""
        import aiohttp
        import ssl

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Create SSL context that doesn't verify certificates (for macOS compatibility)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)

        async with aiohttp.ClientSession(connector=connector) as session:
            # Add org_id parameter which is required by Neon API
            org_id = "org-divine-leaf-04179575"  # Use the confirmed organization ID
            params = {"org_id": org_id}

            async with session.get(
                f"{self.base_url}/projects", headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    projects = []
                    for project_data in data.get("projects", []):
                        projects.append(
                            NeonProject(
                                project_id=project_data["id"],
                                name=project_data["name"],
                                region_id=project_data["region_id"],
                                created_at=datetime.fromisoformat(
                                    project_data["created_at"].replace("Z", "+00:00")
                                ),
                                owner_id=project_data["owner_id"],
                            )
                        )
                    return projects
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to list projects: {response.status} - {error_text}"
                    )


# =============================================================================
# Catalog Database Client (Simplified)
# =============================================================================


class CatalogClient:
    """Simplified catalog database client"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def list_tenant_projects(self) -> List[Dict]:
        """List all tenant projects from catalog"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            rows = await conn.fetch("""
                SELECT 
                    tenant_id,
                    tenant_name,
                    tenant_email,
                    neon_project_id,
                    status,
                    created_at
                FROM tenant_projects 
                WHERE status != 'deleted'
                ORDER BY created_at DESC
            """)
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def delete_tenant_project(self, tenant_id: UUID) -> None:
        """Delete tenant project from catalog"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            await conn.execute(
                """
                DELETE FROM tenant_projects WHERE tenant_id = $1
            """,
                tenant_id,
            )
        finally:
            await conn.close()

    async def update_tenant_status(self, tenant_id: UUID, status: str) -> None:
        """Update tenant status in catalog"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            await conn.execute(
                """
                UPDATE tenant_projects 
                SET status = $2, updated_at = NOW() 
                WHERE tenant_id = $1
            """,
                tenant_id,
                status,
            )
        finally:
            await conn.close()


# =============================================================================
# Reconciliation Service
# =============================================================================


class TenantReconciliationService:
    """Complete tenant reconciliation service"""

    def __init__(self, neon_api_key: str, catalog_conn_string: str):
        self.neon_client = NeonAPIClient(neon_api_key)
        self.catalog_client = CatalogClient(catalog_conn_string)

    async def analyze_drift(self) -> ReconciliationReport:
        """Analyze drift between catalog and Neon projects"""
        logger.info("🔍 Starting tenant drift analysis...")
        timestamp = datetime.now()
        drift_issues = []

        try:
            # Get all tenants from catalog
            catalog_tenants = await self.catalog_client.list_tenant_projects()
            catalog_project_ids = {
                t["neon_project_id"] for t in catalog_tenants if t["neon_project_id"]
            }

            # Get all projects from Neon (filter tenant projects)
            neon_projects = await self.neon_client.list_projects()
            # Filter to only tenant projects (exclude catalog project)
            tenant_projects = [
                p
                for p in neon_projects
                if p.name.startswith("tenant-") or p.project_id in catalog_project_ids
            ]
            neon_project_ids = {p.project_id for p in tenant_projects}

            logger.info(
                f"📊 Found {len(catalog_tenants)} catalog tenants, {len(tenant_projects)} Neon tenant projects"
            )

            # Check for orphaned catalog entries (CRITICAL - your scenario)
            for tenant in catalog_tenants:
                if (
                    tenant["neon_project_id"]
                    and tenant["neon_project_id"] not in neon_project_ids
                ):
                    drift_issues.append(
                        DriftIssue(
                            drift_type=DriftType.ORPHANED_CATALOG_ENTRY,
                            tenant_id=tenant["tenant_id"],
                            tenant_name=tenant["tenant_name"],
                            neon_project_id=tenant["neon_project_id"],
                            neon_project_name=None,
                            catalog_status=tenant["status"],
                            neon_status=None,
                            description=f"Tenant '{tenant['tenant_name']}' exists in catalog but Neon project '{tenant['neon_project_id']}' not found",
                            severity="critical",
                            recommended_action="Remove orphaned catalog entry (project was deleted via Neon UI)",
                        )
                    )

            # Check for orphaned Neon projects
            catalog_projects_map = {
                t["neon_project_id"]: t for t in catalog_tenants if t["neon_project_id"]
            }
            for project in tenant_projects:
                if project.project_id not in catalog_projects_map:
                    drift_issues.append(
                        DriftIssue(
                            drift_type=DriftType.ORPHANED_NEON_PROJECT,
                            tenant_id=None,
                            tenant_name=None,
                            neon_project_id=project.project_id,
                            neon_project_name=project.name,
                            catalog_status=None,
                            neon_status="active",
                            description=f"Neon project '{project.name}' ({project.project_id}) exists but not found in catalog",
                            severity="warning",
                            recommended_action="Add to catalog or delete Neon project if no longer needed",
                        )
                    )

            # Check for status mismatches
            for tenant in catalog_tenants:
                if tenant["neon_project_id"] in neon_project_ids:
                    neon_project = next(
                        p
                        for p in tenant_projects
                        if p.project_id == tenant["neon_project_id"]
                    )
                    # Check if catalog shows deleted/suspended but Neon project still active
                    if tenant["status"] in ["deleted", "suspended"] and neon_project:
                        drift_issues.append(
                            DriftIssue(
                                drift_type=DriftType.STATUS_MISMATCH,
                                tenant_id=tenant["tenant_id"],
                                tenant_name=tenant["tenant_name"],
                                neon_project_id=tenant["neon_project_id"],
                                neon_project_name=neon_project.name,
                                catalog_status=tenant["status"],
                                neon_status="active",
                                description=f"Tenant '{tenant['tenant_name']}' marked as {tenant['status']} in catalog but Neon project is still active",
                                severity="warning",
                                recommended_action="Update catalog status to match Neon reality",
                            )
                        )

            # Generate summary
            summary = {
                "orphaned_catalog_entries": len(
                    [
                        d
                        for d in drift_issues
                        if d.drift_type == DriftType.ORPHANED_CATALOG_ENTRY
                    ]
                ),
                "orphaned_neon_projects": len(
                    [
                        d
                        for d in drift_issues
                        if d.drift_type == DriftType.ORPHANED_NEON_PROJECT
                    ]
                ),
                "status_mismatches": len(
                    [
                        d
                        for d in drift_issues
                        if d.drift_type == DriftType.STATUS_MISMATCH
                    ]
                ),
                "critical_issues": len(
                    [d for d in drift_issues if d.severity == "critical"]
                ),
                "warning_issues": len(
                    [d for d in drift_issues if d.severity == "warning"]
                ),
            }

            report = ReconciliationReport(
                timestamp=timestamp,
                total_catalog_tenants=len(catalog_tenants),
                total_neon_projects=len(tenant_projects),
                drift_issues=drift_issues,
                actions_taken=[],
                summary=summary,
            )

            logger.info(f"✅ Drift analysis complete: {len(drift_issues)} issues found")
            return report

        except Exception as e:
            logger.error(f"❌ Drift analysis failed: {str(e)}")
            raise

    async def fix_orphaned_catalog_entries(self, dry_run: bool = True) -> List[str]:
        """Fix orphaned catalog entries by removing them"""
        report = await self.analyze_drift()
        orphaned_entries = [
            d
            for d in report.drift_issues
            if d.drift_type == DriftType.ORPHANED_CATALOG_ENTRY
        ]

        actions = []

        for issue in orphaned_entries:
            action = f"{'[DRY RUN] ' if dry_run else ''}Remove orphaned catalog entry for tenant '{issue.tenant_name}' (project: {issue.neon_project_id})"
            actions.append(action)

            if not dry_run:
                try:
                    await self.catalog_client.delete_tenant_project(issue.tenant_id)
                    logger.info(
                        f"✅ Removed orphaned catalog entry: {issue.tenant_name}"
                    )
                except Exception as e:
                    error_action = f"❌ Failed to remove catalog entry for {issue.tenant_name}: {str(e)}"
                    actions.append(error_action)
                    logger.error(error_action)

        return actions

    async def fix_status_mismatches(self, dry_run: bool = True) -> List[str]:
        """Fix status mismatches by updating catalog to match Neon reality"""
        report = await self.analyze_drift()
        mismatches = [
            d for d in report.drift_issues if d.drift_type == DriftType.STATUS_MISMATCH
        ]

        actions = []

        for issue in mismatches:
            action = f"{'[DRY RUN] ' if dry_run else ''}Update tenant '{issue.tenant_name}' status from '{issue.catalog_status}' to 'active'"
            actions.append(action)

            if not dry_run:
                try:
                    await self.catalog_client.update_tenant_status(
                        issue.tenant_id, TenantStatus.ACTIVE.value
                    )
                    logger.info(f"✅ Updated tenant status: {issue.tenant_name}")
                except Exception as e:
                    error_action = (
                        f"❌ Failed to update status for {issue.tenant_name}: {str(e)}"
                    )
                    actions.append(error_action)
                    logger.error(error_action)

        return actions

    async def full_reconciliation(
        self,
        fix_orphaned_entries: bool = True,
        fix_status_mismatches: bool = True,
        dry_run: bool = True,
    ) -> ReconciliationReport:
        """Perform full reconciliation with optional fixes"""
        logger.info(f"🔧 Starting full reconciliation (dry_run={dry_run})...")

        # First, analyze current state
        report = await self.analyze_drift()

        all_actions = []

        # Fix orphaned catalog entries (your main issue)
        if fix_orphaned_entries:
            orphaned_actions = await self.fix_orphaned_catalog_entries(dry_run)
            all_actions.extend(orphaned_actions)

        # Fix status mismatches
        if fix_status_mismatches:
            status_actions = await self.fix_status_mismatches(dry_run)
            all_actions.extend(status_actions)

        # Update report with actions taken
        report.actions_taken = all_actions

        logger.info(
            f"✅ Reconciliation complete: {len(all_actions)} actions {'would be taken' if dry_run else 'taken'}"
        )

        return report

    def print_report(self, report: ReconciliationReport):
        """Print a human-readable reconciliation report"""
        print("\n🔍 TENANT RECONCILIATION REPORT")
        print(f"📅 Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📊 Catalog Tenants: {report.total_catalog_tenants}")
        print(f"📊 Neon Projects: {report.total_neon_projects}")
        print("=" * 60)

        print("\n📈 SUMMARY:")
        for key, value in report.summary.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")

        if report.drift_issues:
            print("\n⚠️  DRIFT ISSUES DETECTED:")

            # Group by type
            by_type = {}
            for issue in report.drift_issues:
                if issue.drift_type not in by_type:
                    by_type[issue.drift_type] = []
                by_type[issue.drift_type].append(issue)

            for drift_type, issues in by_type.items():
                print(
                    f"\n  {drift_type.value.replace('_', ' ').title()} ({len(issues)} issues):"
                )
                for issue in issues:
                    severity_icon = "🔴" if issue.severity == "critical" else "🟡"
                    print(f"    {severity_icon} {issue.description}")
                    print(f"       → Recommendation: {issue.recommended_action}")
        else:
            print("\n✅ NO DRIFT ISSUES DETECTED - Catalog and Neon are in sync!")

        if report.actions_taken:
            print("\n🔧 ACTIONS TAKEN:")
            for action in report.actions_taken:
                print(f"  • {action}")

        print("=" * 60)


# =============================================================================
# Main Script Logic
# =============================================================================


async def run_reconciliation(mode: str = "analysis"):
    """Run tenant reconciliation based on mode"""

    # Load environment variables
    load_dotenv()
    load_dotenv("../.env")

    api_key = os.getenv("NEON_API_KEY")
    catalog_conn = os.getenv("CATALOG_DATABASE_URL")

    if not api_key or not catalog_conn:
        logger.error("❌ Missing NEON_API_KEY or CATALOG_DATABASE_URL")
        return False

    logger.info(f"🔧 Starting reconciliation (mode: {mode})")

    try:
        reconciler = TenantReconciliationService(api_key, catalog_conn)

        if mode == "analysis":
            # Just analyze and report
            report = await reconciler.analyze_drift()

            # Log summary
            if report.drift_issues:
                logger.warning(f"⚠️  Found {len(report.drift_issues)} drift issues")
                for drift_type, count in report.summary.items():
                    if count > 0:
                        logger.warning(
                            f"  • {drift_type.replace('_', ' ').title()}: {count}"
                        )
            else:
                logger.info("✅ No drift issues detected")

            # Print detailed report
            reconciler.print_report(report)

        elif mode == "dry-run":
            # Simulate fixes without applying them
            logger.info("🧪 Running dry-run reconciliation...")
            report = await reconciler.full_reconciliation(
                fix_orphaned_entries=True, fix_status_mismatches=True, dry_run=True
            )

            if report.actions_taken:
                logger.info(f"📋 {len(report.actions_taken)} actions would be taken:")
                for action in report.actions_taken:
                    logger.info(f"  • {action}")
            else:
                logger.info("✅ No actions needed")

            reconciler.print_report(report)

        elif mode == "reconcile":
            # Apply fixes (YOUR SCENARIO)
            logger.info("🔧 Running full reconciliation with fixes...")
            report = await reconciler.full_reconciliation(
                fix_orphaned_entries=True, fix_status_mismatches=True, dry_run=False
            )

            if report.actions_taken:
                logger.info(f"✅ {len(report.actions_taken)} actions completed:")
                for action in report.actions_taken:
                    logger.info(f"  • {action}")
            else:
                logger.info("✅ No actions needed")

            reconciler.print_report(report)

        else:
            logger.error(f"❌ Unknown mode: {mode}")
            return False

        logger.info(f"✅ Reconciliation completed successfully (mode: {mode})")
        return True

    except Exception as e:
        logger.error(f"❌ Reconciliation failed: {str(e)}")
        return False


def main():
    """Main entry point with argument parsing"""

    parser = argparse.ArgumentParser(
        description="Complete tenant reconciliation service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["analysis", "dry-run", "reconcile"],
        default="analysis",
        help="Reconciliation mode (default: analysis)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Log startup
    logger.info(
        f"🚀 Starting tenant reconciliation at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )

    # Run reconciliation
    success = asyncio.run(run_reconciliation(args.mode))

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
