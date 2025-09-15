"""
Onyx Cloud ingestion module for dual ingestion pipeline.
Extracts core functionality from standalone onyx_cloud_integration.py for use in unified pipeline.
"""

import os
import json
import time
import asyncio
import logging
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OnyxIngestionError(Exception):
    """Custom exception for Onyx ingestion errors."""
    pass


class OnyxCloudIngestor:
    """Simplified Onyx Cloud ingestor for dual ingestion pipeline."""
    
    def __init__(self, cc_pair_id: Optional[int] = None):
        """
        Initialize Onyx Cloud ingestor.
        
        Args:
            cc_pair_id: Use existing CC-pair ID (default: 285 - validated)
        """
        self.api_key = os.getenv("ONYX_API_KEY")
        self.base_url = "https://cloud.onyx.app"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Use validated CC-pair by default (from our successful standalone tests)
        self.cc_pair_id = cc_pair_id or 285
        self.connector_id = None  # Will be retrieved from CC-pair
        
        # Validate configuration
        if not self.api_key:
            raise OnyxIngestionError("ONYX_API_KEY not found in environment variables")
    
    async def initialize(self) -> bool:
        """
        Initialize ingestor by validating CC-pair and getting connector info.
        
        Returns:
            True if initialization successful
        """
        try:
            # Get CC-pair information to retrieve connector ID
            response = await asyncio.to_thread(
                requests.get,
                f"{self.base_url}/api/manage/admin/cc-pair/{self.cc_pair_id}",
                headers=self.headers
            )
            
            if response.status_code == 200:
                cc_pair_data = response.json()
                self.connector_id = cc_pair_data.get("connector", {}).get("id")
                
                if self.connector_id:
                    logger.info(f"✅ Onyx ingestor initialized with CC-pair {self.cc_pair_id}, connector {self.connector_id}")
                    return True
                else:
                    raise OnyxIngestionError(f"Could not get connector ID from CC-pair {self.cc_pair_id}")
            else:
                raise OnyxIngestionError(f"CC-pair {self.cc_pair_id} not accessible: {response.status_code}")
                
        except Exception as e:
            raise OnyxIngestionError(f"Failed to initialize Onyx ingestor: {e}")
    
    async def upload_document(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a single document to Onyx Cloud.
        
        Args:
            content: Document content
            filename: Document filename
            metadata: Optional metadata
            
        Returns:
            Upload result with document_id and status
        """
        if not self.connector_id:
            await self.initialize()
        
        try:
            # Prepare file content for upload
            file_content = content.encode('utf-8')
            
            # Determine content type
            if filename.endswith('.md'):
                content_type = 'text/markdown'
            elif filename.endswith('.txt'):
                content_type = 'text/plain'
            elif filename.endswith('.json'):
                content_type = 'application/json'
            else:
                content_type = 'text/plain'
            
            # Upload file
            files = {"files": (filename, file_content, content_type)}
            
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/manage/admin/connector/file/upload?connector_id={self.connector_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                file_uuid = result.get("file_paths", [""])[0]
                uploaded_filename = result.get("file_names", [filename])[0]
                
                logger.debug(f"📤 Uploaded to Onyx: {uploaded_filename} (UUID: {file_uuid})")
                
                return {
                    "success": True,
                    "document_id": file_uuid,
                    "filename": uploaded_filename,
                    "size": len(content),
                    "metadata": metadata or {}
                }
            else:
                raise OnyxIngestionError(f"Upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise OnyxIngestionError(f"Document upload failed: {e}")
    
    async def update_connector_with_file(self, file_uuid: str, filename: str) -> bool:
        """
        Update connector configuration to include the uploaded file.
        
        Args:
            file_uuid: UUID of uploaded file
            filename: Original filename
            
        Returns:
            True if update successful
        """
        try:
            # Get current connector configuration
            response = await asyncio.to_thread(
                requests.get,
                f"{self.base_url}/api/manage/admin/connector",
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise OnyxIngestionError(f"Failed to get connectors: {response.status_code}")
            
            # Find our connector
            connectors = response.json()
            target_connector = None
            
            for conn in connectors:
                if conn.get("id") == self.connector_id:
                    target_connector = conn
                    break
            
            if not target_connector:
                raise OnyxIngestionError(f"Connector {self.connector_id} not found")
            
            # Update file configuration
            current_config = target_connector.get("connector_specific_config", {})
            current_files = current_config.get("file_names", [])
            current_locations = current_config.get("file_locations", {})
            
            # Add new file if not already present
            if filename not in current_files:
                current_files.append(filename)
                current_locations[file_uuid] = filename
                
                update_payload = {
                    "name": target_connector["name"],
                    "source": target_connector["source"],
                    "input_type": target_connector["input_type"],
                    "access_type": target_connector.get("access_type", "public"),
                    "connector_specific_config": {
                        **current_config,
                        "file_names": current_files,
                        "file_locations": current_locations,
                    },
                    "refresh_freq": target_connector.get("refresh_freq"),
                    "prune_freq": target_connector.get("prune_freq"),
                    "disabled": target_connector.get("disabled", False),
                }
                
                # Update connector
                response = await asyncio.to_thread(
                    requests.patch,
                    f"{self.base_url}/api/manage/admin/connector/{self.connector_id}",
                    headers={**self.headers, "Content-Type": "application/json"},
                    data=json.dumps(update_payload)
                )
                
                if response.status_code == 200:
                    logger.debug(f"📝 Updated connector configuration with {filename}")
                    return True
                else:
                    raise OnyxIngestionError(f"Configuration update failed: {response.status_code} - {response.text}")
            else:
                logger.debug(f"📋 File {filename} already in connector configuration")
                return True
                
        except Exception as e:
            raise OnyxIngestionError(f"Connector update failed: {e}")
    
    async def trigger_indexing(self) -> bool:
        """
        Trigger indexing for the connector.
        
        Returns:
            True if indexing trigger successful
        """
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/api/manage/admin/connector/run-once",
                headers={**self.headers, "Content-Type": "application/json"},
                data=json.dumps({"connector_id": self.connector_id})
            )
            
            if response.status_code == 200:
                logger.debug(f"🚀 Triggered indexing for connector {self.connector_id}")
                return True
            else:
                logger.warning(f"⚠️ Indexing trigger returned {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to trigger indexing: {e}")
            return False


# Global ingestor instance for reuse
_global_ingestor: Optional[OnyxCloudIngestor] = None


async def get_onyx_ingestor(cc_pair_id: Optional[int] = None) -> OnyxCloudIngestor:
    """
    Get or create global Onyx ingestor instance.
    
    Args:
        cc_pair_id: CC-pair ID (default: 285 - validated)
        
    Returns:
        Initialized OnyxCloudIngestor instance
    """
    global _global_ingestor
    
    if _global_ingestor is None:
        _global_ingestor = OnyxCloudIngestor(cc_pair_id=cc_pair_id)
        await _global_ingestor.initialize()
    
    return _global_ingestor


async def ingest_to_onyx(
    content: str,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    cc_pair_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ingest a single document to Onyx Cloud as part of dual ingestion pipeline.
    
    Args:
        content: Document content
        file_path: Path to the document file
        metadata: Optional metadata dict
        cc_pair_id: CC-pair ID (default: 285 - validated)
        
    Returns:
        Ingestion result dict with success status and details
    """
    try:
        # Get filename from path
        filename = os.path.basename(file_path)
        
        # Get ingestor instance
        ingestor = await get_onyx_ingestor(cc_pair_id=cc_pair_id)
        
        # Upload document
        upload_result = await ingestor.upload_document(
            content=content,
            filename=filename,
            metadata=metadata
        )
        
        if upload_result["success"]:
            # Update connector configuration
            await ingestor.update_connector_with_file(
                file_uuid=upload_result["document_id"],
                filename=upload_result["filename"]
            )
            
            # Trigger indexing (non-blocking)
            await ingestor.trigger_indexing()
            
            return {
                "success": True,
                "document_id": upload_result["document_id"],
                "filename": upload_result["filename"],
                "sections_count": 1,  # Single document = 1 section
                "size": upload_result["size"],
                "cc_pair_id": ingestor.cc_pair_id,
                "connector_id": ingestor.connector_id
            }
        else:
            raise OnyxIngestionError("Upload failed without error details")
            
    except OnyxIngestionError:
        raise  # Re-raise Onyx-specific errors
    except Exception as e:
        raise OnyxIngestionError(f"Unexpected error during Onyx ingestion: {e}")


async def bulk_ingest_to_onyx(
    documents: List[Dict[str, Any]],
    cc_pair_id: Optional[int] = None,
    batch_delay: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Bulk ingest multiple documents to Onyx Cloud.
    
    Args:
        documents: List of dicts with 'content', 'file_path', and optional 'metadata'
        cc_pair_id: CC-pair ID (default: 285 - validated) 
        batch_delay: Delay between uploads in seconds
        
    Returns:
        List of ingestion results
    """
    results = []
    
    # Get ingestor instance once
    ingestor = await get_onyx_ingestor(cc_pair_id=cc_pair_id)
    
    for i, doc in enumerate(documents):
        try:
            result = await ingest_to_onyx(
                content=doc["content"],
                file_path=doc["file_path"],
                metadata=doc.get("metadata"),
                cc_pair_id=cc_pair_id
            )
            results.append(result)
            
            logger.info(f"📤 Onyx ingestion {i+1}/{len(documents)}: {result['filename']} ✅")
            
            # Add delay between uploads to avoid rate limiting
            if i < len(documents) - 1:
                await asyncio.sleep(batch_delay)
                
        except Exception as e:
            error_result = {
                "success": False,
                "document_id": None,
                "filename": os.path.basename(doc["file_path"]),
                "error": str(e)
            }
            results.append(error_result)
            logger.error(f"📤 Onyx ingestion {i+1}/{len(documents)}: {error_result['filename']} ❌ - {e}")
    
    return results


# For backwards compatibility with existing import
__all__ = ["ingest_to_onyx", "bulk_ingest_to_onyx", "OnyxIngestionError", "OnyxCloudIngestor"]
