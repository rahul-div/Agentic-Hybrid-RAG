# Ingestion Options Feature

## Overview
The multi-tenant CLI now supports selective ingestion options, allowing users to choose between:
1. **Vector Database only** - Fast ingestion for search capabilities
2. **Knowledge Graph only** - Slower ingestion for relationship analysis
3. **Both** - Complete ingestion (recommended)

## Changes Made

### 1. Data Models (`tenant_ingestion_models.py`)
- Added `ingest_vector: bool` and `ingest_graph: bool` fields to `DocumentInput`

### 2. CLI Interface (`interactive_multi_tenant_cli_http.py`)
- Enhanced upload menu with ingestion options
- Added time estimates for each option
- Improved error handling for timeouts

### 3. API Endpoint (`interactive_multi_tenant_api.py`)  
- Updated `/documents` endpoint to accept ingestion preferences
- Added validation for option selection

### 4. Core Logic (`tenant_data_ingestion_service.py`)
- Modified ingestion workflow to conditionally process based on options
- Enhanced logging for transparency

## Usage

When uploading a document via CLI (option 4):

```
🛠️ Ingestion Options
Choose what to ingest into:
  1. 🗃️ Vector Database only (typically 10-30 seconds)
  2. 🕸️ Knowledge Graph only (may take 1-2 minutes)  
  3. 🚀 Both Vector DB + Knowledge Graph (may take 1-3 minutes)
```

## Performance Impact

- **Vector only**: ~10-30 seconds (embeddings + vector storage)
- **Graph only**: ~1-2 minutes (entity extraction + relationship building)
- **Both**: ~1-3 minutes (complete processing)

## Technical Details

The implementation ensures:
- Document metadata is always stored regardless of options
- Vector ingestion creates chunks with embeddings for search
- Graph ingestion creates episodes for relationship analysis  
- Proper error handling and rollback on failures
- Backward compatibility with existing code

## Benefits

1. **Faster ingestion** when only search is needed
2. **Reduced resource usage** for simple use cases
3. **User choice** based on their specific requirements
4. **Better user experience** with clear options and time estimates
