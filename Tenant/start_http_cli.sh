#!/bin/bash

# Start HTTP-based Multi-Tenant CLI Client
# This version uses HTTP API endpoints with JWT authentication

cd "$(dirname "$0")"

export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

echo "🚀 Starting HTTP-based Multi-Tenant CLI..."
echo "📡 Connecting to API server at: ${API_URL:-http://localhost:8000}"
echo ""

python3 interactive_multi_tenant_cli_http.py "$@"
