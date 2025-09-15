#!/bin/bash

# 🚀 Start Multi-Tenant API Server Script
# Automatically activates virtual environment and starts the multi-tenant FastAPI server

set -e  # Exit on any error

echo "🚀 Starting Multi-Tenant RAG API Server..."

# Check if we're in the Tenant directory
if [ ! -f "interactive_multi_tenant_api.py" ]; then
    echo "❌ Error: interactive_multi_tenant_api.py not found"
    echo "Please run this script from the Tenant directory:"
    echo "cd /Users/rahul/Desktop/Graphiti/agentic-rag-knowledge-graph/Tenant"
    exit 1
fi

# Check if virtual environment exists (look in parent directory)
if [ ! -d "../.venv" ]; then
    echo "❌ Error: Virtual environment (.venv) not found in parent directory"
    echo "Please create a virtual environment first:"
    echo "cd .. && python -m venv .venv"
    echo "source ../.venv/bin/activate"
    echo "pip install -r requirements_final.txt"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ../.venv/bin/activate

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env file..."
    set -a  # automatically export all variables
    # Use a more robust way to load .env that handles special characters
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ "$line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Export the variable
        if [[ "$line" =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]]; then
            export "$line"
        fi
    done < .env
    set +a  # turn off automatic export
else
    echo "⚠️  Warning: .env file not found in Tenant directory"
fi

# Verify Python is using the virtual environment
PYTHON_PATH=$(which python)
echo "✅ Using Python: $PYTHON_PATH"

# Check if required dependencies are installed
echo "🔍 Checking dependencies..."
if ! python -c "import fastapi, uvicorn, pydantic_ai, jose" 2>/dev/null; then
    echo "⚠️  Some dependencies missing. Installing..."
    pip install fastapi uvicorn pydantic-ai python-jose[cryptography]
fi

# Check environment variables
echo "🔍 Checking environment variables..."
if [[ -z "$GOOGLE_API_KEY" && -z "$GEMINI_API_KEY" ]]; then
    echo "⚠️  Warning: No Google/Gemini API key found"
    echo "   Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment"
else
    echo "✅ Google/Gemini API key found"
fi

if [[ -z "$NEON_API_KEY" ]]; then
    echo "⚠️  Warning: NEON_API_KEY not set - using mock key"
    export NEON_API_KEY="neon_mock_api_key"
else
    echo "✅ Neon API key found"
fi

# Set CATALOG_DB_URL from available sources
if [[ -n "$DATABASE_URL" ]]; then
    export CATALOG_DB_URL="$DATABASE_URL"
    echo "✅ Database URL configured (using DATABASE_URL)"
elif [[ -n "$CATALOG_DATABASE_URL" ]]; then
    export CATALOG_DB_URL="$CATALOG_DATABASE_URL"
    echo "✅ Database URL configured (using CATALOG_DATABASE_URL)"
elif [[ -n "$POSTGRES_URL" ]]; then
    export CATALOG_DB_URL="$POSTGRES_URL"
    echo "✅ Database URL configured (using POSTGRES_URL)"
elif [[ -n "$CATALOG_DB_URL" ]]; then
    echo "✅ Database URL configured (using CATALOG_DB_URL)"
else
    echo "⚠️  Warning: No catalog database URL found"
    echo "   Set DATABASE_URL, CATALOG_DB_URL, or POSTGRES_URL for tenant data storage"
    echo "   Using default: postgresql://postgres:password@localhost:5432/catalog"
    export CATALOG_DB_URL="postgresql://postgres:password@localhost:5432/catalog"
fi

if [[ -n "$NEO4J_URI" && -n "$NEO4J_USER" && -n "$NEO4J_PASSWORD" ]]; then
    echo "✅ Neo4j configuration found"
    # Export Neo4j environment variables for the application
    export NEO4J_URL="$NEO4J_URI"
    export NEO4J_USER="$NEO4J_USER" 
    export NEO4J_PASSWORD="$NEO4J_PASSWORD"
else
    echo "⚠️  Warning: Neo4j configuration incomplete"
fi

# Start the multi-tenant API server
echo "🌟 Starting Multi-Tenant FastAPI server on port 8000..."
echo "📝 Logs will be displayed below. Press Ctrl+C to stop."
echo "🌐 API will be available at: http://localhost:8000"
echo "📋 Health check: http://localhost:8000/health"
echo "📖 API docs: http://localhost:8000/docs"
echo "🔐 Authentication: JWT tokens required for most endpoints"
echo ""

python interactive_multi_tenant_api.py
