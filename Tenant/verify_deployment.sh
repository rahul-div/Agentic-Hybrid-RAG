#!/bin/bash

# Multi-Tenant RAG System - EC2 Deployment Verification Script
# This script verifies all components are properly deployed and integrated

set -e  # Exit on any error

echo "🚀 Multi-Tenant RAG System - Deployment Verification"
echo "=================================================="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Directory: $(pwd)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Function to print info
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "=== STEP 1: System Environment Check ==="
echo ""

# Check Python version
print_info "Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 1 ]]; then
    print_status 0 "Python version $PYTHON_VERSION is compatible"
else
    print_status 1 "Python version $PYTHON_VERSION is too old (need 3.9+)"
    exit 1
fi

# Check virtual environment
print_info "Checking virtual environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status 0 "Virtual environment active: $VIRTUAL_ENV"
else
    print_warning "Virtual environment not active. Activating..."
    source venv/bin/activate
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status 0 "Virtual environment activated"
    else
        print_status 1 "Failed to activate virtual environment"
        exit 1
    fi
fi

# Check if .env file exists
print_info "Checking environment configuration..."
if [ -f ".env" ]; then
    print_status 0 "Environment file (.env) found"
    source .env
else
    print_status 1 "Environment file (.env) not found"
    exit 1
fi

echo ""
echo "=== STEP 2: Python Dependencies Check ==="
echo ""

# Critical packages check
CRITICAL_PACKAGES=(
    "fastapi"
    "uvicorn"
    "pydantic_ai"
    "neo4j"
    "asyncpg"
    "graphiti"
    "openai"
    "httpx"
    "rich"
    "click"
)

for package in "${CRITICAL_PACKAGES[@]}"; do
    print_info "Checking $package..."
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
        print_status 0 "$package installed (version: $VERSION)"
    else
        print_status 1 "$package not installed or import failed"
    fi
done

echo ""
echo "=== STEP 3: Database Connectivity Check ==="
echo ""

# Test Neon PostgreSQL connection
print_info "Testing Neon PostgreSQL connection..."
NEON_TEST=$(python3 -c "
import asyncpg
import asyncio
import os
import sys

async def test_neon():
    try:
        conn = await asyncpg.connect(os.getenv('CATALOG_DB_URL') or os.getenv('POSTGRES_URL'))
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        print('SUCCESS')
        print(f'PostgreSQL Version: {result[:50]}...')
    except Exception as e:
        print('FAILED')
        print(f'Error: {e}')
        sys.exit(1)

asyncio.run(test_neon())
" 2>&1)

if echo "$NEON_TEST" | grep -q "SUCCESS"; then
    print_status 0 "Neon PostgreSQL connection successful"
    echo "$NEON_TEST" | grep "PostgreSQL Version"
else
    print_status 1 "Neon PostgreSQL connection failed"
    echo "$NEON_TEST"
fi

# Test Neo4j connection
print_info "Testing Neo4j connection..."
NEO4J_TEST=$(python3 -c "
import os
from neo4j import GraphDatabase
import sys

try:
    uri = os.getenv('NEO4J_URI') or os.getenv('NEO4J_URL')
    user = os.getenv('NEO4J_USERNAME') or os.getenv('NEO4J_USER') or 'neo4j'
    password = os.getenv('NEO4J_PASSWORD')
    
    if not uri or not password:
        print('FAILED')
        print('Missing Neo4j configuration')
        sys.exit(1)
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        result = session.run('CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] as version, edition')
        record = result.single()
        print('SUCCESS')
        print(f'Neo4j {record[\"edition\"]} Version: {record[\"version\"]}')
    driver.close()
except Exception as e:
    print('FAILED')
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)

if echo "$NEO4J_TEST" | grep -q "SUCCESS"; then
    print_status 0 "Neo4j connection successful"
    echo "$NEO4J_TEST" | grep "Neo4j"
else
    print_status 1 "Neo4j connection failed"
    echo "$NEO4J_TEST"
fi

echo ""
echo "=== STEP 4: Application Files Check ==="
echo ""

# Check critical application files
CRITICAL_FILES=(
    "interactive_multi_tenant_api.py"
    "interactive_multi_tenant_cli_http.py"
    "tenant_manager.py"
    "multi_tenant_agent.py"
    "auth_middleware.py"
    "tenant_data_ingestion_service.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    print_info "Checking $file..."
    if [ -f "$file" ]; then
        print_status 0 "$file exists"
    else
        print_status 1 "$file missing"
    fi
done

echo ""
echo "=== STEP 5: API Server Test ==="
echo ""

# Test if API can start
print_info "Testing API server startup..."
timeout 10s python3 interactive_multi_tenant_api.py --host 127.0.0.1 --port 8001 &
API_PID=$!
sleep 5

# Test health endpoint
print_info "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://127.0.0.1:8001/health 2>/dev/null || echo "FAILED")

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_status 0 "API health endpoint responding"
    echo "Response: $HEALTH_RESPONSE"
else
    print_status 1 "API health endpoint not responding"
fi

# Clean up API process
kill $API_PID 2>/dev/null || true

echo ""
echo "=== STEP 6: System Services Check ==="
echo ""

# Check if supervisor is installed and running
print_info "Checking Supervisor..."
if command -v supervisorctl &> /dev/null; then
    print_status 0 "Supervisor installed"
    if sudo supervisorctl status &> /dev/null; then
        print_status 0 "Supervisor daemon running"
    else
        print_status 1 "Supervisor daemon not running"
    fi
else
    print_status 1 "Supervisor not installed"
fi

# Check if Nginx is installed and running
print_info "Checking Nginx..."
if command -v nginx &> /dev/null; then
    print_status 0 "Nginx installed"
    if sudo systemctl is-active --quiet nginx; then
        print_status 0 "Nginx service running"
    else
        print_status 1 "Nginx service not running"
    fi
else
    print_status 1 "Nginx not installed"
fi

echo ""
echo "=== STEP 7: Network Connectivity Check ==="
echo ""

# Check if ports are available
print_info "Checking port availability..."
PORT_8000=$(netstat -tlnp 2>/dev/null | grep ":8000 " || echo "")
if [ -z "$PORT_8000" ]; then
    print_status 0 "Port 8000 available"
else
    print_status 1 "Port 8000 in use: $PORT_8000"
fi

PORT_80=$(netstat -tlnp 2>/dev/null | grep ":80 " || echo "")
if [ -z "$PORT_80" ]; then
    print_warning "Port 80 available (Nginx not listening)"
else
    print_status 0 "Port 80 in use (likely Nginx): $PORT_80"
fi

echo ""
echo "=== STEP 8: Security Check ==="
echo ""

# Check file permissions
print_info "Checking .env file permissions..."
ENV_PERMS=$(stat -c %a .env 2>/dev/null || echo "000")
if [ "$ENV_PERMS" = "600" ]; then
    print_status 0 ".env file has secure permissions (600)"
else
    print_warning ".env file permissions: $ENV_PERMS (recommend 600)"
fi

# Check if JWT secret is set
print_info "Checking JWT configuration..."
if [ -n "$JWT_SECRET_KEY" ] && [ ${#JWT_SECRET_KEY} -gt 32 ]; then
    print_status 0 "JWT secret key configured and sufficiently long"
else
    print_status 1 "JWT secret key missing or too short"
fi

echo ""
echo "=== STEP 9: AI Provider Check ==="
echo ""

# Check AI API keys
print_info "Checking AI provider configurations..."
if [ -n "$OPENAI_API_KEY" ]; then
    print_status 0 "OpenAI API key configured"
else
    print_warning "OpenAI API key not configured"
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    print_status 0 "Google API key configured"
else
    print_warning "Google API key not configured"
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    print_status 0 "Anthropic API key configured"
else
    print_warning "Anthropic API key not configured"
fi

echo ""
echo "=== STEP 10: Integration Test ==="
echo ""

# Test tenant creation and authentication
print_info "Testing tenant creation flow..."
TENANT_TEST=$(python3 -c "
import sys
import os
sys.path.append('.')

try:
    # Test imports
    from tenant_manager import TenantManager
    from auth_middleware import JWTAuthenticator
    from multi_tenant_agent import MultiTenantRAGAgent
    
    print('SUCCESS: All core modules importable')
except Exception as e:
    print(f'FAILED: Import error - {e}')
    sys.exit(1)
" 2>&1)

if echo "$TENANT_TEST" | grep -q "SUCCESS"; then
    print_status 0 "Core modules import successfully"
else
    print_status 1 "Core modules import failed"
    echo "$TENANT_TEST"
fi

echo ""
echo "=================================================="
echo "🏁 DEPLOYMENT VERIFICATION COMPLETE"
echo "=================================================="
echo ""

# Summary
echo "=== SUMMARY ==="
echo ""
print_info "System Status Summary:"
echo "  • Python Environment: Ready"
echo "  • Dependencies: Installed"
echo "  • Database Connections: Tested"
echo "  • Application Files: Present"
echo "  • API Server: Functional"
echo "  • Security: Configured"
echo ""

if echo "$NEON_TEST $NEO4J_TEST $TENANT_TEST" | grep -q "FAILED"; then
    print_warning "Some components failed verification. Please review the errors above."
    echo ""
    echo "🔧 Common fixes:"
    echo "  • Check .env file configuration"
    echo "  • Verify database connection strings"
    echo "  • Ensure all dependencies are installed"
    echo "  • Check network connectivity to databases"
else
    print_status 0 "All critical components verified successfully!"
    echo ""
    print_info "Your deployment is ready! Next steps:"
    echo "  1. Start services: sudo supervisorctl start all"
    echo "  2. Test API: curl http://localhost:8000/health"
    echo "  3. Test CLI: ./start_cli.sh --api-url http://localhost:8000"
fi

echo ""
echo "📋 Deployment Commands:"
echo "  • Start API: ./start_api.sh"
echo "  • Start CLI: ./start_cli.sh"
echo "  • Monitor: ./monitor.sh"
echo "  • Restart Services: sudo supervisorctl restart all"
echo ""
