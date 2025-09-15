# Multi-Tenant RAG System with Interactive CLI & FastAPI

## 🎯 **Overview**

A **production-ready, multi-tenant Retrieval-Augmented Generation (RAG) system** with complete data isolation, featuring both **FastAPI server** and **interactive CLI client**. Combines **Neon PostgreSQL + pgvector** for vector/hybrid search and **Neo4j + Graphiti** for knowledge graph capabilities.

Built following **official Neon and Graphiti best practices** with project-per-tenant database isolation and namespace-based graph isolation. Features a comprehensive **Pydantic AI agent** with intelligent tool routing and tenant-aware operations.

## ✨ **Key Features**

- 🔒 **Complete Tenant Isolation**: Project-per-tenant databases + namespace isolation
- 🚀 **Dual Interface**: FastAPI server + Interactive CLI client
- 🤖 **AI-Powered**: Enhanced Pydantic AI agent with 10+ tenant-aware tools
- 📊 **Multi-Modal Search**: Vector, Graph, Hybrid, and Comprehensive search
- 🔐 **JWT Authentication**: Secure tenant context with middleware validation
- 📈 **Production Ready**: Supervisor process management + Nginx reverse proxy
- 💰 **AWS EC2 Optimized**: Complete deployment guides for cloud infrastructure
- 🧪 **Fully Tested**: Comprehensive verification and testing scripts
- 🔄 **Live Integration**: Real-time Neon PostgreSQL + Neo4j Desktop connectivity

## 🏗️ **Architecture**

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Amazon EC2 / Local Deployment                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              FastAPI + Interactive CLI Stack               ││
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐ ││
│  │  │   FastAPI       │    │   Interactive CLI               │ ││
│  │  │   Multi-Tenant  │◄──►│   HTTP Client                   │ ││
│  │  │   API Server    │    │   (Rich Console)                │ ││
│  │  │   (Port 8000)   │    │                                 │ ││
│  │  └─────────────────┘    └─────────────────────────────────┘ ││
│  │  ┌─────────────────────────────────────────────────────────┐ ││
│  │  │         Enhanced Pydantic AI Agent                     │ ││
│  │  │  • Multi-Modal Search    • Tenant Context Injection   │ ││
│  │  │  • 10+ AI Tools         • Graph + Vector Integration  │ ││
│  │  │  • JWT Authentication   • Intelligent Tool Routing    │ ││
│  │  └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐    ┌──────────────────────────────┐
│       Neon PostgreSQL       │    │       Neo4j Desktop          │
│         (Cloud)             │    │     (Local/Remote)           │
│  • Project-per-Tenant      │    │  • Graphiti Integration      │
│  • Vector Search (pgvector) │    │  • Namespace Isolation       │
│  • Hybrid Search (BM25)     │    │  • Group ID Separation       │
│  • Catalog Database         │    │  • Knowledge Graph           │
│  • Complete Isolation       │    │  • Entity Relationships      │
└─────────────────────────────┘    └──────────────────────────────┘
```

## 📁 **File Structure**

```
Tenant/
├── README.md                           # This comprehensive guide
├── requirements.txt                    # Python dependencies
├── requirements_ec2_complete.txt       # Complete EC2 deployment dependencies
├── .env                               # Environment configuration (create from template)
│
# Core Application Files
├── interactive_multi_tenant_api.py     # FastAPI server with tenant isolation
├── interactive_multi_tenant_cli_http.py # Rich CLI client with HTTP communication
├── multi_tenant_agent.py              # Enhanced Pydantic AI agent
├── tenant_manager.py                  # Neon project management & tenant lifecycle
├── tenant_data_ingestion_service.py   # Multi-modal search services
├── tenant_graphiti_client.py          # Neo4j + Graphiti with namespacing
├── auth_middleware.py                 # JWT authentication & tenant context
│
# Deployment & Operations
├── start_api.sh                       # API server startup script
├── start_cli.sh                       # CLI client startup script
├── EC2_DEPLOYMENT_GUIDE.md            # Complete AWS EC2 deployment guide
├── EC2_DEPLOYMENT_CHECKLIST.md        # Pre/post deployment verification
├── ec2_quick_setup.sh                 # Automated EC2 setup script
├── verify_deployment.sh               # Deployment verification script
│
# Configuration & Schema
├── catalog_schema.sql                 # Catalog database schema for tenant metadata
└── deployment_guide.md                # Original deployment instructions
```

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone and navigate to Tenant directory
cd Tenant/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Environment Setup**

Create `.env` file:

```bash
# Neon PostgreSQL Configuration (Project-per-Tenant)
NEON_API_KEY=your_neon_api_key_here
CATALOG_DB_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require
POSTGRES_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require

# Neo4j Configuration (Local Desktop or Remote)
NEO4J_URI=neo4j://localhost:7687
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_AUTH=neo4j/your_neo4j_password

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256

# AI Providers
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional

# Application Configuration
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000
LOG_LEVEL=info
```

### **3. Database Setup**

```bash
# Setup catalog database (control plane) - if needed
psql "$CATALOG_DB_URL" -f catalog_schema.sql
```

### **4. Start the System**

```bash
# Method 1: Direct startup
python3 interactive_multi_tenant_api.py

# Method 2: Using startup scripts
chmod +x start_api.sh start_cli.sh
./start_api.sh

# In another terminal for CLI
./start_cli.sh
```

### **5. Verify Installation**

```bash
# Test API health
curl http://localhost:8000/health

# Run deployment verification
chmod +x verify_deployment.sh
./verify_deployment.sh
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📚 **Usage Examples**

### **🔐 Authentication & Setup**

```bash
# 1. Create a tenant
curl -X POST "http://localhost:8000/tenants" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acme Corporation", 
    "email": "admin@acme.com",
    "region": "aws-us-east-1",
    "plan": "basic"
  }'

# Response includes tenant_id for authentication
```

### **🗝️ Get Authentication Token**

```bash
# 2. Authenticate with tenant
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "TENANT_ID_FROM_ABOVE",
    "api_key": "your_api_key",
    "user_id": "john_doe"
  }'

# Save the JWT token from response
```

### **📄 Upload Document**

```bash
# 3. Upload document for ingestion
curl -X POST "http://localhost:8000/documents" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@document.pdf"

# Documents are automatically processed for vector and graph search
```

### **🔍 Multi-Modal Search**

```bash
# 4a. Vector Search (Semantic similarity)
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the company vacation policies?",
    "search_type": "vector",
    "limit": 10
  }'

# 4b. Graph Search (Knowledge relationships)
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "employee benefits structure",
    "search_type": "graph",
    "limit": 10
  }'

# 4c. Hybrid Search (Vector + BM25 text search)
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vacation policy details",
    "search_type": "hybrid",
    "limit": 10,
    "text_weight": 0.3
  }'

# 4d. Comprehensive Search (All methods combined)
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "comprehensive company policy information",
    "search_type": "comprehensive",
    "limit": 10
  }'
```

### **💬 Interactive Chat**

```bash
# 5. Chat with AI agent
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Can you explain our remote work policy and its benefits?",
    "session_id": "optional_session_id"
  }'
```

### **🖥️ Interactive CLI Usage**

```bash
# Start the interactive CLI
./start_cli.sh

# Or with custom API URL
./start_cli.sh --api-url http://your-server:8000

# CLI Features:
# 1. 🔐 Authenticate - Login with tenant credentials
# 2. ➕ Create New Tenant - Set up new tenant
# 3. ℹ️  Show Tenant Info - View tenant details
# 4. 📄 Upload Document - Upload files for processing
# 5. 🔍 Advanced Search (Technical) - Multi-modal search options
# 6. 💬 Chat Mode - Interactive AI conversation
# 7. 🏥 API Health Check - System status
# 8. 🚪 Exit - Close application
```

## 🔧 **Core Components**

### **TenantManager** (`tenant_manager.py`)
- **Neon Project Management**: Automated project creation and lifecycle management via Neon API
- **Database Routing**: Dynamic connection management to tenant-specific databases
- **Catalog Operations**: Centralized tenant metadata and configuration management
- **Isolation Guarantees**: Complete project-level separation ensuring zero cross-tenant access

### **TenantDataIngestionService** (`tenant_data_ingestion_service.py`)
- **Multi-Modal Search**: Vector, Graph, Hybrid, and Comprehensive search capabilities
- **Vector Search**: Semantic similarity using pgvector in tenant databases
- **Hybrid Search**: Combined vector + BM25 text search with configurable weights
- **Graph Integration**: Seamless integration with Graphiti for knowledge relationships

### **TenantGraphitiClient** (`tenant_graphiti_client.py`)
- **Namespace Isolation**: Complete tenant separation using group_id namespacing
- **Knowledge Graph**: Entity and relationship extraction with temporal tracking
- **Graph Analytics**: Advanced queries and relationship discovery within tenant boundaries
- **Episode Management**: Document ingestion with automatic knowledge graph updates

### **MultiTenantRAGAgent** (`multi_tenant_agent.py`)
- **Enhanced Pydantic AI**: 10+ specialized tools with tenant context injection
- **Intelligent Routing**: Automatic tool selection based on query complexity
- **Dual Storage**: Seamlessly combines vector and graph search results
- **Context Preservation**: Maintains tenant isolation across all agent operations

### **Authentication & Security** (`auth_middleware.py`)
- **JWT Authentication**: Secure token-based authentication with tenant claims
- **Context Injection**: Automatic tenant context validation and routing
- **Permission Management**: Fine-grained access control and audit logging
- **Cross-Tenant Prevention**: Multiple layers of isolation validation

### **Interactive CLI** (`interactive_multi_tenant_cli_http.py`)
- **Rich Console Interface**: Beautiful CLI with progress indicators and tables
- **HTTP Communication**: Seamless integration with FastAPI server
- **Multi-Modal Operations**: Full access to all search types and chat functionality
- **Session Management**: Persistent authentication and context handling

### **FastAPI Application** (`interactive_multi_tenant_api.py`)
- **Tenant-Aware Routing**: Automatic request routing to correct tenant resources
- **Comprehensive Endpoints**: Full REST API with authentication and documentation
- **Real-Time Processing**: Async operations with proper connection pooling
- **Production Ready**: Health checks, monitoring, and error handling

## 🔒 **Security Guarantees**

### **Database Level**

- ✅ Complete project-level isolation (one Neon project per tenant)
- ✅ No cross-tenant data access possible (physical separation)
- ✅ Independent scaling and performance per tenant
- ✅ Built-in backup and recovery per tenant

### **Graph Level**

- ✅ Namespace isolation using `group_id`
- ✅ Tenant-tagged entities and relationships
- ✅ Namespace-scoped search and analytics
- ✅ Prevent cross-tenant data access

### **Application Level**

- ✅ JWT tokens with tenant claims
- ✅ Middleware-level tenant validation
- ✅ Permission-based access control
- ✅ Comprehensive audit logging

## 📊 **API Endpoints**

### **Tenant Management**

- `POST /tenants` - Create new tenant with automatic Neon project setup
- `GET /tenants/info` - Get current tenant information and status
- `GET /health` - System health check and tenant count

### **Authentication**

- `POST /auth/login` - JWT authentication with tenant context
- Automatic token validation and tenant routing

### **Document Management**

- `POST /documents` - Upload files with automatic vector + graph ingestion
- Multi-format support (PDF, TXT, DOCX, etc.)

### **Multi-Modal Search**

- `POST /search` - Unified search endpoint with multiple types:
  - `search_type: "vector"` - Semantic similarity search
  - `search_type: "graph"` - Knowledge graph search
  - `search_type: "hybrid"` - Vector + BM25 text search
  - `search_type: "comprehensive"` - All methods combined

### **Interactive Chat**

- `POST /chat` - Conversational AI with context awareness
- Session management and source attribution
- Full integration with search capabilities

### **Monitoring & Analytics**

- `GET /health` - System health and database connectivity
- Comprehensive logging and audit trails
- Real-time performance metrics

## 🧪 **Testing & Verification**

### **Deployment Verification**

```bash
# Run comprehensive verification script
chmod +x verify_deployment.sh
./verify_deployment.sh

# This tests:
# - Python environment and dependencies
# - Database connectivity (Neon + Neo4j)
# - Application integrity
# - Security configuration
# - API functionality
```

### **Manual Testing**

```bash
# Test API health
curl http://localhost:8000/health

# Test authentication flow
curl -X POST "http://localhost:8000/tenants" -H "Content-Type: application/json" \
  -d '{"name": "Test Corp", "email": "test@corp.com"}'

# Test search functionality  
curl -X POST "http://localhost:8000/search" -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "search_type": "vector"}'
```

### **CLI Testing**

```bash
# Interactive CLI testing
./start_cli.sh --api-url http://localhost:8000

# Automated CLI testing
python3 interactive_multi_tenant_cli_http.py --help
```

## 🚀 **Deployment**

### **Local Development**

```bash
# Direct startup
python3 interactive_multi_tenant_api.py

# Using startup scripts
./start_api.sh
```

### **AWS EC2 Production Deployment**

**Complete deployment guides included:**

```bash
# 1. Upload to EC2 and run automated setup
scp -r Tenant/* ubuntu@your-ec2-ip:/tmp/Tenant/
ssh ubuntu@your-ec2-ip
cd /tmp/Tenant && ./ec2_quick_setup.sh

# 2. Configure environment
nano /opt/multi-tenant-rag/.env

# 3. Verify deployment
/opt/multi-tenant-rag/verify_deployment.sh

# 4. Start services
sudo supervisorctl start multi-tenant-rag-api
```

**See comprehensive deployment guides:**
- `EC2_DEPLOYMENT_GUIDE.md` - Complete step-by-step instructions
- `EC2_DEPLOYMENT_CHECKLIST.md` - Pre/post deployment verification
- `ec2_quick_setup.sh` - Automated setup script

## 📈 **Performance & Scaling**

### **Optimizations Included**
- Async database operations with connection pooling
- Efficient vector search with proper indexing
- Rate limiting and request throttling
- Caching strategies for frequently accessed data

### **Scaling Considerations**
- Horizontal scaling with load balancers
- Database read replicas for improved performance
- Redis caching for session management
- Kubernetes deployment for orchestration

## 🛠️ **Configuration Options**

### **Environment Variables**
- `NEON_CONNECTION_STRING` - PostgreSQL connection
- `NEO4J_URI` - Neo4j connection
- `JWT_SECRET_KEY` - JWT signing key
- `OPENAI_API_KEY` - AI model access
- `APP_ENV` - Environment (development/production)
- `LOG_LEVEL` - Logging verbosity

### **Tenant Limits**
- `max_documents` - Maximum documents per tenant
- `max_storage_mb` - Maximum storage per tenant  
- Custom quotas and rate limits per tenant

## 🔍 **Monitoring & Observability**

### **Built-in Monitoring**
- Health check endpoints
- Performance metrics collection
- Error tracking and alerting
- Audit logs for security events

### **Recommended Tools**
- **Prometheus** + **Grafana** for metrics
- **ELK Stack** for log aggregation
- **Sentry** for error tracking
- **DataDog** for APM

## 🤝 **Contributing**

1. **Follow the Architecture**: Maintain tenant isolation patterns
2. **Write Tests**: All new features require comprehensive tests
3. **Security First**: Validate tenant boundaries in all operations
4. **Documentation**: Update guides for any new features
5. **Performance**: Consider scalability in all implementations

## 📄 **License**

MIT License - see LICENSE file for details.

## 🆘 **Support & Troubleshooting**

### **Common Issues**
- **Database Connection**: Verify connection strings and SSL settings
- **Authentication Failures**: Check JWT secret keys and token expiration
- **Performance Issues**: Monitor connection pools and query performance

### **Debug Commands**
```bash
# Test database connectivity
python -c "import asyncpg; print('Database connection test')"

# Validate JWT tokens  
python -c "from jose import jwt; print('JWT validation test')"

# Check application logs
tail -f multi_tenant_rag.log
```

### **Getting Help**
- Check the `deployment_guide.md` for setup issues
- Review `testing_guide.md` for validation procedures
- Examine logs for detailed error information

## 🎯 **What Makes This System Special**

### **🔒 Complete Tenant Isolation**
- **Database Level**: Project-per-tenant with Neon PostgreSQL (zero cross-tenant data access)
- **Graph Level**: Namespace isolation using Graphiti group_id (tenant-tagged entities)
- **Application Level**: JWT context injection with middleware validation
- **Infrastructure Level**: Independent scaling and performance per tenant

### **🤖 Advanced AI Integration**
- **Enhanced Pydantic AI Agent**: 10+ specialized tools with intelligent routing
- **Multi-Modal Search**: Vector, Graph, Hybrid, and Comprehensive search modes
- **Real-Time Processing**: Async operations with proper connection pooling
- **Context Preservation**: Maintains tenant boundaries across all AI operations

### **🚀 Production-Ready Architecture**
- **Dual Interface**: FastAPI server + Rich interactive CLI
- **AWS EC2 Optimized**: Complete deployment automation and guides
- **Process Management**: Supervisor + Nginx for production reliability
- **Monitoring**: Health checks, logging, and verification scripts

### **📊 Proven Performance**
- **Scalable Design**: Linear scaling with tenant growth
- **Cost Efficient**: Scale-to-zero for inactive tenants
- **High Availability**: Multi-layer redundancy and error handling
- **Security First**: Multiple isolation layers and audit trails

### **🛠️ Developer Experience**
- **Comprehensive Documentation**: Step-by-step guides for all scenarios
- **Automated Setup**: One-command deployment with verification
- **Rich CLI Interface**: Beautiful console with progress indicators
- **Testing Suite**: Complete verification and validation tools

Perfect for **SaaS companies**, **enterprise applications**, and **multi-client AI systems** requiring complete data separation with shared infrastructure efficiency.

---

## 📋 **Quick Command Reference**

```bash
# 🚀 Get Started
./start_api.sh                    # Start FastAPI server
./start_cli.sh                    # Start interactive CLI
./verify_deployment.sh            # Verify installation

# 🔧 Development
python3 interactive_multi_tenant_api.py    # Direct API startup
curl http://localhost:8000/health          # Health check
curl http://localhost:8000/docs            # API documentation

# 📦 Production Deployment
./ec2_quick_setup.sh              # Automated EC2 setup
sudo supervisorctl status         # Check services
sudo tail -f /var/log/multi-tenant-rag-api.log  # View logs

# 🧪 Testing
curl -X POST "http://localhost:8000/tenants" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Corp", "email": "test@corp.com"}'
```

**Start building your secure, scalable, multi-tenant RAG system today!** 🚀
