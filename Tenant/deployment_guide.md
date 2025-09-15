# Multi-Tenant RAG System Deployment Guide

## 🚀 **Overview**

This guide provides step-by-step instructions for deploying the multi-tenant RAG system using **Neon PostgreSQL (Project-per-Tenant)** and **Neo4j + Graphiti (group_id namespacing)** following official best practices for production-ready multi-tenancy.

## 📋 **Prerequisites**

### **System Requirements**
- Python 3.9+ 
- Docker (optional for containerized deployment)
- 4GB+ RAM for development, 8GB+ for production
- 20GB+ storage for databases

### **External Services**
- **Neon PostgreSQL**: Serverless PostgreSQL with pgvector (one project per tenant)
- **Neon API Key**: For automated project creation and management
- **Neo4j**: Graph database (single shared instance with namespacing)
- **OpenAI API Key**: For embeddings and LLM (or alternative providers)

## 🏗️ **CORRECTED Architecture Components**

```
┌─────────────────────────────────────────────────────────────┐
│                  Multi-Tenant API Layer                     │
│  ┌─────────────────┐          ┌────────────────────┐      │
│  │   FastAPI       │          │   Tenant Auth      │      │
│  │   + JWT Auth    │          │   Context Injection│      │
│  └────────┬────────┘          └────────────────────┘      │
├───────────┴─────────────────────────────────────────────────┤
│              Enhanced Pydantic AI Agent                    │
│  ┌─────────────────┐          ┌────────────────────┐      │
│  │  Tenant-Aware   │◄────────►│   Same 10+ Tools   │      │
│  │  Agent          │          │   + Tenant Context │      │
│  │  (Same LLMs)    │          │   Injection        │      │
│  └─────────────────┘          └────────────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                Multi-Tenant Data Layer                     │
│  ┌─────────────────┐          ┌────────────────────┐      │
│  │   Neon Projects │          │    Neo4j + Graphiti│      │
│  │   (Per-Tenant)  │          │    + group_id      │      │
│  │   Isolated DBs  │          │    Namespacing     │      │
│  └─────────────────┘          └────────────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                Tenant Management Layer                     │
│  ┌─────────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │   Catalog DB    │  │   Per-Tenant  │  │  Neon API    │ │
│  │   Tenant        │  │   Database    │  │  Integration │ │
│  │   Metadata      │  │   Complete    │  │  (Automated) │ │
│  │   (Catalog DB)  │  │   Isolation   │  │              │ │
│  └─────────────────┘  └───────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ **Deployment Steps**

### **Step 1: Environment Setup**

1. **Clone the repository and navigate to Tenant directory:**
   ```bash
   cd /path/to/agentic-rag-knowledge-graph/Tenant
   ```

2. **Create Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Create `requirements.txt`:**
   ```txt
   # Core Dependencies
   asyncpg==0.29.0
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   pydantic==2.5.0
   pydantic-ai==0.0.7
   python-jose[cryptography]==3.3.0
   python-multipart==0.0.6
   
   # Multi-tenant Architecture
   neon-api-client==0.2.0
   
   # Graph Database
   graphiti-core==0.1.0
   neo4j==5.15.0
   
   # AI and Embeddings
   openai==1.3.0
   
   # Vector Database
   pgvector==0.2.4
   
   # Additional utilities
   httpx==0.25.0
   asyncio-throttle==1.0.2
   ```

### **Step 2: Database Setup**

#### **2.1 Neon PostgreSQL Setup (Project-per-Tenant)**

1. **Create Neon account and get API key:**
   - Visit [neon.tech](https://neon.tech)
   - Create account and generate API key
   - This will be used for automated tenant project creation

2. **Create catalog database (control plane):**
   ```bash
   # Create main Neon project for catalog database
   # This is done once and manages all tenant projects
   psql "postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require"
   
   # Run the catalog schema
   \i catalog_schema.sql
   ```

3. **Create catalog schema (`catalog_schema.sql`):**
   ```sql
   -- Extensions
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   
   -- Tenant project mappings
   CREATE TABLE tenant_projects (
       tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       tenant_name VARCHAR(255) NOT NULL,
       tenant_email VARCHAR(255) UNIQUE NOT NULL,
       neon_project_id VARCHAR(100) NOT NULL UNIQUE,
       neon_database_url TEXT NOT NULL,
       region VARCHAR(50) NOT NULL DEFAULT 'aws-us-east-1',
       status VARCHAR(20) DEFAULT 'active',
       created_at TIMESTAMPTZ DEFAULT NOW(),
       updated_at TIMESTAMPTZ DEFAULT NOW()
   );
   
   -- Tenant configurations
   CREATE TABLE tenant_configs (
       tenant_id UUID PRIMARY KEY REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
       settings JSONB DEFAULT '{}',
       feature_flags JSONB DEFAULT '{}',
       updated_at TIMESTAMPTZ DEFAULT NOW()
   );
   
   -- Usage tracking for billing
   CREATE TABLE tenant_usage (
       id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
       tenant_id UUID REFERENCES tenant_projects(tenant_id) ON DELETE CASCADE,
       metric_name VARCHAR(100) NOT NULL,
       metric_value DECIMAL NOT NULL,
       period_date DATE NOT NULL,
       created_at TIMESTAMPTZ DEFAULT NOW(),
       UNIQUE(tenant_id, metric_name, period_date)
   );
   
   -- Performance indexes
   CREATE INDEX idx_tenant_projects_status ON tenant_projects(status);
   CREATE INDEX idx_tenant_projects_created ON tenant_projects(created_at);
   CREATE INDEX idx_tenant_usage_period ON tenant_usage(tenant_id, period_date);
   ```

4. **Verify catalog database:**
   ```sql
   SELECT table_name FROM information_schema.tables 
   WHERE table_schema = 'public' 
   AND table_name IN ('tenant_projects', 'tenant_configs', 'tenant_usage');
   ```

**Note**: Individual tenant databases are created automatically via Neon API when tenants are provisioned.

#### **2.2 Neo4j Setup**

1. **Option A: Neo4j Cloud (Recommended)**
   - Visit [neo4j.com/cloud](https://neo4j.com/cloud)
   - Create AuraDB instance
   - Note connection URI, username, and password

2. **Option B: Self-hosted Neo4j**
   ```bash
   # Using Docker
   docker run \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -d \
     -v $HOME/neo4j/data:/data \
     -v $HOME/neo4j/logs:/logs \
     -v $HOME/neo4j/import:/var/lib/neo4j/import \
     -v $HOME/neo4j/plugins:/plugins \
     --env NEO4J_AUTH=neo4j/password \
     neo4j:latest
   ```

3. **Install Graphiti:**
   ```bash
   pip install graphiti-core
   ```

### **Step 3: Configuration**

1. **Create `.env` file:**
   ```bash
   # Neon Configuration (Project-per-Tenant)
   NEON_API_KEY=your_neon_api_key_here
   CATALOG_DATABASE_URL=postgresql://catalog_user:pass@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require
   
   # Neo4j Configuration (Shared with namespacing)
   NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   
   # Authentication
   JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
   JWT_ALGORITHM=HS256
   
   # OpenAI (or alternative)
   OPENAI_API_KEY=your_openai_api_key
   
   # Application
   APP_ENV=production
   APP_HOST=0.0.0.0
   APP_PORT=8000
   LOG_LEVEL=info
   ```

2. **Create `config.py`:**
   ```python
   import os
   from typing import Optional
   
   class Config:
       # Neon Configuration (Project-per-Tenant)
       NEON_API_KEY = os.getenv("NEON_API_KEY")
       CATALOG_DATABASE_URL = os.getenv("CATALOG_DATABASE_URL")
       
       # Neo4j Configuration (Shared with namespacing)
       NEO4J_URI = os.getenv("NEO4J_URI")
       NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
       NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
       
       # JWT Configuration
       JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
       JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
       
       # OpenAI Configuration
       OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
       
       # Application Configuration
       APP_ENV = os.getenv("APP_ENV", "development")
       APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
       APP_PORT = int(os.getenv("APP_PORT", 8000))
       LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
       
       @classmethod
       def validate(cls):
           required_vars = [
               "NEON_API_KEY",
               "CATALOG_DATABASE_URL",
               "NEO4J_URI", 
               "NEO4J_PASSWORD",
               "JWT_SECRET_KEY",
               "OPENAI_API_KEY"
           ]
           
           missing = [var for var in required_vars if not getattr(cls, var)]
           if missing:
               raise ValueError(f"Missing required environment variables: {missing}")
   
   config = Config()
   ```

### **Step 4: Application Deployment**

#### **4.1 Create Main Application File**

Create `main.py`:
```python
import logging
import uvicorn
from multi_tenant_api import create_app
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Validate configuration
config.validate()

# Create FastAPI app with project-per-tenant architecture
app = create_app(
    neon_api_key=config.NEON_API_KEY,
    catalog_database_url=config.CATALOG_DATABASE_URL,
    neo4j_uri=config.NEO4J_URI,
    neo4j_user=config.NEO4J_USER,
    neo4j_password=config.NEO4J_PASSWORD,
    jwt_secret_key=config.JWT_SECRET_KEY,
    openai_api_key=config.OPENAI_API_KEY,
    title="Multi-Tenant RAG API (Project-per-Tenant)",
    version="2.0.0"
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=(config.APP_ENV == "development"),
        log_level=config.LOG_LEVEL
    )
```

#### **4.2 Test the Deployment**

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Verify health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Access API documentation:**
   - Open browser to `http://localhost:8000/docs`

### **Step 5: Production Deployment**

#### **5.1 Using Docker**

1. **Create `Dockerfile`:**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application
   COPY . .
   
   # Create non-root user
   RUN useradd --create-home --shell /bin/bash appuser
   USER appuser
   
   EXPOSE 8000
   
   CMD ["python", "main.py"]
   ```

2. **Create `docker-compose.yml`:**
   ```yaml
   version: '3.8'
   
   services:
     app:
       build: .
       ports:
         - "8000:8000"
       environment:
         - NEON_CONNECTION_STRING=${NEON_CONNECTION_STRING}
         - NEO4J_URI=${NEO4J_URI}
         - NEO4J_USER=${NEO4J_USER}
         - NEO4J_PASSWORD=${NEO4J_PASSWORD}
         - JWT_SECRET_KEY=${JWT_SECRET_KEY}
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - APP_ENV=production
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **Deploy with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

#### **5.2 Production Checklist**

- [ ] **Security:**
  - [ ] Strong JWT secret key
  - [ ] HTTPS/TLS certificates
  - [ ] Environment variables secured
  - [ ] Database connections encrypted

- [ ] **Performance:**
  - [ ] Connection pooling configured
  - [ ] Database indexes optimized
  - [ ] Rate limiting enabled
  - [ ] Caching implemented (Redis)

- [ ] **Monitoring:**
  - [ ] Application logs centralized
  - [ ] Health checks configured
  - [ ] Performance metrics collected
  - [ ] Error tracking enabled

- [ ] **Backup:**
  - [ ] Database backup strategy
  - [ ] Disaster recovery plan
  - [ ] Data retention policies

## 🧪 **Testing Your Deployment**

### **1. Create a Test Tenant**

```bash
curl -X POST "http://localhost:8000/tenants" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_corp",
    "name": "Test Corporation",
    "email": "test@corp.com",
    "max_documents": 100,
    "max_storage_mb": 50
  }'
```

### **2. Get Authentication Token**

```bash
curl -X POST "http://localhost:8000/auth/token?tenant_id=test_corp&user_id=test_user"
```

### **3. Create a Document**

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Document",
    "source": "test.md",
    "content": "This is a test document for the multi-tenant RAG system."
  }'
```

### **4. Query the System**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "use_vector": true,
    "use_graph": true,
    "max_results": 5
  }'
```

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Database Connection Errors**
   - Verify connection strings
   - Check firewall/security groups
   - Ensure SSL certificates are valid

2. **Authentication Failures**
   - Verify JWT secret key
   - Check token expiration
   - Validate tenant permissions

3. **Performance Issues**
   - Monitor database connections
   - Check query performance
   - Review memory usage

### **Debug Commands**

```bash
# Check application logs
docker logs container_name

# Test database connectivity
python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('YOUR_CONNECTION_STRING')
    print('Connected successfully')
    await conn.close()
asyncio.run(test())
"

# Validate JWT tokens
python -c "
from jose import jwt
token = 'YOUR_TOKEN'
decoded = jwt.decode(token, 'YOUR_SECRET', algorithms=['HS256'])
print(decoded)
"
```

## 🚀 **Scaling Considerations**

### **Horizontal Scaling**
- Deploy multiple API instances behind load balancer
- Use Redis for session management
- Implement database read replicas

### **Performance Optimization**
- Enable connection pooling (configured in tenant_manager.py)
- Implement caching layers (Redis/Memcached)
- Use database query optimization
- Enable async operations

### **Security Hardening**
- Implement rate limiting per tenant
- Add request/response logging
- Enable CORS for specific domains
- Use secrets management (AWS Secrets Manager, etc.)

## 📈 **Monitoring and Maintenance**

### **Key Metrics to Monitor**
- Response times per tenant
- Database connection pools
- Memory and CPU usage
- Error rates and authentication failures
- Tenant usage statistics

### **Maintenance Tasks**
- Regular database maintenance (VACUUM, ANALYZE)
- Log rotation and cleanup
- Security updates
- Performance tuning
- Backup verification

---

## 🎯 **Next Steps**

Once deployed successfully:

1. **Add More Tenants**: Scale to multiple clients
2. **Implement Caching**: Add Redis for performance
3. **Set Up Monitoring**: Use Prometheus/Grafana
4. **Add Features**: Implement advanced RAG features
5. **Scale Infrastructure**: Use Kubernetes for orchestration

Your multi-tenant RAG system is now ready for production use with complete data isolation and industry-standard security!
