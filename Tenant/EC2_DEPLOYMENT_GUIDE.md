# Amazon EC2 Deployment Guide - Multi-Tenant RAG CLI/API

## 🚀 **Overview**

This guide provides complete step-by-step instructions for deploying your interactive CLI FastAPI application on Amazon EC2, integrating with your existing:
- **Neon PostgreSQL Cloud** (tenant data + vector/hybrid search)
- **Neo4j Desktop Local** (graph data with Graphiti namespacing)

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Amazon EC2 Instance                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Your Application Stack                         ││
│  │  ┌─────────────────┐    ┌─────────────────────────────────┐ ││
│  │  │   FastAPI       │    │   Interactive CLI               │ ││
│  │  │   Server        │    │   HTTP Client                   │ ││
│  │  │   (Port 8000)   │    │                                 │ ││
│  │  └─────────────────┘    └─────────────────────────────────┘ ││
│  │  ┌─────────────────────────────────────────────────────────┐ ││
│  │  │         Multi-Tenant Agent & Dependencies              │ ││
│  │  │  • TenantManager   • Auth Middleware                   │ ││
│  │  │  • Pydantic AI     • Search Services                   │ ││
│  │  └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌─────────────────────────────┐    ┌──────────────────────────────┐
│       Neon PostgreSQL       │    │       Neo4j Desktop          │
│         (Cloud)             │    │        (Local)               │
│  • Tenant Databases         │    │  • Graph Data                │
│  • Vector Search (pgvector) │    │  • Graphiti Namespacing     │
│  • Hybrid Search (BM25)     │    │  • Tenant Isolation         │
│  • Catalog Database         │    │                              │
└─────────────────────────────┘    └──────────────────────────────┘
```

## 📋 **Prerequisites**

### **AWS Requirements**
- AWS account with EC2 access
- Key pair for SSH access
- Security group configured for HTTP/HTTPS traffic
- Elastic IP (recommended for production)

### **Local Requirements (Your Machine)**
- Neo4j Desktop running locally
- Neon PostgreSQL access credentials
- Your application code

### **Network Requirements**
- Your local Neo4j Desktop accessible from EC2 (port 7687)
- Stable internet connection for Neon PostgreSQL

## 🖥️ **Step 1: EC2 Instance Setup**

### **1.1 Launch EC2 Instance**

1. **Log into AWS Console** → EC2 Dashboard
2. **Launch Instance:**
   ```
   Name: multi-tenant-rag-api
   AMI: Ubuntu Server 22.04 LTS (64-bit x86)
   Instance Type: t3.medium (2 vCPU, 4 GB RAM) or larger
   Key Pair: Your existing key pair
   Storage: 20 GB gp3 (minimum)
   ```

3. **Configure Security Group:**
   ```
   Inbound Rules:
   - SSH (22): Your IP
   - HTTP (80): 0.0.0.0/0
   - HTTPS (443): 0.0.0.0/0
   - Custom TCP (8000): 0.0.0.0/0  # FastAPI
   - Custom TCP (7687): Your Local IP  # Neo4j access
   ```

4. **Launch and Connect:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-public-ip
   ```

### **1.2 System Updates and Basic Setup**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    htop \
    nginx \
    supervisor \
    postgresql-client

# Verify Python version (should be 3.10+)
python3 --version
```

## 🐍 **Step 2: Python Environment Setup**

### **2.1 Create Application User and Directory**

```bash
# Create application user
sudo useradd -m -s /bin/bash appuser
sudo usermod -aG sudo appuser

# Create application directory
sudo mkdir -p /opt/multi-tenant-rag
sudo chown appuser:appuser /opt/multi-tenant-rag

# Switch to app user
sudo su - appuser
cd /opt/multi-tenant-rag
```

### **2.2 Upload Your Application Code**

**Option A: Using SCP (from your local machine):**
```bash
# From your local machine
scp -i your-key.pem -r /Users/rahul/Desktop/Graphiti/agentic-rag-knowledge-graph/Tenant ubuntu@your-ec2-ip:/tmp/

# On EC2, move to app directory
sudo mv /tmp/Tenant/* /opt/multi-tenant-rag/
sudo chown -R appuser:appuser /opt/multi-tenant-rag/
```

**Option B: Using Git (if you have a repository):**
```bash
# Clone your repository
git clone https://github.com/yourusername/your-repo.git .
```

### **2.3 Python Virtual Environment**

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## 📦 **Step 3: Dependencies Installation**

### **3.1 Create Complete Requirements File**

Create `/opt/multi-tenant-rag/requirements_ec2.txt`:

```bash
cat > requirements_ec2.txt << 'EOF'
# Core FastAPI and Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0

# Pydantic and AI Agent
pydantic==2.5.0
pydantic-ai==0.0.12

# HTTP Client and CLI
httpx==0.25.0
click==8.1.7
rich==13.7.0

# Database Drivers
asyncpg==0.29.0
psycopg2-binary==2.9.9
neo4j==5.15.0

# Vector and Search
pgvector==0.2.4

# Graphiti for Knowledge Graphs
graphiti-core==0.3.7

# AI and LLM Providers
openai==1.3.0
google-generativeai==0.3.2
anthropic==0.8.1

# Authentication and Security
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0

# Data Processing
pandas==2.1.4
numpy==1.24.4

# Async and Concurrency
asyncio-throttle==1.0.2
aiofiles==23.2.1

# Logging and Monitoring
structlog==23.2.0

# Additional utilities
uuid==1.30
python-dateutil==2.8.2
requests==2.31.0

# Production WSGI server
gunicorn==21.2.0
EOF
```

### **3.2 Install Dependencies**

```bash
# Install all dependencies
pip install -r requirements_ec2.txt

# Verify critical packages
python3 -c "import fastapi; import pydantic_ai; import neo4j; import asyncpg; import graphiti; print('✅ All critical packages installed')"
```

## 🔧 **Step 4: Configuration Setup**

### **4.1 Environment Configuration**

Create `/opt/multi-tenant-rag/.env`:

```bash
cat > .env << 'EOF'
# Neon PostgreSQL Configuration (Cloud)
NEON_API_KEY=your_neon_api_key_here
CATALOG_DB_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require
POSTGRES_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require

# Neo4j Configuration (Local Desktop - accessible from EC2)
NEO4J_URI=neo4j://YOUR_LOCAL_IP:7687
NEO4J_URL=neo4j://YOUR_LOCAL_IP:7687
NEO4J_USERNAME=neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_AUTH=neo4j/your_neo4j_password

# Authentication
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production-$(openssl rand -hex 32)
JWT_ALGORITHM=HS256

# AI Providers
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Application Configuration
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=info

# Neo4j Connection Timeout
NEO4J_CONNECTION_TIMEOUT=30
NEO4J_MAX_CONNECTION_LIFETIME=300

# Vector Search Configuration
VECTOR_DIMENSION=1536
EMBEDDING_MODEL=text-embedding-ada-002
EOF
```

**⚠️ Important:** Replace the placeholder values:
- `your_neon_api_key_here`: Your actual Neon API key
- `ep-catalog.us-east-2.aws.neon.tech/neondb`: Your actual Neon connection string
- `YOUR_LOCAL_IP`: Your local machine's public IP where Neo4j Desktop runs
- `your_neo4j_password`: Your Neo4j Desktop password
- API keys for AI providers

### **4.2 Secure Environment File**

```bash
# Set proper permissions
chmod 600 .env
chown appuser:appuser .env
```

### **4.3 Test Database Connections**

```bash
# Test Neon PostgreSQL connection
python3 -c "
import asyncpg
import asyncio
import os
from dotenv import load_dotenv

async def test_neon():
    load_dotenv()
    try:
        conn = await asyncpg.connect(os.getenv('CATALOG_DB_URL'))
        result = await conn.fetchval('SELECT version()')
        print(f'✅ Neon PostgreSQL: {result[:50]}...')
        await conn.close()
    except Exception as e:
        print(f'❌ Neon PostgreSQL Error: {e}')

asyncio.run(test_neon())
"

# Test Neo4j connection
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

def test_neo4j():
    load_dotenv()
    try:
        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
        )
        with driver.session() as session:
            result = session.run('RETURN \"Hello Neo4j\" as message')
            record = result.single()
            print(f'✅ Neo4j: {record[\"message\"]}')
        driver.close()
    except Exception as e:
        print(f'❌ Neo4j Error: {e}')

test_neo4j()
"
```

## 🚀 **Step 5: Application Deployment**

### **5.1 Create Startup Script**

Create `/opt/multi-tenant-rag/start_api.sh`:

```bash
cat > start_api.sh << 'EOF'
#!/bin/bash

# Multi-Tenant RAG API Startup Script
cd /opt/multi-tenant-rag

# Activate virtual environment
source venv/bin/activate

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Start the FastAPI application
exec python3 interactive_multi_tenant_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
EOF

chmod +x start_api.sh
```

### **5.2 Create CLI Startup Script**

Create `/opt/multi-tenant-rag/start_cli.sh`:

```bash
cat > start_cli.sh << 'EOF'
#!/bin/bash

# Multi-Tenant RAG CLI Startup Script
cd /opt/multi-tenant-rag

# Activate virtual environment
source venv/bin/activate

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Start the CLI client
exec python3 interactive_multi_tenant_cli_http.py \
    --api-url http://localhost:8000 \
    "$@"
EOF

chmod +x start_cli.sh
```

### **5.3 Test Manual Startup**

```bash
# Test API startup
./start_api.sh &

# Wait a few seconds, then test
sleep 5
curl http://localhost:8000/health

# If successful, stop the test
pkill -f "interactive_multi_tenant_api.py"
```

## 📋 **Step 6: Process Management with Supervisor**

### **6.1 Install and Configure Supervisor**

```bash
# Create supervisor configuration
sudo tee /etc/supervisor/conf.d/multi-tenant-rag.conf << 'EOF'
[program:multi-tenant-rag-api]
command=/opt/multi-tenant-rag/start_api.sh
directory=/opt/multi-tenant-rag
user=appuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/multi-tenant-rag-api.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=PATH="/opt/multi-tenant-rag/venv/bin"

[program:multi-tenant-rag-api-worker]
command=/opt/multi-tenant-rag/venv/bin/python3 interactive_multi_tenant_api.py --host 0.0.0.0 --port 8000
directory=/opt/multi-tenant-rag
user=appuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/multi-tenant-rag-worker.log
environment=PATH="/opt/multi-tenant-rag/venv/bin"
EOF
```

### **6.2 Start Services**

```bash
# Reload supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update

# Start the application
sudo supervisorctl start multi-tenant-rag-api

# Check status
sudo supervisorctl status

# View logs
sudo tail -f /var/log/multi-tenant-rag-api.log
```

## 🌐 **Step 7: Nginx Reverse Proxy Setup**

### **7.1 Configure Nginx**

```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/multi-tenant-rag << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain or use IP

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    # API documentation
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
    }

    # File size limits
    client_max_body_size 100M;
}
EOF

# Enable the site
sudo ln -s /etc/nginx/sites-available/multi-tenant-rag /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

## 🔒 **Step 8: Neo4j Desktop Connection Setup**

### **8.1 Configure Neo4j Desktop for Remote Access**

**On your local machine (where Neo4j Desktop runs):**

1. **Open Neo4j Desktop**
2. **Go to your database → Settings**
3. **Add these configurations:**
   ```
   # Enable remote connections
   dbms.default_listen_address=0.0.0.0
   dbms.connector.bolt.listen_address=0.0.0.0:7687
   dbms.connector.http.listen_address=0.0.0.0:7474
   
   # Security settings
   dbms.security.auth_enabled=true
   ```

4. **Restart your Neo4j database**

### **8.2 Configure Local Firewall**

**For macOS:**
```bash
# Allow Neo4j ports (if firewall is enabled)
sudo pfctl -f /etc/pf.conf
```

**For Windows:**
```powershell
# Allow Neo4j ports through Windows Firewall
New-NetFirewallRule -DisplayName "Neo4j-Bolt" -Direction Inbound -Port 7687 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Neo4j-HTTP" -Direction Inbound -Port 7474 -Protocol TCP -Action Allow
```

### **8.3 Test Connection from EC2**

```bash
# From your EC2 instance, test Neo4j connection
python3 -c "
from neo4j import GraphDatabase
import os

# Replace with your actual values
NEO4J_URI = 'neo4j://YOUR_LOCAL_PUBLIC_IP:7687'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = 'your_password'

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run('MATCH (n) RETURN count(n) as node_count')
        record = result.single()
        print(f'✅ Neo4j Connection: {record[\"node_count\"]} nodes found')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j Connection Error: {e}')
"
```

## 🧪 **Step 9: Deployment Testing**

### **9.1 API Health Check**

```bash
# Test from EC2
curl http://localhost:8000/health

# Test from external (replace with your EC2 public IP)
curl http://your-ec2-public-ip/health
```

### **9.2 Complete Functionality Test**

```bash
# Create a test tenant
curl -X POST "http://your-ec2-public-ip/tenants" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Corp",
    "email": "test@corp.com",
    "region": "aws-us-east-1",
    "plan": "basic"
  }'

# Authenticate
curl -X POST "http://your-ec2-public-ip/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "TENANT_ID_FROM_ABOVE",
    "api_key": "test_key",
    "user_id": "test_user"
  }'

# Test search (replace TOKEN with actual token)
curl -X POST "http://your-ec2-public-ip/search" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "search_type": "vector",
    "limit": 5
  }'
```

### **9.3 CLI Testing**

```bash
# Test CLI from EC2
cd /opt/multi-tenant-rag
./start_cli.sh --api-url http://localhost:8000

# Test CLI from your local machine
python3 interactive_multi_tenant_cli_http.py --api-url http://your-ec2-public-ip
```

## 📊 **Step 10: Monitoring and Maintenance**

### **10.1 Log Monitoring**

```bash
# Application logs
sudo tail -f /var/log/multi-tenant-rag-api.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log

# System logs
sudo journalctl -u supervisor -f
```

### **10.2 Performance Monitoring**

Create `/opt/multi-tenant-rag/monitor.sh`:

```bash
cat > monitor.sh << 'EOF'
#!/bin/bash

echo "=== Multi-Tenant RAG System Status ==="
echo "Date: $(date)"
echo ""

echo "=== System Resources ==="
free -h
df -h /
echo ""

echo "=== Application Status ==="
sudo supervisorctl status
echo ""

echo "=== Database Connections ==="
# Test Neon PostgreSQL
python3 -c "
import asyncpg, asyncio, os
from dotenv import load_dotenv
async def test():
    load_dotenv()
    try:
        conn = await asyncpg.connect(os.getenv('CATALOG_DB_URL'))
        await conn.fetchval('SELECT 1')
        print('✅ Neon PostgreSQL: Connected')
        await conn.close()
    except Exception as e:
        print(f'❌ Neon PostgreSQL: {e}')
asyncio.run(test())
"

# Test Neo4j
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
load_dotenv()
try:
    driver = GraphDatabase.driver(os.getenv('NEO4J_URI'), auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')))
    with driver.session() as session:
        session.run('RETURN 1')
    print('✅ Neo4j: Connected')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j: {e}')
"

echo ""
echo "=== API Health ==="
curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not responding"
EOF

chmod +x monitor.sh
```

### **10.3 Automated Monitoring with Cron**

```bash
# Add monitoring cron job
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/multi-tenant-rag/monitor.sh >> /var/log/rag-monitor.log 2>&1") | crontab -
```

## 🔧 **Step 11: Production Optimizations**

### **11.1 SSL/HTTPS Setup with Let's Encrypt**

```bash
# Install Certbot
sudo apt install snapd
sudo snap install --classic certbot

# Create certificate (replace your-domain.com)
sudo certbot --nginx -d your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### **11.2 Database Connection Pooling**

Add to your `.env`:
```bash
# Database pool settings
MAX_POOL_SIZE=20
MIN_POOL_SIZE=5
POOL_TIMEOUT=30
```

### **11.3 Application Performance Tuning**

Create `/opt/multi-tenant-rag/gunicorn.conf.py`:

```python
# Gunicorn configuration
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 120
keepalive = 5
preload_app = True
```

Update supervisor config to use Gunicorn:
```bash
sudo tee /etc/supervisor/conf.d/multi-tenant-rag-gunicorn.conf << 'EOF'
[program:multi-tenant-rag-gunicorn]
command=/opt/multi-tenant-rag/venv/bin/gunicorn -c gunicorn.conf.py interactive_multi_tenant_api:app
directory=/opt/multi-tenant-rag
user=appuser
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/multi-tenant-rag-gunicorn.log
environment=PATH="/opt/multi-tenant-rag/venv/bin"
EOF
```

## 🚨 **Troubleshooting Guide**

### **Common Issues and Solutions**

1. **Neo4j Connection Timeout**
   ```bash
   # Check local Neo4j status
   # Verify firewall settings
   # Test network connectivity: telnet YOUR_LOCAL_IP 7687
   ```

2. **Neon PostgreSQL SSL Issues**
   ```bash
   # Ensure SSL mode in connection string
   # Update connection string: ?sslmode=require
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   free -h
   
   # Adjust worker processes in gunicorn.conf.py
   workers = 2  # Reduce if memory limited
   ```

4. **API Not Responding**
   ```bash
   # Check supervisor status
   sudo supervisorctl status
   
   # Restart services
   sudo supervisorctl restart all
   
   # Check logs
   sudo tail -f /var/log/multi-tenant-rag-api.log
   ```

### **Quick Diagnostic Commands**

```bash
# Full system check
cd /opt/multi-tenant-rag
./monitor.sh

# Restart all services
sudo supervisorctl restart all
sudo systemctl restart nginx

# Check all ports
sudo netstat -tlnp | grep -E "(8000|80|443)"
```

## ✅ **Deployment Verification Checklist**

- [ ] **EC2 Instance Running** - Instance accessible via SSH
- [ ] **Python Environment** - Virtual environment activated, dependencies installed
- [ ] **Environment Variables** - All `.env` variables configured correctly
- [ ] **Database Connections** - Both Neon PostgreSQL and Neo4j accessible
- [ ] **Application Services** - FastAPI running via Supervisor
- [ ] **Reverse Proxy** - Nginx configured and running
- [ ] **External Access** - API accessible from internet
- [ ] **Health Checks** - `/health` endpoint responding
- [ ] **Authentication** - JWT token generation working
- [ ] **Search Functions** - Vector, graph, and hybrid search working
- [ ] **CLI Access** - Both local and remote CLI functionality
- [ ] **Monitoring** - Logs and monitoring scripts configured
- [ ] **Security** - Firewall rules, SSL certificates (if applicable)

## 🎯 **Final Steps**

1. **Document your deployment:**
   - EC2 instance details
   - Database connection strings
   - API endpoints
   - Access credentials

2. **Set up backups:**
   - Application code backup
   - Database backup strategy
   - Configuration files backup

3. **Monitor performance:**
   - Set up CloudWatch (optional)
   - Monitor resource usage
   - Track API response times

Your multi-tenant RAG application is now deployed and ready for production use! 🚀

## 📞 **Support Commands**

```bash
# Quick restart everything
sudo supervisorctl restart all && sudo systemctl restart nginx

# View all logs
sudo tail -f /var/log/multi-tenant-rag-api.log /var/log/nginx/error.log

# Check application status
cd /opt/multi-tenant-rag && ./monitor.sh
```
