# EC2 Deployment Configuration Checklist

## 🔧 Pre-Deployment Configuration

### **1. Neo4j Desktop Configuration (Local Machine)**

**Before deploying to EC2, configure your local Neo4j Desktop:**

#### Step 1: Enable Remote Access
1. Open Neo4j Desktop
2. Select your database → Settings (⚙️)
3. Add these configuration lines:
   ```
   # Enable remote connections
   dbms.default_listen_address=0.0.0.0
   dbms.connector.bolt.listen_address=0.0.0.0:7687
   dbms.connector.http.listen_address=0.0.0.0:7474
   
   # Security (ensure authentication is enabled)
   dbms.security.auth_enabled=true
   ```
4. Apply settings and restart the database

#### Step 2: Firewall Configuration

**For macOS:**
```bash
# Check if firewall is enabled
sudo pfctl -sr

# If using Application Firewall, allow Neo4j Desktop
# System Preferences → Security & Privacy → Firewall → Firewall Options
# Allow "Neo4j Desktop" and "Java" incoming connections
```

**For Windows:**
```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "Neo4j-Bolt" -Direction Inbound -Port 7687 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Neo4j-HTTP" -Direction Inbound -Port 7474 -Protocol TCP -Action Allow
```

**For Linux:**
```bash
sudo ufw allow 7687
sudo ufw allow 7474
```

#### Step 3: Get Your Public IP
```bash
# Find your public IP address
curl ifconfig.me
# or
curl ipinfo.io/ip
```

**Note this IP address - you'll need it for the EC2 .env configuration**

### **2. Neon PostgreSQL Configuration**

#### Collect Required Information:
- [ ] **Neon API Key**: From [Neon Console](https://console.neon.tech) → Account Settings → API Keys
- [ ] **Catalog Database URL**: Your main Neon project connection string
- [ ] **Region**: Note the region (e.g., `aws-us-east-1`, `aws-eu-west-1`)

#### Example connection string format:
```
postgresql://username:password@ep-example-123456.aws-region.neon.tech/dbname?sslmode=require
```

### **3. AI Provider API Keys**

Collect API keys from:
- [ ] **OpenAI**: [platform.openai.com](https://platform.openai.com/api-keys)
- [ ] **Google AI**: [ai.google.dev](https://ai.google.dev/)
- [ ] **Anthropic** (optional): [console.anthropic.com](https://console.anthropic.com/)

---

## 🚀 EC2 Deployment Steps

### **Step 1: Launch EC2 Instance**

#### Instance Configuration:
```
Name: multi-tenant-rag-api
AMI: Ubuntu Server 22.04 LTS
Instance Type: t3.medium (minimum) or t3.large (recommended)
Storage: 20 GB gp3 (minimum)
Key Pair: Your existing SSH key pair
```

#### Security Group Rules:
```
Inbound Rules:
- SSH (22): Your IP address
- HTTP (80): 0.0.0.0/0
- HTTPS (443): 0.0.0.0/0 (if using SSL)
- Custom TCP (8000): 0.0.0.0/0 (FastAPI direct access)
- Custom TCP (7687): YOUR_LOCAL_PUBLIC_IP/32 (Neo4j from your machine)
```

### **Step 2: Upload Application Files**

**Method A: SCP Upload**
```bash
# From your local machine (in the Tenant directory)
scp -i your-key.pem -r ./* ubuntu@your-ec2-public-ip:/tmp/Tenant/
```

**Method B: Git Clone** (if you have a repository)
```bash
# On EC2 instance
git clone https://github.com/yourusername/your-repo.git /tmp/Tenant
```

### **Step 3: Run Quick Setup Script**

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Run the setup script
cd /tmp/Tenant
chmod +x ec2_quick_setup.sh
./ec2_quick_setup.sh
```

### **Step 4: Configure Environment Variables**

```bash
# Edit the .env file with your actual values
cd /opt/multi-tenant-rag
nano .env
```

**Critical values to replace:**
```bash
# Replace with your actual Neon credentials
NEON_API_KEY=neon_api_1a2b3c4d5e6f...
CATALOG_DB_URL=postgresql://user:pass@ep-example.aws-us-east-1.neon.tech/db?sslmode=require

# Replace YOUR_LOCAL_IP with your actual public IP from Step 1.3
NEO4J_URI=neo4j://203.0.113.123:7687
NEO4J_PASSWORD=your_actual_neo4j_password

# Replace with your actual AI API keys
OPENAI_API_KEY=sk-1234567890abcdef...
GOOGLE_API_KEY=AIza1234567890abcdef...
```

### **Step 5: Verify Deployment**

```bash
# Run verification script
cd /opt/multi-tenant-rag
./verify_deployment.sh
```

**This will test:**
- [ ] Python environment and dependencies
- [ ] Neon PostgreSQL connection
- [ ] Neo4j Desktop connection
- [ ] Application file integrity
- [ ] API server functionality
- [ ] Security configuration

### **Step 6: Start Services**

```bash
# Start the application via Supervisor
sudo supervisorctl start multi-tenant-rag-api

# Check status
sudo supervisorctl status

# Test health endpoint
curl http://localhost/health

# View logs
sudo tail -f /var/log/multi-tenant-rag-api.log
```

---

## ✅ Post-Deployment Testing

### **1. API Endpoint Testing**

```bash
# Health check
curl http://your-ec2-public-ip/health

# Create tenant
curl -X POST "http://your-ec2-public-ip/tenants" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Corp",
    "email": "test@corp.com",
    "region": "aws-us-east-1",
    "plan": "basic"
  }'

# Authenticate (use tenant_id from above response)
curl -X POST "http://your-ec2-public-ip/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "TENANT_ID_HERE",
    "api_key": "test_key",
    "user_id": "test_user"
  }'

# Test search (use token from above response)
curl -X POST "http://your-ec2-public-ip/search" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test search",
    "search_type": "vector",
    "limit": 5
  }'
```

### **2. CLI Testing**

```bash
# Test CLI locally
cd /opt/multi-tenant-rag
./start_cli.sh --api-url http://localhost:8000

# Test CLI from external machine
python3 interactive_multi_tenant_cli_http.py --api-url http://your-ec2-public-ip
```

### **3. Database Integration Testing**

```bash
# Test Neon PostgreSQL operations
python3 -c "
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def test_neon():
    load_dotenv()
    conn = await asyncpg.connect(os.getenv('CATALOG_DB_URL'))
    
    # Test creating a table
    await conn.execute('CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY, name TEXT)')
    await conn.execute('INSERT INTO test_table (name) VALUES ($1)', 'test')
    result = await conn.fetchval('SELECT name FROM test_table WHERE name = $1', 'test')
    await conn.execute('DROP TABLE test_table')
    await conn.close()
    
    print(f'Neon PostgreSQL test: {result}')

asyncio.run(test_neon())
"

# Test Neo4j operations
python3 -c "
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)

with driver.session() as session:
    # Test creating and querying a node
    session.run('CREATE (n:TestNode {name: \"deployment_test\"}) RETURN n')
    result = session.run('MATCH (n:TestNode {name: \"deployment_test\"}) RETURN n.name as name')
    record = result.single()
    session.run('MATCH (n:TestNode {name: \"deployment_test\"}) DELETE n')
    
print(f'Neo4j test: {record[\"name\"]}')
driver.close()
"
```

---

## 🔍 Troubleshooting

### **Common Issues and Solutions**

#### 1. **Neo4j Connection Failed**
```bash
# Check if Neo4j is accessible from EC2
telnet YOUR_LOCAL_IP 7687

# Verify firewall settings on local machine
# Ensure Neo4j Desktop is configured for remote access
# Check if your public IP has changed
```

#### 2. **Neon PostgreSQL SSL Error**
```bash
# Ensure connection string includes SSL mode
# Example: postgresql://user:pass@host/db?sslmode=require
```

#### 3. **API Not Starting**
```bash
# Check logs
sudo tail -f /var/log/multi-tenant-rag-api.log

# Check dependencies
cd /opt/multi-tenant-rag
source venv/bin/activate
python3 -c "import fastapi, pydantic_ai, neo4j, asyncpg"
```

#### 4. **Permission Errors**
```bash
# Fix file ownership
sudo chown -R ubuntu:ubuntu /opt/multi-tenant-rag

# Fix .env permissions
chmod 600 /opt/multi-tenant-rag/.env
```

#### 5. **Memory Issues**
```bash
# Monitor memory usage
free -h
htop

# Consider upgrading to t3.large if using t3.medium
# Reduce worker processes in gunicorn configuration
```

### **Diagnostic Commands**

```bash
# System status
./monitor.sh

# Service status
sudo supervisorctl status
sudo systemctl status nginx

# Network connectivity
netstat -tlnp | grep -E "(8000|80|7687)"

# Process monitoring
ps aux | grep -E "(python|nginx|supervisor)"

# Disk usage
df -h
du -sh /opt/multi-tenant-rag/*
```

---

## 🛡️ Production Security Checklist

- [ ] **SSL/TLS Certificate** - Configure Let's Encrypt for HTTPS
- [ ] **Environment Variables** - Secure .env file (chmod 600)
- [ ] **JWT Secret** - Strong, randomly generated secret key
- [ ] **Database Credentials** - Use strong passwords
- [ ] **Firewall Rules** - Restrict access to necessary ports only
- [ ] **Regular Updates** - Keep system and dependencies updated
- [ ] **Backup Strategy** - Configure automated backups
- [ ] **Monitoring** - Set up application and infrastructure monitoring
- [ ] **Log Management** - Configure log rotation and retention
- [ ] **Access Control** - Limit SSH access and use key-based authentication

---

## 📋 Maintenance Commands

```bash
# Restart all services
sudo supervisorctl restart all
sudo systemctl restart nginx

# View application logs
sudo tail -f /var/log/multi-tenant-rag-api.log

# Update application code
cd /opt/multi-tenant-rag
git pull  # if using git
sudo supervisorctl restart multi-tenant-rag-api

# System monitoring
./monitor.sh

# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz .env *.py *.sh
```

This completes your EC2 deployment configuration checklist! 🚀
