#!/bin/bash

# Multi-Tenant RAG System - EC2 Quick Setup Script
# This script automates the initial setup on a fresh Ubuntu EC2 instance

set -e  # Exit on any error

echo "🚀 Multi-Tenant RAG System - EC2 Quick Setup"
echo "============================================="
echo "This script will set up your multi-tenant RAG system on EC2"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root"
    exit 1
fi

echo "=== STEP 1: System Update and Dependencies ==="
echo ""

print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

print_info "Installing essential packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libpq-dev \
    git \
    curl \
    wget \
    htop \
    nginx \
    supervisor \
    postgresql-client \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    bc
print_status "Essential packages installed"

echo ""
echo "=== STEP 2: Application Directory Setup ==="
echo ""

print_info "Creating application directory..."
sudo mkdir -p /opt/multi-tenant-rag
sudo chown $USER:$USER /opt/multi-tenant-rag
print_status "Application directory created: /opt/multi-tenant-rag"

print_info "Moving application files..."
if [ -d "/tmp/Tenant" ]; then
    cp -r /tmp/Tenant/* /opt/multi-tenant-rag/
    print_status "Application files copied from /tmp/Tenant"
elif [ "$(basename $(pwd))" = "Tenant" ]; then
    cp -r ./* /opt/multi-tenant-rag/
    print_status "Application files copied from current directory"
else
    print_warning "Application files not found. Please upload them to /opt/multi-tenant-rag/"
fi

cd /opt/multi-tenant-rag

echo ""
echo "=== STEP 3: Python Virtual Environment ==="
echo ""

print_info "Creating Python virtual environment..."
python3 -m venv venv
print_status "Virtual environment created"

print_info "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

print_info "Upgrading pip and installing wheel..."
pip install --upgrade pip setuptools wheel
print_status "Pip and wheel upgraded"

echo ""
echo "=== STEP 4: Python Dependencies Installation ==="
echo ""

print_info "Installing Python dependencies..."
if [ -f "requirements_ec2_complete.txt" ]; then
    pip install -r requirements_ec2_complete.txt
    print_status "Dependencies installed from requirements_ec2_complete.txt"
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Dependencies installed from requirements.txt"
else
    print_warning "No requirements file found. Installing critical packages..."
    pip install fastapi uvicorn pydantic-ai neo4j asyncpg graphiti-core openai httpx rich click python-jose python-dotenv
    print_status "Critical packages installed"
fi

echo ""
echo "=== STEP 5: Environment Configuration ==="
echo ""

if [ ! -f ".env" ]; then
    print_info "Creating .env template..."
    cat > .env << 'EOF'
# Neon PostgreSQL Configuration (REPLACE WITH YOUR VALUES)
NEON_API_KEY=your_neon_api_key_here
CATALOG_DB_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require
POSTGRES_URL=postgresql://username:password@ep-catalog.us-east-2.aws.neon.tech/neondb?sslmode=require

# Neo4j Configuration (REPLACE YOUR_LOCAL_IP)
NEO4J_URI=neo4j://YOUR_LOCAL_IP:7687
NEO4J_URL=neo4j://YOUR_LOCAL_IP:7687
NEO4J_USERNAME=neo4j
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_AUTH=neo4j/your_neo4j_password

# Authentication (GENERATE SECURE KEY)
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
JWT_ALGORITHM=HS256

# AI Providers (ADD YOUR KEYS)
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Application Configuration
APP_ENV=production
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=info
EOF
    
    # Generate a secure JWT secret
    JWT_SECRET=$(openssl rand -hex 32)
    sed -i "s/your-super-secret-jwt-key-change-in-production/$JWT_SECRET/" .env
    
    print_status ".env file created with template values"
    print_warning "IMPORTANT: Edit .env file with your actual database credentials!"
else
    print_status ".env file already exists"
fi

# Set secure permissions
chmod 600 .env
print_status "Secure permissions set on .env file"

echo ""
echo "=== STEP 6: Startup Scripts ==="
echo ""

print_info "Creating startup scripts..."

# API startup script
cat > start_api.sh << 'EOF'
#!/bin/bash
cd /opt/multi-tenant-rag
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
exec python3 interactive_multi_tenant_api.py --host 0.0.0.0 --port 8000 --log-level info
EOF

# CLI startup script
cat > start_cli.sh << 'EOF'
#!/bin/bash
cd /opt/multi-tenant-rag
source venv/bin/activate
export $(grep -v '^#' .env | xargs)
exec python3 interactive_multi_tenant_cli_http.py --api-url http://localhost:8000 "$@"
EOF

# Monitoring script
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
sudo supervisorctl status 2>/dev/null || echo "Supervisor not configured"
echo ""
echo "=== API Health ==="
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "API not responding"
EOF

chmod +x start_api.sh start_cli.sh monitor.sh
print_status "Startup scripts created and made executable"

echo ""
echo "=== STEP 7: Supervisor Configuration ==="
echo ""

print_info "Configuring Supervisor for process management..."
sudo tee /etc/supervisor/conf.d/multi-tenant-rag.conf > /dev/null << 'EOF'
[program:multi-tenant-rag-api]
command=/opt/multi-tenant-rag/start_api.sh
directory=/opt/multi-tenant-rag
user=ubuntu
autostart=false
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/multi-tenant-rag-api.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=5
environment=PATH="/opt/multi-tenant-rag/venv/bin"
EOF

sudo supervisorctl reread
sudo supervisorctl update
print_status "Supervisor configured"

echo ""
echo "=== STEP 8: Nginx Configuration ==="
echo ""

print_info "Configuring Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/multi-tenant-rag > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;

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
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }

    client_max_body_size 100M;
}
EOF

sudo ln -sf /etc/nginx/sites-available/multi-tenant-rag /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
if sudo nginx -t; then
    print_status "Nginx configured successfully"
    sudo systemctl restart nginx
    sudo systemctl enable nginx
else
    print_error "Nginx configuration test failed"
fi

echo ""
echo "=== STEP 9: Firewall Configuration ==="
echo ""

print_info "Configuring UFW firewall..."
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw allow 8000
sudo ufw --force enable
print_status "Firewall configured"

echo ""
echo "=== STEP 10: Final Setup ==="
echo ""

print_info "Setting up log directory..."
sudo mkdir -p /var/log
sudo touch /var/log/multi-tenant-rag-api.log
sudo chown ubuntu:ubuntu /var/log/multi-tenant-rag-api.log
print_status "Log directory configured"

print_info "Creating verification script..."
if [ -f "verify_deployment.sh" ]; then
    chmod +x verify_deployment.sh
    print_status "Deployment verification script ready"
fi

echo ""
echo "============================================="
echo "🎉 EC2 SETUP COMPLETE!"
echo "============================================="
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. 🔧 CONFIGURE YOUR ENVIRONMENT:"
echo "   Edit /opt/multi-tenant-rag/.env with your actual:"
echo "   • Neon PostgreSQL connection string"
echo "   • Neo4j Desktop IP and credentials"
echo "   • AI provider API keys"
echo ""
echo "2. 🔗 CONFIGURE NEO4J DESKTOP ACCESS:"
echo "   On your local machine with Neo4j Desktop:"
echo "   • Enable remote connections (0.0.0.0:7687)"
echo "   • Configure firewall to allow EC2 IP"
echo "   • Update .env with your public IP"
echo ""
echo "3. ✅ VERIFY DEPLOYMENT:"
echo "   cd /opt/multi-tenant-rag"
echo "   ./verify_deployment.sh"
echo ""
echo "4. 🚀 START SERVICES:"
echo "   sudo supervisorctl start multi-tenant-rag-api"
echo "   curl http://localhost/health"
echo ""
echo "5. 🖥️  TEST CLI:"
echo "   ./start_cli.sh"
echo ""
echo "📁 Important files:"
echo "   • Application: /opt/multi-tenant-rag/"
echo "   • Environment: /opt/multi-tenant-rag/.env"
echo "   • Logs: /var/log/multi-tenant-rag-api.log"
echo "   • Nginx config: /etc/nginx/sites-available/multi-tenant-rag"
echo "   • Supervisor config: /etc/supervisor/conf.d/multi-tenant-rag.conf"
echo ""
echo "🔍 Monitor your deployment:"
echo "   • ./monitor.sh"
echo "   • sudo supervisorctl status"
echo "   • sudo tail -f /var/log/multi-tenant-rag-api.log"
echo ""

print_warning "Remember to configure your database connections in .env before starting!"
echo ""
