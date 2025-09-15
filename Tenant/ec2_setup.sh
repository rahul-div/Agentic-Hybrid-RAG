#!/bin/bash

# Multi-Tenant RAG Prototype - EC2 Quick Setup Script
# Usage: curl -fsSL https://raw.githubusercontent.com/your-repo/setup.sh | bash
# Or: wget -O - https://raw.githubusercontent.com/your-repo/setup.sh | bash

set -e

echo "🚀 Multi-Tenant RAG Prototype - EC2 Setup"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
if [[ $(lsb_release -rs) != "22.04" ]]; then
    log_warn "This script is designed for Ubuntu 22.04. Your version: $(lsb_release -rs)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
log_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
log_info "Installing essential packages..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    nginx \
    htop \
    unzip \
    jq

# Install Docker
log_info "Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    log_info "Docker installed. You may need to log out and back in for group changes to take effect."
else
    log_info "Docker already installed."
fi

# Install Node.js and npm packages
log_info "Installing Node.js and global packages..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

sudo npm install -g neonctl pm2

# Create application directory
log_info "Setting up application directory..."
cd /home/ubuntu
if [ ! -d "multitenant-rag" ]; then
    log_warn "Please clone your repository manually:"
    echo "  git clone <your-repo-url> multitenant-rag"
    echo "  cd multitenant-rag/Tenant"
else
    log_info "Application directory already exists."
fi

# Setup Neo4j with Docker
log_info "Setting up Neo4j..."
if ! docker ps | grep -q neo4j-prototype; then
    docker run -d \
      --name neo4j-prototype \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/prototype123 \
      -v neo4j_data:/data \
      -v neo4j_logs:/logs \
      --restart unless-stopped \
      neo4j:5.15
    
    log_info "Waiting for Neo4j to start..."
    sleep 30
    
    if docker ps | grep -q neo4j-prototype; then
        log_info "Neo4j started successfully!"
        echo "  Web Interface: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):7474"
        echo "  Bolt URI: bolt://localhost:7687"
        echo "  Username: neo4j"
        echo "  Password: prototype123"
    else
        log_error "Neo4j failed to start. Check logs: docker logs neo4j-prototype"
    fi
else
    log_info "Neo4j container already running."
fi

# Create logs directory
mkdir -p /home/ubuntu/logs

# Create environment template
log_info "Creating environment template..."
cat > /home/ubuntu/multitenant-rag/Tenant/.env.production.template << 'EOF'
# Neon Configuration - UPDATE THESE VALUES
NEON_API_KEY=your_neon_api_key_here
CATALOG_DATABASE_URL=postgresql://user:pass@ep-catalog.us-east-2.aws.neon.tech/neondb
TENANT_A_CONNECTION_STRING=postgresql://user:pass@ep-tenant-a.us-east-2.aws.neon.tech/neondb
TENANT_B_CONNECTION_STRING=postgresql://user:pass@ep-tenant-b.us-east-2.aws.neon.tech/neondb
TENANT_C_CONNECTION_STRING=postgresql://user:pass@ep-tenant-c.us-east-2.aws.neon.tech/neondb

# Neo4j Configuration (Local Docker)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=prototype123

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO

# AI Configuration - UPDATE THESE VALUES
OPENAI_API_KEY=your_openai_api_key_here
GRAPHITI_API_KEY=your_graphiti_api_key_here
EOF

# Setup Nginx configuration
log_info "Configuring Nginx..."
sudo tee /etc/nginx/sites-available/multitenant-rag << 'EOF'
server {
    listen 80;
    server_name _;
    
    # Increase client max body size for file uploads
    client_max_body_size 100M;
    
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
    
    location /docs {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable Nginx site
sudo ln -sf /etc/nginx/sites-available/multitenant-rag /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
sudo systemctl enable nginx

# Setup UFW firewall
log_info "Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

# Create PM2 ecosystem template
log_info "Creating PM2 configuration..."
cat > /home/ubuntu/multitenant-rag/Tenant/ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'multitenant-rag-api',
    script: 'prototype_api.py',
    interpreter: '/home/ubuntu/multitenant-rag/Tenant/venv/bin/python',
    cwd: '/home/ubuntu/multitenant-rag/Tenant',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env_file: '.env.production',
    error_file: '/home/ubuntu/logs/api-error.log',
    out_file: '/home/ubuntu/logs/api-out.log',
    log_file: '/home/ubuntu/logs/api-combined.log',
    time: true
  }]
};
EOF

# Create useful scripts
log_info "Creating utility scripts..."

# Deployment script
cat > /home/ubuntu/deploy.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/multitenant-rag/Tenant

echo "🔄 Updating application..."
git pull origin main

echo "📦 Installing dependencies..."
source venv/bin/activate
pip install -r requirements_prototype.txt

echo "🔄 Restarting application..."
pm2 restart multitenant-rag-api

echo "✅ Deployment complete!"
pm2 status
EOF

# Status check script
cat > /home/ubuntu/status.sh << 'EOF'
#!/bin/bash
echo "🖥️  System Status"
echo "================"
echo "CPU/Memory:"
free -h
echo ""
echo "Disk Usage:"
df -h /
echo ""
echo "🐳 Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "🚀 PM2 Processes:"
pm2 status
echo ""
echo "🌐 Nginx Status:"
sudo systemctl status nginx --no-pager -l
echo ""
echo "🔗 Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "📊 API Health: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/health"
echo "📚 API Docs: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/docs"
EOF

# Make scripts executable
chmod +x /home/ubuntu/deploy.sh
chmod +x /home/ubuntu/status.sh

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "Unable to get public IP")

# Final instructions
log_info "✅ EC2 setup complete!"
echo ""
echo "🔧 Next Steps:"
echo "=============="
echo "1. Clone your repository:"
echo "   cd /home/ubuntu"
echo "   git clone <your-repo-url> multitenant-rag"
echo ""
echo "2. Setup Python environment:"
echo "   cd multitenant-rag/Tenant"
echo "   python3.11 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements_prototype.txt"
echo ""
echo "3. Configure environment:"
echo "   cp .env.production.template .env.production"
echo "   nano .env.production  # Edit with your actual values"
echo ""
echo "4. Create Neon projects:"
echo "   neonctl auth"
echo "   # Create projects and update .env.production with connection strings"
echo ""
echo "5. Start the application:"
echo "   pm2 start ecosystem.config.js"
echo "   pm2 save"
echo "   pm2 startup"
echo ""
echo "🌐 Your server will be available at:"
echo "   http://$PUBLIC_IP"
echo "   API Docs: http://$PUBLIC_IP/docs"
echo "   Health Check: http://$PUBLIC_IP/health"
echo ""
echo "🛠️  Useful commands:"
echo "   ./status.sh    - Check system status"
echo "   ./deploy.sh    - Deploy updates"
echo "   pm2 logs       - View application logs"
echo "   docker logs neo4j-prototype  - View Neo4j logs"
echo ""
echo "🔒 Security Notes:"
echo "   - Change default Neo4j password in production"
echo "   - Consider setting up SSL certificates"
echo "   - Monitor logs for security issues"

log_info "Setup script completed successfully!"
