#!/bin/bash

# AWS EC2 Deployment Script for Network Security API
# Run this script after connecting to your EC2 instance

echo "🚀 Starting Network Security API Deployment..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
echo "🐍 Installing Python..."
sudo apt install python3-pip python3-venv git -y

# Create application directory
echo "📁 Creating application directory..."
mkdir -p ~/network-security-app
cd ~/network-security-app

# Install application dependencies
echo "📥 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Create necessary directories
echo "📂 Creating directories..."
mkdir -p final_model prediction_output Artifacts logs

# Install PM2 (Process Manager) for keeping app running
echo "🔄 Installing PM2 process manager..."
sudo apt install npm -y
sudo npm install -g pm2

# Create PM2 ecosystem file
cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'network-security-api',
    script: 'python3',
    args: 'app.py',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    }
  }]
};
EOF

# Start the application with PM2
echo "✅ Starting application..."
pm2 start ecosystem.config.js
pm2 save
pm2 startup

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "📍 Your API is running on port 8080"
echo "🌐 Access at: http://$(curl -s ifconfig.me):8080/docs"
echo ""
echo "Useful commands:"
echo "  pm2 status          - Check app status"
echo "  pm2 logs            - View logs"
echo "  pm2 restart all     - Restart app"
echo "  pm2 stop all        - Stop app"
