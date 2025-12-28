# Step-by-Step AWS EC2 Deployment Guide

## ✅ Prerequisites
- AWS Account
- AWS CLI installed (optional)
- Your code ready in: `C:\Users\jaswa\Downloads\NetworkSecurity`

## 🚀 STEP 1: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Configure Instance:**
   - **Name:** network-security-api
   - **AMI:** Ubuntu Server 22.04 LTS (Free Tier)
   - **Instance Type:** t2.medium (recommended) or t2.small
   - **Key Pair:** Create new or select existing (.pem file)
   - **Storage:** 20 GB gp3

3. **Security Group Settings (IMPORTANT!):**
   Click "Edit" next to Network Settings:
   - ✅ SSH (Port 22) - Source: My IP
   - ✅ Custom TCP (Port 8080) - Source: 0.0.0.0/0
   - ✅ HTTP (Port 80) - Source: 0.0.0.0/0 (optional)

4. **Click "Launch Instance"**

5. **Wait 2-3 minutes** for instance to start

6. **Note down:**
   - Public IPv4 Address (e.g., 3.85.123.45)
   - Key pair file location (e.g., my-key.pem)

---

## 🔐 STEP 2: Connect to EC2

### On Windows (PowerShell):

```powershell
# Navigate to your key file location
cd C:\Users\jaswa\Downloads

# Set correct permissions (if needed)
icacls my-key.pem /inheritance:r
icacls my-key.pem /grant:r "$($env:USERNAME):(R)"

# Connect to EC2
ssh -i "my-key.pem" ubuntu@YOUR-EC2-PUBLIC-IP
```

**Example:**
```powershell
ssh -i "my-key.pem" ubuntu@3.85.123.45
```

Type "yes" when prompted to continue connecting.

---

## 📦 STEP 3: Upload Your Code

### Option A: Using SCP (Recommended)

**On your LOCAL Windows PowerShell:**

```powershell
cd C:\Users\jaswa\Downloads

scp -i "my-key.pem" -r NetworkSecurity ubuntu@YOUR-EC2-PUBLIC-IP:~/
```

### Option B: Using Git

**On EC2 terminal:**

```bash
cd ~
git clone YOUR-GITHUB-REPO-URL network-security-app
cd network-security-app
```

---

## 🛠️ STEP 4: Install and Run

**On EC2 terminal:**

```bash
# Navigate to app directory
cd ~/NetworkSecurity  # or ~/network-security-app if using git

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install dependencies
pip3 install -r requirements.txt

# Create necessary directories
mkdir -p final_model prediction_output Artifacts logs

# Run the application
python3 app.py
```

**Your API is now running!** 🎉

Access at: `http://YOUR-EC2-PUBLIC-IP:8080/docs`

---

## 🔄 STEP 5: Keep App Running (Even After Closing SSH)

Currently, if you close SSH, the app stops. Let's fix that:

```bash
# Install PM2 (Process Manager)
sudo apt install npm -y
sudo npm install -g pm2

# Start app with PM2
pm2 start app.py --interpreter python3 --name network-security-api

# Save PM2 configuration
pm2 save

# Enable PM2 to start on boot
pm2 startup
# Copy and run the command it shows

# Check status
pm2 status
```

**PM2 Commands:**
```bash
pm2 status          # Check if app is running
pm2 logs            # View logs
pm2 restart all     # Restart app
pm2 stop all        # Stop app
pm2 delete all      # Remove app from PM2
```

---

## ✅ STEP 6: Verify Deployment

1. **Open browser:**
   ```
   http://YOUR-EC2-PUBLIC-IP:8080/docs
   ```

2. **Test endpoints:**
   - GET /train - Start training
   - POST /predict - Upload CSV

3. **Check logs on EC2:**
   ```bash
   pm2 logs
   # or
   tail -f logs/app.log
   ```

---

## 🛡️ STEP 7: Secure Your API (Optional but Recommended)

### Add HTTPS with Let's Encrypt:

```bash
# Install Nginx
sudo apt install nginx -y

# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate (requires domain name)
sudo certbot --nginx -d yourdomain.com

# Configure Nginx to proxy to your app
sudo nano /etc/nginx/sites-available/default
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name YOUR-EC2-PUBLIC-IP;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Restart Nginx
sudo systemctl restart nginx
```

---

## 📊 Monitoring

```bash
# CPU and Memory usage
htop  # Install: sudo apt install htop

# Disk space
df -h

# Check app status
pm2 monit

# View system logs
journalctl -u nginx -f
```

---

## 🐛 Troubleshooting

### App not starting?
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check dependencies
pip3 list

# Run manually to see errors
python3 app.py
```

### Can't connect from browser?
```bash
# Check if port 8080 is open
sudo netstat -tulpn | grep 8080

# Check EC2 Security Group
# Make sure port 8080 is allowed in AWS Console
```

### Out of memory?
```bash
# Check memory
free -h

# If low, upgrade to t2.medium instance
# Or add swap space:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## 💰 Cost Estimation

- **t2.small:** ~$15/month
- **t2.medium:** ~$30/month
- **Data transfer:** ~$0.09/GB

**Free Tier:** First 12 months get 750 hours/month of t2.micro FREE!

---

## 🎯 Quick Commands Reference

```bash
# Start app
pm2 start app.py --interpreter python3 --name network-security-api

# Stop app
pm2 stop network-security-api

# Restart app
pm2 restart network-security-api

# View logs
pm2 logs

# Update code
cd ~/NetworkSecurity
git pull  # if using git
pm2 restart all

# Check app status
pm2 status

# View resource usage
pm2 monit
```

---

## ✅ Success Checklist

- [ ] EC2 instance running
- [ ] Security group allows port 8080
- [ ] SSH connection works
- [ ] Code uploaded to EC2
- [ ] Dependencies installed
- [ ] App running with PM2
- [ ] Can access http://YOUR-IP:8080/docs
- [ ] Training endpoint works
- [ ] Prediction endpoint works

---

## 🚀 Your API is Live!

**API URL:** `http://YOUR-EC2-PUBLIC-IP:8080`
**Docs:** `http://YOUR-EC2-PUBLIC-IP:8080/docs`

**Share with users:**
```
API Endpoint: http://YOUR-EC2-PUBLIC-IP:8080
Documentation: http://YOUR-EC2-PUBLIC-IP:8080/docs
```

---

Need help? Your app logs are at:
- PM2 logs: `pm2 logs`
- App logs: `~/NetworkSecurity/logs/`
