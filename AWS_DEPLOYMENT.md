# AWS Deployment Guide for Network Security API

## Option 1: AWS EC2 (Recommended for beginners)

### Step 1: Launch EC2 Instance
1. Go to AWS Console → EC2
2. Click "Launch Instance"
3. Choose: **Ubuntu 22.04 LTS**
4. Instance type: **t2.medium** (or t2.small for testing)
5. Create/Select key pair (.pem file)
6. Security Group: Allow ports **22 (SSH)** and **8080 (Custom TCP)**
7. Launch instance

### Step 2: Connect to EC2
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

### Step 3: Setup on EC2
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Install Docker (optional, for containerized deployment)
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Create app directory
mkdir ~/network-security
cd ~/network-security
```

### Step 4: Upload Your Code
On your local machine:
```bash
scp -i "your-key.pem" -r C:\Users\jaswa\Downloads\NetworkSecurity ubuntu@your-ec2-public-ip:~/network-security/
```

### Step 5: Run the Application
```bash
cd ~/network-security

# Install dependencies
pip3 install -r requirements.txt

# Run the app
python3 app.py
```

### Step 6: Access Your API
- `http://your-ec2-public-ip:8080/docs`

---

## Option 2: AWS ECR + ECS (Docker Container)

### Step 1: Build Docker Image Locally
```bash
cd C:\Users\jaswa\Downloads\NetworkSecurity
docker build -t network-security-api .
```

### Step 2: Create ECR Repository
```bash
aws ecr create-repository --repository-name network-security-api --region us-east-1
```

### Step 3: Push to ECR
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag network-security-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/network-security-api:latest

# Push
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/network-security-api:latest
```

### Step 4: Deploy to ECS
1. Go to AWS ECS Console
2. Create Cluster (Fargate)
3. Create Task Definition:
   - Container: Your ECR image
   - Port: 8080
   - Memory: 2GB
   - CPU: 1 vCPU
4. Create Service
5. Configure Load Balancer (optional)

---

## Option 3: AWS App Runner (Easiest!)

### Step 1: Push code to GitHub
```bash
git init
git add .
git commit -m "Deploy to AWS"
git push origin main
```

### Step 2: AWS App Runner
1. Go to AWS App Runner Console
2. Create Service
3. Connect GitHub repository
4. Configure:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Port: 8080
5. Deploy!

App Runner will auto-deploy and give you a URL.

---

## Quick Test on EC2 (Copy-Paste Ready)

After connecting to EC2:
```bash
# One-line setup
sudo apt update && sudo apt install python3-pip -y && mkdir ~/app && cd ~/app

# Then upload your code and run:
pip3 install -r requirements.txt && python3 app.py
```

---

## Important Notes
- ✅ Your MongoDB connection will work (it's cloud-based)
- ✅ MLflow DagHub tracking will work
- ⚠️ Make sure final_model/ directory exists with your trained models
- ⚠️ Update security group to allow port 8080

Which option would you like to proceed with?
