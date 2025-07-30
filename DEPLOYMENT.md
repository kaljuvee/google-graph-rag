# Deployment Guide

This document provides detailed instructions for deploying the Google Graph RAG MVP application in various environments.

## 🚀 Deployment Options

### 1. Local Development Deployment

#### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM recommended

#### Quick Start
```bash
git clone https://github.com/kaljuvee/google-graph-rag.git
cd google-graph-rag
pip install -r requirements.txt
streamlit run Home.py
```

### 2. Docker Deployment

#### Build and Run
```bash
# Build the Docker image
docker build -t google-graph-rag .

# Run the container
docker run -p 8501:8501 \
  -e GOOGLE_KG_API_KEY="your_api_key" \
  -e GOOGLE_CLOUD_PROJECT="your_project" \
  google-graph-rag
```

#### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - GOOGLE_KG_API_KEY=${GOOGLE_KG_API_KEY}
      - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
    volumes:
      - ./data:/app/data
      - ./test-results:/app/test-results
```

### 3. Cloud Deployment

#### Google Cloud Run
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/google-graph-rag
gcloud run deploy --image gcr.io/PROJECT_ID/google-graph-rag --platform managed
```

#### Heroku Deployment
```bash
# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set GOOGLE_KG_API_KEY="your_api_key"
heroku config:set GOOGLE_CLOUD_PROJECT="your_project"

# Deploy
git push heroku main
```

#### AWS EC2 Deployment
```bash
# Launch EC2 instance with Ubuntu
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/kaljuvee/google-graph-rag.git
cd google-graph-rag
pip3 install -r requirements.txt

# Run with nohup for persistent execution
nohup streamlit run Home.py --server.port 8501 --server.address 0.0.0.0 &
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required for Google services
export GOOGLE_KG_API_KEY="your_google_knowledge_graph_api_key"
export GOOGLE_CLOUD_PROJECT="your_gcp_project_id"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Optional configurations
export STREAMLIT_SERVER_PORT="8501"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_neo4j_password"
```

### Google Cloud Setup

1. **Enable APIs**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable discoveryengine.googleapis.com
   gcloud services enable kgsearch.googleapis.com
   ```

2. **Create Service Account**
   ```bash
   gcloud iam service-accounts create rag-service-account
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:rag-service-account@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Generate API Key**
   - Go to Google Cloud Console
   - Navigate to APIs & Services > Credentials
   - Create API Key for Knowledge Graph Search API

## 🐳 Docker Configuration

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### .dockerignore
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis
.venv
venv/
.env
```

## 🔒 Security Considerations

### API Key Management
- Never commit API keys to version control
- Use environment variables or secret management services
- Rotate API keys regularly
- Implement rate limiting

### Network Security
```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw allow 22/tcp
sudo ufw allow 8501/tcp
sudo ufw enable

# For production, use reverse proxy
sudo apt install nginx
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## 📊 Monitoring and Logging

### Application Monitoring
```python
# Add to your Streamlit app
import logging
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Monitor performance
@st.cache_data
def monitor_performance():
    # Implementation for performance monitoring
    pass
```

### Health Checks
```python
# health_check.py
import requests
import sys

def check_health():
    try:
        response = requests.get('http://localhost:8501/_stcore/health')
        if response.status_code == 200:
            print("Application is healthy")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    if not check_health():
        sys.exit(1)
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        cd tests && python run_all_tests.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Cloud Run
      run: |
        # Add deployment commands here
```

## 🚨 Troubleshooting

### Common Deployment Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   # Kill the process
   kill -9 PID
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   htop
   # Increase swap space if needed
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   chmod +x run_app.sh
   # Fix directory permissions
   chmod -R 755 /path/to/app
   ```

4. **Google Cloud Authentication**
   ```bash
   # Check authentication
   gcloud auth list
   # Re-authenticate if needed
   gcloud auth application-default login
   ```

### Performance Optimization

1. **Streamlit Configuration**
   ```toml
   # .streamlit/config.toml
   [server]
   port = 8501
   address = "0.0.0.0"
   maxUploadSize = 200
   maxMessageSize = 200

   [browser]
   gatherUsageStats = false

   [theme]
   primaryColor = "#FF6B6B"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   ```

2. **Caching Strategy**
   ```python
   import streamlit as st

   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_data():
       # Expensive data loading operation
       pass

   @st.cache_resource
   def init_model():
       # Initialize ML models
       pass
   ```

## 📋 Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Google Cloud APIs enabled
- [ ] Service accounts created
- [ ] API keys generated and secured
- [ ] Firewall rules configured
- [ ] SSL certificates installed (production)
- [ ] Monitoring and logging set up
- [ ] Health checks implemented
- [ ] Backup strategy defined
- [ ] CI/CD pipeline configured
- [ ] Documentation updated
- [ ] Performance testing completed
- [ ] Security review conducted

## 🔄 Maintenance

### Regular Tasks
- Update dependencies monthly
- Rotate API keys quarterly
- Review logs weekly
- Monitor performance daily
- Backup data regularly

### Scaling Considerations
- Implement load balancing for high traffic
- Use container orchestration (Kubernetes)
- Consider database clustering for Neo4j
- Implement caching layers (Redis)
- Monitor and optimize resource usage

---

For additional support or questions about deployment, please refer to the main README.md or create an issue on GitHub.

