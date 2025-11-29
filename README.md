# MLOps Kubeflow Assignment

A complete Machine Learning Operations (MLOps) pipeline for Boston Housing price prediction using Kubeflow Pipelines, DVC, and Jenkins/GitHub Actions.

## 📋 Project Overview

This project demonstrates an end-to-end MLOps pipeline that includes:
- **Data Versioning** with DVC
- **Pipeline Orchestration** with Kubeflow Pipelines
- **Model Training** using Random Forest Regressor
- **Continuous Integration** with Jenkins and GitHub Actions
- **Containerization** with Docker

### ML Problem
Predict median house prices in Boston using the Boston Housing dataset with features like crime rate, number of rooms, property tax rate, etc.

## 🏗️ Project Structure

```
mlops-kubeflow-assignment/
├── data/                          # Data directory (tracked by DVC)
│   └── raw_data.csv              # Boston Housing dataset
├── src/                          # Source code
│   ├── pipeline_components.py    # Kubeflow component definitions
│   └── model_training.py         # Standalone training script
├── components/                   # Compiled Kubeflow components (YAML)
├── models/                       # Trained model artifacts
├── metrics/                      # Model evaluation metrics
├── .github/
│   └── workflows/
│       └── main.yml               # GitHub Actions workflow
├── pipeline.py                   # Main Kubeflow pipeline definition
├── requirements.txt              # Python dependencies
├── Jenkinsfile                   # Jenkins CI/CD pipeline
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10.1
- Git
- Docker
- Minikube
- kubectl
- Jenkins (optional, for CI/CD)

### 1. Clone the Repository

```bash
git clone https://github.com/MuhAhmad312/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize DVC

```bash
# Initialize DVC
dvc init

# Set up DVC remote storage (example with local directory)
mkdir -p /tmp/dvc-storage
dvc remote add -d myremote /tmp/dvc-storage

# Or use cloud storage (S3 example)
# dvc remote add -d myremote s3://my-bucket/dvc-storage
# dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
# dvc remote modify myremote secret_access_key YOUR_SECRET_KEY

# Add data to DVC tracking
dvc add data/raw_data.csv

# Push data to remote
dvc push

# Commit DVC files to Git
git add data/raw_data.csv.dvc data/.gitignore .dvc/config
git commit -m "Add dataset with DVC tracking"
```

### 4. Set Up Minikube and Kubeflow Pipelines

```bash
# Start Minikube
minikube start --cpus 4 --memory 8192 --disk-size=40g

# Deploy Kubeflow Pipelines (standalone)
export PIPELINE_VERSION=2.0.1
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=$PIPELINE_VERSION"

# Wait for all pods to be ready
kubectl wait --for=condition=ready --timeout=300s pods --all -n kubeflow

# Access Kubeflow Pipelines UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Access the UI at: http://localhost:8080

### 5. Download Dataset

The pipeline automatically downloads the Boston Housing dataset from:
```
https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv
```

Alternatively, you can manually download it:

```bash
# Create data directory
mkdir -p data

# Download dataset
curl -o data/raw_data.csv https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv
```

## 🔄 Pipeline Walkthrough

### Pipeline Components

The ML pipeline consists of four main components:

1. **Data Extraction Component**
   - Fetches the Boston Housing dataset
   - Validates data integrity
   - Outputs: Dataset artifact

2. **Data Preprocessing Component**
   - Handles missing values
   - Scales features using StandardScaler
   - Splits data into train/test sets (80/20)
   - Outputs: Training and test datasets

3. **Model Training Component**
   - Trains Random Forest Regressor
   - Hyperparameters: n_estimators=100, max_depth=10
   - Outputs: Trained model artifact

4. **Model Evaluation Component**
   - Evaluates model on test data
   - Calculates metrics: R², MSE, RMSE, MAE
   - Outputs: Metrics JSON file

### Compile the Pipeline

```bash
# Compile pipeline to YAML
python pipeline.py

# This generates pipeline.yaml
```

### Run the Pipeline on Kubeflow

#### Option 1: Using Kubeflow UI

1. Open Kubeflow Pipelines UI at http://localhost:8080
2. Click "Upload Pipeline"
3. Upload `pipeline.yaml`
4. Create a new run
5. Configure parameters (optional):
   - `data_url`: Dataset URL
   - `test_size`: Test split ratio (default: 0.2)
   - `n_estimators`: Number of trees (default: 100)
   - `max_depth`: Tree depth (default: 10)
   - `random_state`: Random seed (default: 42)
6. Click "Start"

#### Option 2: Using Python SDK

```python
import kfp

# Connect to Kubeflow Pipelines
client = kfp.Client(host='http://localhost:8070')

# Upload and run pipeline
client.create_run_from_pipeline_package(
    'pipeline.yaml',
    arguments={
        'data_url': 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',
        'test_size': 0.2,
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
)
```

### Local Training (Without Kubeflow)

For local development and testing:

```bash
# Run standalone training script
python src/model_training.py
```

This will:
- Load data from `data/raw_data.csv`
- Train the model
- Save model to `models/random_forest_model.pkl`
- Save metrics to `metrics/model_metrics.json`

## 🔧 Continuous Integration

### Jenkins Setup

1. **Install Jenkins** and required plugins:
   - Git plugin
   - Pipeline plugin
   - Python plugin (optional)

2. **Create Pipeline Job**:
   - New Item → Pipeline
   - Configure Git repository URL
   - Set branch to `main`
   - Pipeline script from SCM
   - Script path: `Jenkinsfile`

3. **Run Pipeline**:
   - Click "Build Now"
   - View console output

The Jenkins pipeline includes three stages:
- **Environment Setup**: Install dependencies
- **Pipeline Compilation**: Compile Kubeflow pipeline
- **Code Quality Check**: Validate Python syntax

### GitHub Actions

GitHub Actions automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual workflow dispatch

View workflow runs in the "Actions" tab of your GitHub repository.

## 📊 Model Performance

Expected metrics on Boston Housing dataset:
- **R² Score**: ~0.85-0.90
- **RMSE**: ~3.5-4.5
- **MAE**: ~2.5-3.5

## 🐳 Docker Usage

### Build Docker Image

```bash
docker build -t mlops-pipeline:latest .
```

### Run Container

```bash
docker run -it mlops-pipeline:latest python src/model_training.py
```

## 📁 DVC Commands Reference

```bash
# Pull data from remote
dvc pull

# Add new data file
dvc add data/new_data.csv

# Push data to remote
dvc push

# Check DVC status
dvc status

# Show data pipeline
dvc dag
```

## 🔍 Troubleshooting

### Kubeflow Pipelines Issues

**Problem**: Pods stuck in pending state
```bash
# Check pod status
kubectl get pods -n kubeflow

# Describe pod for details
kubectl describe pod <pod-name> -n kubeflow

# Check logs
kubectl logs <pod-name> -n kubeflow
```

**Problem**: Cannot access UI
```bash
# Verify port forwarding
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80

# Check service status
kubectl get svc -n kubeflow
```

### DVC Issues

**Problem**: DVC remote not configured
```bash
# List remotes
dvc remote list

# Add remote
dvc remote add -d myremote /path/to/storage
```

### Pipeline Compilation Issues

**Problem**: Import errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify Python version
python --version  # Should be 3.10.1
```

## 📚 Additional Resources

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [DVC Documentation](https://dvc.org/doc)
- [Jenkins Pipeline Documentation](https://www.jenkins.io/doc/book/pipeline/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📝 License

This project is for educational purposes as part of the MLOps assignment.

## 👥 Contributors

- Your Name - Initial work

## 🙏 Acknowledgments

- Boston Housing dataset from UCI Machine Learning Repository
- Kubeflow community
- DVC team
