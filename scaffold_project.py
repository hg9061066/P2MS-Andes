import os
import subprocess

def create_workspace():
    print("[Workspace] Initializing P²MS-Andes Project Scaffolding...")
    
    # 1. Define Directories
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "tests"
    ]
    
    for folder in directories:
        os.makedirs(folder, exist_ok=True)
        print(f"-> Created directory path: {folder}")

    # 2. Define File Payloads
    # .gitignore template configured for ML/DL data protection
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Environments & OS files
.venv/
venv/
ENV/
.env
.DS_Store
Thumbs.db

# Data & Large Artifact Storage (Never commit local datasets)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep
*.npz
*.pkl
*.csv
*.fasta

# Jupyter Notebook checkpoints
.ipynb_checkpoints/
"""

    requirements_content = """numpy>=1.24.0
pandas>=2.0.0
requests>=2.30.0
scikit-learn>=1.2.0
torch>=2.0.0
captum>=0.6.0
shap>=0.42.0
matplotlib>=3.7.0
"""

    readme_content = """# P²MS-Andes
**A Privacy-Preserving, Multi-Scale Federated Framework for Explainable Andes Virus Risk Forecasting**

## Project Architecture
* `src/data_loader.py`: Remote NASA POWER API ingestion & multi-scale $k$-mer feature engineering.
* `src/model.py`: Multi-scale decoupled dual-branch Deep Learning Neural Network.
* `src/federated_server.py`: Cross-border decentralized FedAvg optimization loop orchestration.
* `src/explainability.py`: Post-hoc structural interpretability utilizing SHAP and Integrated Gradients.

## Setup Instructions
1. Initialize virtual environment: `python -m venv venv`
2. Activate and install dependencies: `pip install -r requirements.txt`
"""

    # Model file template incorporating your decoupled dual-branch architecture
    model_content = """import torch
import torch.nn as nn

class P2MSAndesNet(nn.Module):
    \"\"\"
    Multi-Scale Decoupled Architecture for Andes Virus Risk Forecasting.
    Splits a 1033-dim vector into 1024-dim Genomic and 9-dim Environmental tracks.
    \"\"\"
    def __init__(self, genomic_dim=1024, environmental_dim=9, num_classes=3):
        super(P2MSAndesNet, self).__init__()
        self.genomic_dim = genomic_dim
        self.environmental_dim = environmental_dim
        
        # Genomic Feature Matrix Extraction Track
        self.genomic_branch = nn.Sequential(
            nn.Linear(genomic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Environmental Eco-Climatic Lag Integration Track
        self.env_branch = nn.Sequential(
            nn.Linear(environmental_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Unified Multi-Scale Fusion Head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        g_feat = x[:, :self.genomic_dim]
        e_feat = x[:, self.genomic_dim:]
        
        g_out = self.genomic_branch(g_feat)
        e_out = self.env_branch(e_feat)
        
        fused = torch.cat((g_out, e_out), dim=1)
        return self.classifier(fused)
"""

    # Injecting your verified data loader code directly into the workspace build
    data_loader_content = """import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

class GenomicFeatureExtractor:
    def __init__(self, k_sizes=[3, 5, 7], target_dim=1024):
        self.k_sizes = k_sizes
        self.target_dim = target_dim
        self.hasher = FeatureHasher(n_features=target_dim, input_type='pair')

    def _extract_kmers(self, sequence):
        kmers = []
        sequence = sequence.upper().replace("N", "").replace("-", "")
        for k in self.k_sizes:
            for i in range(len(sequence) - k + 1):
                kmers.append(sequence[i:i+k])
        return kmers

    def transform_sequences(self, fasta_records):
        hashed_features = []
        for seq_id, seq in fasta_records.items():
            kmers = self._extract_kmers(seq)
            counts = {}
            for kmer in kmers:
                counts[kmer] = counts.get(kmer, 0) + 1
            total = sum(counts.values()) if counts else 1
            freq_pairs = [(kmer, count / total) for kmer, count in counts.items()]
            hashed_features.append(freq_pairs)
        return self.hasher.transform(hashed_features).toarray().astype(np.float32)

class ClimateDataIngestor:
    def __init__(self, start_date="20240101", end_date="20250101"):
        self.start_date = start_date
        self.end_date = end_date
        self.params = "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"

    def fetch_regional_climate(self, lat, lon):
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={self.params}&community=AG&longitude={lon}&latitude={lat}&start={self.start_date}&end={self.end_date}&format=JSON"
        response = requests.get(url, timeout=30)
        if response.status_code != 200: raise RuntimeError("API Request Failed")
        param_data = response.json()['properties']['parameter']
        dfs = [pd.DataFrame.from_dict(values, orient="index", columns=[param]) for param, values in param_data.items()]
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.columns = ["temperature", "precipitation", "solar_radiation"]
        return df.resample('W').mean().sort_index()

def construct_multiscale_dataset(genomic_centroid, climate_df, case_labels_df, tau=4):
    scaler = StandardScaler()
    climate_scaled_df = pd.DataFrame(scaler.fit_transform(climate_df), index=climate_df.index, columns=climate_df.columns)
    fused_features, target_labels = [], []
    for current_week in case_labels_df.index:
        t_minus_tau = current_week - pd.Timedelta(weeks=tau)
        t_minus_2tau = current_week - pd.Timedelta(weeks=2*tau)
        if t_minus_2tau >= climate_scaled_df.index.min():
            e_t = climate_scaled_df.asof(current_week).values
            e_tau = climate_scaled_df.asof(t_minus_tau).values
            e_2tau = climate_scaled_df.asof(t_minus_2tau).values
            x_vector = np.concatenate([genomic_centroid, np.concatenate([e_t, e_tau, e_2tau])])
            fused_features.append(x_vector)
            target_labels.append(case_labels_df.loc[current_week, 'risk_class'])
    return np.array(fused_features, dtype=np.float32), np.array(target_labels, dtype=np.int64)
"""

    # 3. Create Files and Inject Payloads
    files_to_write = {
        ".gitignore": gitignore_content,
        "requirements.txt": requirements_content,
        "README.md": readme_content,
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "src/__init__.py": "",
        "tests/__init__.py": "",
        "src/data_loader.py": data_loader_content,
        "src/model.py": model_content,
        "src/federated_server.py": "# Placeholder for Federated Server Aggregation Strategy Execution Loop\n",
        "src/explainability.py": "# Placeholder for Captum (Integrated Gradients) and SHAP pipelines\n"
    }

    for path, content in files_to_write.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.strip() + "\n")
        print(f"-> Generated file: {path}")

    # 4. Automate Local Git Initialization
    print("[Git Synchronization] Running localized workspace repository setup...")
    try:
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit: P2MS-Andes Multi-Scale Architecture Scaffolding Base"], check=True)
        print("-> Git Repository successfully initialized and tracking baseline assets.")
    except Exception as e:
        print(f"[Git Warning] Automation skipped initialization. Ensure Git CLI tool configurations are functional: {e}")

    print("\n[Complete] Run 'pip install -r requirements.txt' inside your target environment to build dependency binaries.")

if __name__ == "__main__":
    create_workspace()