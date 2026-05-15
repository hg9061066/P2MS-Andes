# P²MS-Andes
**A Privacy-Preserving, Multi-Scale Federated Framework for Explainable Andes Virus Risk Forecasting**

## Project Architecture
* `src/data_loader.py`: Remote NASA POWER API ingestion & multi-scale $k$-mer feature engineering.
* `src/model.py`: Multi-scale decoupled dual-branch Deep Learning Neural Network.
* `src/federated_server.py`: Cross-border decentralized FedAvg optimization loop orchestration.
* `src/explainability.py`: Post-hoc structural interpretability utilizing SHAP and Integrated Gradients.

## Setup Instructions
1. Initialize virtual environment: `python -m venv venv`
2. Activate and install dependencies: `pip install -r requirements.txt`
