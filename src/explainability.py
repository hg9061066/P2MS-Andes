import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from captum.attr import IntegratedGradients
from src.model import P2MSAndesNet

def run_explainability_suite(node_name="Node_1_Neuquen"):
    print(f"--- Launching Explainability Suite for {node_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the Global Model
    model = P2MSAndesNet()
    model.load_state_dict(torch.load("data/processed/global_model.pth"))
    model.to(device)
    model.eval()

    # 2. Load Node Data for interpretation
    data = np.load(f"data/processed/{node_name}.npz")
    X = torch.tensor(data['features'], dtype=torch.float32).to(device)
    
    # 3. ENVIRONMENTAL INTERPRETABILITY (SHAP)
    # Focus on the last 9 features (3 variables * 3 lags)
    print("[XAI] Computing SHAP values for environmental lags...")
    
    # We use a background distribution (the first 50 samples) to explain a 'High Risk' sample
    background = X[:50]
    test_sample = X[X.shape[0]//2 : X.shape[0]//2 + 20] # Middle 20 samples
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_sample)

    # Environmental feature names for the plot
    env_names = [
        "Temp (t)", "Precip (t)", "Solar (t)",
        "Temp (t-4w)", "Precip (t-4w)", "Solar (t-4w)",
        "Temp (t-8w)", "Precip (t-8w)", "Solar (t-8w)"
    ]

    # Extract SHAP for the environmental slice
    # Note: SHAP returns a list of arrays (one per class). We look at Class 2 (High Risk)
    shap_env = shap_values[2][:, 1024:] 
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_env, feature_names=env_names, show=False)
    plt.title(f"Environmental Drivers of High Risk: {node_name}")
    plt.savefig("data/processed/env_explanation.png")
    print("-> Environmental Saliency Map saved to: data/processed/env_explanation.png")

    # 4. GENOMIC INTERPRETABILITY (INTEGRATED GRADIENTS)
    print("[XAI] Computing Integrated Gradients for genomic motifs...")
    ig = IntegratedGradients(model)
    
    # Explain why a specific sample was predicted as Class 2 (High Risk)
    target_class = 2
    attributions, delta = ig.attribute(test_sample, target=target_class, return_convergence_delta=True)
    
    # Focus on the 1024 genomic features
    genomic_attr = attributions[:, :1024].cpu().detach().numpy()
    mean_genomic_attr = np.abs(genomic_attr).mean(axis=0)

    plt.figure(figsize=(12, 4))
    plt.plot(mean_genomic_attr, color='teal', alpha=0.7)
    plt.fill_between(range(1024), mean_genomic_attr, color='teal', alpha=0.3)
    plt.title(f"Genomic Saliency Profile: Motif Importance (k-mer bins 1-1024)")
    plt.xlabel("K-mer Feature Bin")
    plt.ylabel("Importance Weight")
    plt.savefig("data/processed/genomic_explanation.png")
    print("-> Genomic Saliency Map saved to: data/processed/genomic_explanation.png")

if __name__ == "__main__":
    run_explainability_suite()