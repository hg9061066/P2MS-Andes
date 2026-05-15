import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.model import P2MSAndesNet
from sklearn.metrics import f1_score, accuracy_score

def train_standalone_local(X, y, epochs=50):
    """Trains a model on a single node's data only."""
    model = P2MSAndesNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simple split for local validation
    split = int(len(X) * 0.8)
    train_X, val_X = X[:split], X[split:]
    train_y, val_y = y[:split], y[split:]

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
    return model

def run_comparison():
    print("--- Starting Comparative Baseline Evaluation ---")
    data_path = "data/processed"
    results = []

    # 1. Load the Global Federated Model
    global_model = P2MSAndesNet()
    global_model.load_state_dict(torch.load(os.path.join(data_path, "global_model.pth")))
    global_model.eval()

    # 2. Iterate through nodes
    for file in os.listdir(data_path):
        if file.endswith(".npz"):
            node_name = file.replace(".npz", "")
            data = np.load(os.path.join(data_path, file))
            X = torch.tensor(data['features'], dtype=torch.float32)
            y = torch.tensor(data['labels'], dtype=torch.long)

            # Train a 'Local-Only' version
            print(f"Training Local-Only model for {node_name}...")
            local_model = train_standalone_local(X, y)
            local_model.eval()

            # Evaluate both on the node's full data
            with torch.no_grad():
                # Global Model Preds
                g_out = global_model(X)
                g_pred = torch.argmax(g_out, dim=1)
                g_f1 = f1_score(y, g_pred, average='macro')

                # Local Model Preds
                l_out = local_model(X)
                l_pred = torch.argmax(l_out, dim=1)
                l_f1 = f1_score(y, l_pred, average='macro')

            results.append({
                "Node": node_name,
                "Samples": len(X),
                "Local_F1": round(l_f1, 4),
                "Federated_F1": round(g_f1, 4),
                "Boost": round(g_f1 - l_f1, 4)
            })

    # 3. Display Final Table
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("TABLE 1: LOCAL VS. FEDERATED PERFORMANCE (F1-SCORE)")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    df_results.to_csv("data/processed/final_metrics.csv", index=False)

if __name__ == "__main__":
    import pandas as pd
    run_comparison()