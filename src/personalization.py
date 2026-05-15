import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.model import P2MSAndesNet
from sklearn.metrics import f1_score

def rescue_outlier_node(node_name="Node_4_Aysen", epochs=5):
    print(f"--- Starting Personalized Fine-Tuning for {node_name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the Global Model (The "Pre-trained" Knowledge)
    model = P2MSAndesNet().to(device)
    model.load_state_dict(torch.load("data/processed/global_model.pth"))

    # 2. Load Local Data
    data = np.load(f"data/processed/{node_name}.npz")
    X = torch.tensor(data['features'], dtype=torch.float32).to(device)
    y = torch.tensor(data['labels'], dtype=torch.long).to(device)

    # 3. Fine-Tuning Loop (Low Learning Rate to preserve global knowledge)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"Fine-tuning Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    # 4. Final Evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X)
        final_pred = torch.argmax(final_outputs, dim=1)
        p_f1 = f1_score(y.cpu(), final_pred.cpu(), average='macro')
    
    print(f"\nFinal Personalized F1-Score for {node_name}: {p_f1:.4f}")
    print("--- Personalization Complete ---")

if __name__ == "__main__":
    rescue_outlier_node()