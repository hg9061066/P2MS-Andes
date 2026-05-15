import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import P2MSAndesNet

class FederatedOrchestrator:
    def __init__(self, data_path="data/processed", rounds=50, local_epochs=5, lr=0.001, 
                 dp_sigma=0.1, dp_clip_norm=1.0):
        self.data_path = data_path
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = lr
        
        # Privacy Parameters
        self.dp_sigma = dp_sigma
        self.dp_clip_norm = dp_clip_norm
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Debug: Check if directory exists
        if not os.path.exists(data_path):
            print(f"[DEBUG] ERROR: Directory not found: {os.path.abspath(data_path)}")
            self.nodes = {}
        else:
            self.nodes = {}
            files = [f for f in os.listdir(data_path) if f.endswith(".npz")]
            print(f"[DEBUG] Found {len(files)} data files in {data_path}")
            
            for file in files:
                name = file.replace(".npz", "")
                data = np.load(os.path.join(data_path, file))
                X = torch.tensor(data['features'], dtype=torch.float32)
                y = torch.tensor(data['labels'], dtype=torch.long)
                dataset = TensorDataset(X, y)
                self.nodes[name] = {
                    "loader": DataLoader(dataset, batch_size=32, shuffle=True),
                    "n_k": len(X),
                    "X_val": X, 
                    "y_val": y
                }
        
        if not self.nodes:
            print("[DEBUG] CRITICAL: No nodes loaded. Federation cannot start.")
            return

        self.total_samples = sum(node['n_k'] for node in self.nodes.values())
        self.global_model = P2MSAndesNet().to(self.device)

    def train_local(self, node_loader):
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        optimizer = optim.Adam(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for batch_X, batch_y in node_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = local_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return local_model.state_dict()

    def aggregate(self, local_weights_list):
        """FedAvg with Differential Privacy (Clipping + Noise)."""
        avg_weights = copy.deepcopy(local_weights_list[0][0])
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
            
            for weights, n_k in local_weights_list:
                param_update = weights[key].float()
                # --- DP Step 1: Clipping ---
                l2_norm = torch.norm(param_update)
                clip_factor = min(1.0, self.dp_clip_norm / (l2_norm.item() + 1e-6))
                
                weight_factor = n_k / self.total_samples
                avg_weights[key] += (param_update * clip_factor) * weight_factor
            
            # --- DP Step 2: Adding Gaussian Noise ---
            if "weight" in key or "bias" in key:
                noise = torch.randn_like(avg_weights[key]) * (self.dp_sigma * self.dp_clip_norm)
                avg_weights[key] += noise

            # Restore original dtypes
            original_dtype = local_weights_list[0][0][key].dtype
            if original_dtype == torch.long:
                avg_weights[key] = torch.round(avg_weights[key]).to(torch.long)
            else:
                avg_weights[key] = avg_weights[key].to(original_dtype)
        return avg_weights

    def run_federation(self):
        if not hasattr(self, 'nodes') or not self.nodes:
            print("[DEBUG] Federation aborted: No nodes.")
            return

        print("="*70)
        print(f"P²MS-Andes DP-Federated Training | {len(self.nodes)} Nodes")
        print(f"Privacy: Sigma={self.dp_sigma}, Clip={self.dp_clip_norm}")
        print("="*70)

        for r in range(self.rounds):
            local_updates = []
            for name, node in self.nodes.items():
                w_local = self.train_local(node['loader'])
                local_updates.append((w_local, node['n_k']))
            
            new_global_weights = self.aggregate(local_updates)
            self.global_model.load_state_dict(new_global_weights)

            val_loss = self.evaluate_global()
            if r % 5 == 0 or r == self.rounds - 1:
                print(f"Round {r:02d}/{self.rounds} | Privacy-Hardened Loss: {val_loss:.4f}")
        
        torch.save(self.global_model.state_dict(), "data/processed/global_model.pth")
        print("="*70)
        print("Training Complete. Privacy-Hardened Global model saved.")

    def evaluate_global(self):
        self.global_model.eval()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for node in self.nodes.values():
                X, y = node['X_val'].to(self.device), node['y_val'].to(self.device)
                outputs = self.global_model(X)
                total_loss += criterion(outputs, y).item() * (node['n_k'] / self.total_samples)
        return total_loss

if __name__ == "__main__":
    orchestrator = FederatedOrchestrator()
    orchestrator.run_federation()