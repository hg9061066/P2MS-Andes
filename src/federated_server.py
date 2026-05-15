import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import P2MSAndesNet

class FederatedOrchestrator:
    def __init__(self, data_path="data/processed", rounds=50, local_epochs=5, lr=0.001):
        self.data_path = data_path
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load all nodes from the processed data directory
        self.nodes = {}
        for file in os.listdir(data_path):
            if file.endswith(".npz"):
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
            raise FileNotFoundError(f"No .npz files found in {data_path}. Run Phase I first.")
            
        self.total_samples = sum(node['n_k'] for node in self.nodes.values())
        self.global_model = P2MSAndesNet().to(self.device)

    def train_local(self, node_loader):
        """Standard local training step on a node."""
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
        """
        FedAvg: Weighted average of local model parameters.
        Includes dtype restoration for BatchNorm counters (Long tensors).
        """
        # Start with a deep copy of the first node's weights to get the model structure
        avg_weights = copy.deepcopy(local_weights_list[0][0])
        
        for key in avg_weights.keys():
            # 1. Initialize an accumulator as Float to allow precise weighted multiplication
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
            
            # 2. Sum the weighted contributions from every node
            for weights, n_k in local_weights_list:
                weight_factor = n_k / self.total_samples
                avg_weights[key] += weights[key].float() * weight_factor
            
            # 3. Type Restoration: Cast back to original dtype (e.g., Long for batch counters)
            # This prevents the 'Float can't be cast to Long' RuntimeError
            original_dtype = local_weights_list[0][0][key].dtype
            if original_dtype == torch.long:
                avg_weights[key] = torch.round(avg_weights[key]).to(torch.long)
            else:
                avg_weights[key] = avg_weights[key].to(original_dtype)
                
        return avg_weights

    def run_federation(self):
        print("="*70)
        print(f"P²MS-Andes Federated Training | {len(self.nodes)} Nodes | Device: {self.device}")
        print("="*70)
        history = []

        for r in range(self.rounds):
            local_updates = []
            
            # Local Training Phase
            for name, node in self.nodes.items():
                w_local = self.train_local(node['loader'])
                local_updates.append((w_local, node['n_k']))
            
            # Aggregation Phase (Global weight update)
            new_global_weights = self.aggregate(local_updates)
            self.global_model.load_state_dict(new_global_weights)

            # Global Evaluation Phase
            val_loss = self.evaluate_global()
            history.append(val_loss)
            
            if r % 5 == 0 or r == self.rounds - 1:
                print(f"Round {r:02d}/{self.rounds} | Global Aggregated Loss: {val_loss:.4f}")
        
        # Save the global checkpoint for Phase III (XAI)
        save_path = "data/processed/global_model.pth"
        torch.save(self.global_model.state_dict(), save_path)
        print("="*70)
        print(f"Training Complete. Global model saved to: {save_path}")
        print("="*70)

    def evaluate_global(self):
        """Weighted global loss across all decentralized nodes."""
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