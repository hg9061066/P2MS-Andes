import torch
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from src.model import P2MSAndesNet
from src.data_loader import NASAPowerClient, GenomicFeatureExtractor

class P2MSLiveMonitor:
    def __init__(self, lat=-38.95, lon=-68.06): # Default to Neuquén
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lat = lat
        self.lon = lon
        
        # Load the Privacy-Hardened Model
        self.model = P2MSAndesNet()
        self.model.load_state_dict(torch.load("data/processed/global_model.pth"))
        self.model.to(self.device).eval()
        
        self.nasa_client = NASAPowerClient()
        self.genomic_extractor = GenomicFeatureExtractor()

    def fetch_latest_environmental_vector(self, target_date):
        """Pulls the 9-dimensional vector [t, t-4, t-8] for a specific date."""
        # We need a 9-week window to cover the t-8 week lag
        start_date = target_date - timedelta(weeks=10)
        
        print(f"[LIVE] Fetching NASA satellite data for {target_date.date()}...")
        df = self.nasa_client.fetch_nasa_data(
            self.lat, self.lon, 
            start_date.strftime('%Y%m%d'), 
            target_date.strftime('%Y%m%d')
        )
        
        # Scale the data (using same logic as training)
        # Note: In production, you'd use the fitted scaler from Phase I
        df_norm = (df - df.mean()) / (df.std() + 1e-6)
        
        # Extract the specific lags
        e_t = df_norm.iloc[-1].values
        e_4 = df_norm.asof(target_date - timedelta(weeks=4)).values
        e_8 = df_norm.asof(target_date - timedelta(weeks=8)).values
        
        return np.concatenate([e_t, e_4, e_8])

    def run_prediction(self, fasta_sequence):
        # 1. Prepare Target Date (Simulating "Now")
        target_date = datetime.now()
        
        # 2. Get Environmental Data
        try:
            env_vector = self.fetch_latest_environmental_vector(target_date)
        except Exception as e:
            print(f"[ERROR] API Connection failed: {e}")
            return

        # 3. Get Genomic Data
        genomic_vector = self.genomic_extractor.transform_sequences({"current": fasta_sequence})[0]
        
        # 4. Final Fusion (1033-dim)
        input_vector = np.concatenate([genomic_vector, env_vector])
        input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 5. Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            risk_class = torch.argmax(logits, dim=1).item()
        
        classes = {0: "🟢 LOW", 1: "🟡 MODERATE", 2: "🔴 HIGH"}
        print(f"\n" + "="*40)
        print(f"P²MS-ANDES LIVE REPORT: {target_date.strftime('%Y-%m-%d')}")
        print(f"REGION COORDINATES: {self.lat}, {self.lon}")
        print(f"RISK STATUS: {classes[risk_class]}")
        print(f"CONFIDENCE: {probs[0][risk_class]*100:.2f}%")
        print("="*40)

if __name__ == "__main__":
    # Mock Genomic Sequence for testing (a snippet of Andes virus)
    mock_fasta = "ATGGAGGTGGACCCGGATGAGGTTAACGAGTGGCTCCAGAGGAAC" 
    
    monitor = P2MSLiveMonitor()
    monitor.run_prediction(mock_fasta)