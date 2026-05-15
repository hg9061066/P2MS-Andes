import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.model import P2MSAndesNet
from src.data_loader import NASAPowerClient, GenomicFeatureExtractor
import numpy as np

def run_historical_backtest():
    print("--- Starting Phase IV: Historical Backtest (Epuyen 2018) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Parameters for the 2018 Epuyen Outbreak
    lat, lon = -42.18, -71.48
    # Outbreak started in Nov 2018, peaked in Jan 2019
    start_date = datetime(2018, 9, 1) 
    end_date = datetime(2019, 3, 1)
    
    # 2. Initialize Components
    model = P2MSAndesNet()
    model.load_state_dict(torch.load("data/processed/global_model.pth"))
    model.to(device).eval()
    
    nasa_client = NASAPowerClient()
    extractor = GenomicFeatureExtractor()
    
    # Mock Genomic Sequence representing the Epuyen strain (Andes orthohantavirus)
    epuyen_strain = "GAGGAACGCGTAAGGCAGTACCGGTAGCTAGCTAGCTAGCTAGCTA" 
    genomic_vector = extractor.transform_sequences({"epuyen": epuyen_strain})[0]

    # 3. Fetch Historical Climate Data
    print(f"[BACKTEST] Fetching NASA data for late 2018...")
    climate_df = nasa_client.fetch_nasa_data(
        lat, lon, 
        (start_date - timedelta(weeks=10)).strftime('%Y%m%d'), 
        end_date.strftime('%Y%m%d')
    )
    
    # Basic normalization consistent with training
    climate_norm = (climate_df - climate_df.mean()) / (climate_df.std() + 1e-6)

    # 4. Rolling Prediction Loop
    dates, risk_scores = [], []
    current_ptr = start_date
    
    while current_ptr <= end_date:
        try:
            # Extract t, t-4, t-8 lags
            e_t = climate_norm.asof(current_ptr).values
            e_4 = climate_norm.asof(current_ptr - timedelta(weeks=4)).values
            e_8 = climate_norm.asof(current_ptr - timedelta(weeks=8)).values
            
            env_vector = np.concatenate([e_t, e_4, e_8])
            fused_vector = np.concatenate([genomic_vector, env_vector])
            input_tensor = torch.tensor(fused_vector, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                # Weighted risk score for visualization
                risk_val = (probs[0][1] * 1 + probs[0][2] * 2).item()
                
            dates.append(current_ptr)
            risk_scores.append(risk_val)
        except:
            pass
        current_ptr += timedelta(weeks=1)

    # 5. Visualization: The "Lead Time" Graph
    plt.figure(figsize=(12, 5))
    plt.plot(dates, risk_scores, color='crimson', linewidth=2, label='Predicted Risk Level')
    plt.axvspan(datetime(2018, 11, 15), datetime(2019, 2, 1), color='gray', alpha=0.2, label='Actual Outbreak Duration')
    plt.axhline(y=1.5, color='orange', linestyle='--', label='Warning Threshold')
    
    plt.title("Historical Backtest: 2018 Epuyen Outbreak Detection")
    plt.ylabel("Computed Risk Index")
    plt.xlabel("Timeline")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("data/processed/backtest_results.png")
    print("✅ Backtest complete. Lead-time analysis saved to: data/processed/backtest_results.png")

if __name__ == "__main__":
    run_historical_backtest()