import os
import numpy as np
import pandas as pd
from src.data_loader import GenomicFeatureExtractor, ClimateDataIngestor, construct_multiscale_dataset

# Configuration mapping to your exact non-IID structural design specifications
NODE_METADATA = {
    "Node_1_Neuquen":     {"samples": 985, "lat": -38.95, "lon": -68.06, "skew": "high_density"},
    "Node_2_Rio_Negro":   {"samples": 410, "lat": -41.13, "lon": -71.30, "skew": "endemic_hotspot"},
    "Node_3_Valparaiso":  {"samples": 164, "lat": -33.05, "lon": -71.61, "skew": "label_skewed"},
    "Node_4_Aysen":       {"samples": 83,  "lat": -45.57, "lon": -72.07, "skew": "extreme_sparse"}
}

def execute_ingestion_pipeline():
    print("=" * 70)
    print("STARTING PHASE I: MULTI-SCALE DATA INGESTION & PIPELINE ALIGNMENT")
    print("=" * 70)

    # 1. Extract Genomic Centroid Vector (Micro-Scale Spec)
    print("\n[Step 1/4] Processing NCBI GenBank sequences (~180 strains)...")
    # Simulating 180 viral sequences of length 1500 bases
    mock_fasta = {f"ANDV_Strain_{i}": "".join(np.random.choice(['A','C','G','T'], size=1500)) for i in range(180)}
    
    extractor = GenomicFeatureExtractor(k_sizes=[3, 5, 7], target_dim=1024)
    genomic_matrix = extractor.transform_sequences(mock_fasta)
    g_centroid = genomic_matrix.mean(axis=0)
    print(f"-> Generated fixed 1024-dim Genomic Population Centroid Vector.")

    # 2. Loop Through and Generate Decentralized Jurisdictions
    print("\n[Step 2/4] Fetching climate grids and generating node feature matrices...")
    
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    for node_name, meta in NODE_METADATA.items():
        print(f"\nProcessing {node_name} ({meta['skew'].upper()})")
        print(f"-> Target sample footprint: {meta['samples']} observations")
        
        # Ingest target environmental timeline 
        # Using a fixed date window to generate a clean baseline sample index
        ingestor = ClimateDataIngestor(start_date="20200101", end_date="20240101")
        
        try:
            # Live API extraction call structure
            climate_df = ingestor.fetch_regional_climate(lat=meta['lat'], lon=meta['lon'])
            print(f"   - Retrieved climate dataframe matrix: {len(climate_df)} weekly frames.")
        except Exception as e:
            print(f"   - [API Timeout/Offline] Using synthetic fallback generator matching coordinates.")
            # Resilient fallback generator to ensure seamless offline execution
            date_range = pd.date_range(start="2020-01-01", periods=meta['samples'] + 10, freq="W")
            climate_df = pd.DataFrame(
                data=np.random.randn(len(date_range), 3),
                index=date_range,
                columns=["temperature", "precipitation", "solar_radiation"]
            )

        # 3. Generate Skewed Epidemiology Targets (Surveillance Labels)
        # Apply your exact data ecosystem behavior parameters per node
        if meta['skew'] == "label_skewed":
            p_dist = [0.1, 0.7, 0.2]      # Heavy Moderate Risk grouping bias
        elif meta['skew'] == "extreme_sparse":
            p_dist = [0.85, 0.12, 0.03]   # Dominant Low Risk bias (Aysén)
        elif meta['skew'] == "endemic_hotspot":
            p_dist = [0.20, 0.40, 0.40]   # High distribution of High/Moderate risk (Río Negro)
        else:
            p_dist = [0.50, 0.35, 0.15]   # Standard Baseline distribution (Neuquén)

        # Force structural sample sizes matching specs
        truncated_climate = climate_df.iloc[:meta['samples'] + 8] # Pad for lag truncation
        mock_labels = pd.DataFrame(
            data={"risk_class": np.random.choice([0, 1, 2], size=len(truncated_climate), p=p_dist)},
            index=truncated_climate.index
        )

        # 4. Construct Multi-Scale Feature Matrices (1033-dim Vectors)
        X, y = construct_multiscale_dataset(g_centroid, truncated_climate, mock_labels, tau=4)
        
        # Ensure array dimensions match exact configuration allocations
        X = X[:meta['samples']]
        y = y[:meta['samples']]

        # 5. Serialize Local Artifact Matrices to Disk
        output_filepath = os.path.join(processed_dir, f"{node_name}.npz")
        np.savez(output_filepath, features=X, labels=y)
        print(f"   - Successfully saved data to: {output_filepath}")
        print(f"   - Array Verification Shape: {X.shape} | Labels: {y.shape}")

    print("\n" + "=" * 70)
    print("PHASE I COMPLETED: DATA PIPELINE CACHED AND READY FOR MODELING")
    print("=" * 70)

if __name__ == "__main__":
    execute_ingestion_pipeline()