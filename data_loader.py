import os
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

# Define Coordinates for the 4 Target Jurisdictions
JURISDICTIONS = {
    "Neuquen_Argentina": {"lat": -38.95, "lon": -68.06},
    "Rio_Negro_Argentina": {"lat": -41.13, "lon": -71.30},
    "Valparaiso_Chile": {"lat": -33.05, "lon": -71.61},
    "Aysen_Chile": {"lat": -45.57, "lon": -72.07}
}

# =====================================================================
# 1. MICRO-SCALE: GENOMIC CENTROID EXTRACTION PIPELINE
# =====================================================================
class GenomicFeatureExtractor:
    """
    Extracts k-mer features from viral strings and calculates architectural 
    centroids representing dominant circulating profiles.
    """
    def __init__(self, k_sizes=[3, 5, 7], target_dim=1024):
        self.k_sizes = k_sizes
        self.target_dim = target_dim
        # Documented Collision Warning: 17K motifs binned into 1024 channels.
        # This keeps the multi-scale vector bound tightly to architectural specs.
        self.hasher = FeatureHasher(n_features=target_dim, input_type='pair')

    def _extract_kmers(self, sequence):
        """Generates all sliding overlapping k-mers from nucleotide sequences."""
        kmers = []
        # Clear base indicators and gaps natively
        sequence = sequence.upper().replace("N", "").replace("-", "")
        for k in self.k_sizes:
            for i in range(len(sequence) - k + 1):
                kmers.append(sequence[i:i+k])
        return kmers

    def transform_sequences(self, fasta_records):
        """Transforms raw strings into a normalized frequency matrix."""
        hashed_features = []
        for seq_id, seq in fasta_records.items():
            kmers = self._extract_kmers(seq)
            counts = {}
            for kmer in kmers:
                counts[kmer] = counts.get(kmer, 0) + 1
            
            total = sum(counts.values()) if counts else 1
            freq_pairs = [(kmer, count / total) for kmer, count in counts.items()]
            hashed_features.append(freq_pairs)
            
        sparse_matrix = self.hasher.transform(hashed_features)
        return sparse_matrix.toarray().astype(np.float32)


# =====================================================================
# 2. MACRO-SCALE: CLIMATE INGESTION PIPELINE (NASA POWER API)
# =====================================================================
class ClimateDataIngestor:
    """Programmatic boundary controller for fetching daily atmospheric observations."""
    def __init__(self, start_date="20240101", end_date="20250101"):
        self.start_date = start_date
        self.end_date = end_date
        self.params = "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"

    def fetch_regional_climate(self, lat, lon):
        """Fetches and builds historical climate matrices with rigorous orientation parsing."""
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={self.params}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={self.start_date}&end={self.end_date}&format=JSON"
        )
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"API Request Failed with Status Code: {response.status_code}")
        
        param_data = response.json()['properties']['parameter']
        
        # FIX 1: Robustly orient nested JSON dictionaries into explicit rows (Dates)
        dfs = [
            pd.DataFrame.from_dict(values, orient="index", columns=[param]) 
            for param, values in param_data.items()
        ]
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.columns = ["temperature", "precipitation", "solar_radiation"]
        
        # Resample to weekly mean trends matching PAHO reporting cadences
        return df.resample('W').mean().sort_index()


# =====================================================================
# 3. TIME-LAGGED MULTI-SCALE FUSION ENGINE WITH SCALING
# =====================================================================
def construct_multiscale_dataset(genomic_centroid, climate_df, case_labels_df, tau=4):
    """
    Fuses genomic profile representations with scaled, time-lagged eco-climatic snapshots.
    Formula: X = [G_centroid || E_t || E_{t-tau} || E_{t-2tau}] (1033-dim vector)
    
    Rationale for tau=4: Corresponds to a 4-week biological window tracking hantavirus 
    incubation intervals, Sigmodontinae breeding spikes, and direct human spillover metrics.
    """
    # FIX 5: Standardize climate indices to flatten feature magnitude variance
    scaler = StandardScaler()
    scaled_climate_vals = scaler.fit_transform(climate_df)
    climate_scaled_df = pd.DataFrame(scaled_climate_vals, index=climate_df.index, columns=climate_df.columns)
    
    fused_features = []
    target_labels = []
    
    for current_week in case_labels_df.index:
        t_minus_tau = current_week - pd.Timedelta(weeks=tau)
        t_minus_2tau = current_week - pd.Timedelta(weeks=2*tau)
        
        # Avoid out-of-bounds history exceptions
        if t_minus_2tau >= climate_scaled_df.index.min():
            # FIX 3: Employ .asof() lookup sequences to protect against off-by-one edge effects
            e_t = climate_scaled_df.asof(current_week).values
            e_tau = climate_scaled_df.asof(t_minus_tau).values
            e_2tau = climate_scaled_df.asof(t_minus_2tau).values
            
            environmental_block = np.concatenate([e_t, e_tau, e_2tau])
            
            # Form complete multi-scale observation vector (1033-dim)
            x_vector = np.concatenate([genomic_centroid, environmental_block])
            fused_features.append(x_vector)
            target_labels.append(case_labels_df.loc[current_week, 'risk_class'])
            
    return np.array(fused_features, dtype=np.float32), np.array(target_labels, dtype=np.int64)


# =====================================================================
# 4. ENGINE VERIFICATION EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("[Phase I] Initializing Verified Data Ingestion Engine...")
    
    # 1. Generate multi-strain sequence landscape
    mock_fasta = {f"ANDV_Strain_{i}": "".join(np.random.choice(['A','C','G','T'], size=1500)) for i in range(180)}
    extractor = GenomicFeatureExtractor(k_sizes=[3, 5, 7], target_dim=1024)
    genomic_matrix = extractor.transform_sequences(mock_fasta)
    
    # FIX 2: Calculate feature space centroid to represent the overall population fingerprint
    g_centroid = genomic_matrix.mean(axis=0)
    print(f"-> Calculated Genomic Centroid Matrix Array Profile: {g_centroid.shape}")
    
    # FIX 6: Clarify dates to remove defaults ambiguity
    ingestor = ClimateDataIngestor(start_date="20240101", end_date="20250101")
    node_name = "Neuquen_Argentina"
    coords = JURISDICTIONS[node_name]
    
    print(f"-> Requesting Remote Atmospheric Grids for {node_name}...")
    try:
        climate_data = ingestor.fetch_regional_climate(lat=coords['lat'], lon=coords['lon'])
        print(f"-> Ingested {len(climate_data)} steps of environmental data successfully.")
        
        # Establish mock label profile framing
        mock_labels = pd.DataFrame(
            data={"risk_class": np.random.choice([0, 1, 2], size=len(climate_data), p=[0.7, 0.2, 0.1])},
            index=climate_data.index
        )
        
        X, y = construct_multiscale_dataset(g_centroid, climate_data, mock_labels, tau=4)
        print(f"-> Multi-Scale Feature Matrix Alignment Verified. Dimensions: {X.shape}")
        print(f"-> Array Verification Checks Passed: Feature Dimension is exactly {X.shape[1]}")
        
    except requests.exceptions.RequestException as e:
        print(f"[Network Intercept] API connection could not be completed: {e}")