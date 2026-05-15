import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler

class GenomicFeatureExtractor:
    def __init__(self, k_sizes=[3, 5, 7], target_dim=1024):
        self.k_sizes = k_sizes
        self.target_dim = target_dim
        self.hasher = FeatureHasher(n_features=target_dim, input_type='pair')

    def _extract_kmers(self, sequence):
        kmers = []
        sequence = sequence.upper().replace("N", "").replace("-", "")
        for k in self.k_sizes:
            for i in range(len(sequence) - k + 1):
                kmers.append(sequence[i:i+k])
        return kmers

    def transform_sequences(self, fasta_records):
        hashed_features = []
        for seq_id, seq in fasta_records.items():
            kmers = self._extract_kmers(seq)
            counts = {}
            for kmer in kmers:
                counts[kmer] = counts.get(kmer, 0) + 1
            total = sum(counts.values()) if counts else 1
            freq_pairs = [(kmer, count / total) for kmer, count in counts.items()]
            hashed_features.append(freq_pairs)
        return self.hasher.transform(hashed_features).toarray().astype('float32')

class NASAPowerClient:
    """Handles real-time environmental data fetching from NASA satellites."""
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def fetch_nasa_data(self, lat, lon, start_date, end_date):
        params = {
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"NASA API Request Failed: {response.status_code}")
        
        param_data = response.json()['properties']['parameter']
        # Combine parameters into a single DataFrame
        dfs = [pd.DataFrame.from_dict(values, orient="index", columns=[param]) 
               for param, values in param_data.items()]
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.columns = ["temperature", "precipitation", "solar_radiation"]
        
        # Resample to weekly mean to match training resolution
        return df.resample('W').mean().sort_index()

def construct_multiscale_dataset(genomic_centroid, climate_df, case_labels_df, tau=4):
    scaler = StandardScaler()
    climate_scaled_df = pd.DataFrame(scaler.fit_transform(climate_df), 
                                     index=climate_df.index, 
                                     columns=climate_df.columns)
    fused_features, target_labels = [], []
    for current_week in case_labels_df.index:
        t_minus_tau = current_week - pd.Timedelta(weeks=tau)
        t_minus_2tau = current_week - pd.Timedelta(weeks=2*tau)
        
        if t_minus_2tau >= climate_scaled_df.index.min():
            e_t = climate_scaled_df.asof(current_week).values
            e_tau = climate_scaled_df.asof(t_minus_tau).values
            e_2tau = climate_scaled_df.asof(t_minus_2tau).values
            
            env_vector = np.concatenate([e_t, e_tau, e_2tau])
            fused_vector = np.concatenate([genomic_centroid, env_vector])
            fused_features.append(fused_vector)
            target_labels.append(case_labels_df.loc[current_week])
            
    return np.array(fused_features), np.array(target_labels)