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
        return self.hasher.transform(hashed_features).toarray().astype(np.float32)

class ClimateDataIngestor:
    def __init__(self, start_date="20240101", end_date="20250101"):
        self.start_date = start_date
        self.end_date = end_date
        self.params = "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"

    def fetch_regional_climate(self, lat, lon):
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={self.params}&community=AG&longitude={lon}&latitude={lat}&start={self.start_date}&end={self.end_date}&format=JSON"
        response = requests.get(url, timeout=30)
        if response.status_code != 200: raise RuntimeError("API Request Failed")
        param_data = response.json()['properties']['parameter']
        dfs = [pd.DataFrame.from_dict(values, orient="index", columns=[param]) for param, values in param_data.items()]
        df = pd.concat(dfs, axis=1)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.columns = ["temperature", "precipitation", "solar_radiation"]
        return df.resample('W').mean().sort_index()

def construct_multiscale_dataset(genomic_centroid, climate_df, case_labels_df, tau=4):
    scaler = StandardScaler()
    climate_scaled_df = pd.DataFrame(scaler.fit_transform(climate_df), index=climate_df.index, columns=climate_df.columns)
    fused_features, target_labels = [], []
    for current_week in case_labels_df.index:
        t_minus_tau = current_week - pd.Timedelta(weeks=tau)
        t_minus_2tau = current_week - pd.Timedelta(weeks=2*tau)
        if t_minus_2tau >= climate_scaled_df.index.min():
            e_t = climate_scaled_df.asof(current_week).values
            e_tau = climate_scaled_df.asof(t_minus_tau).values
            e_2tau = climate_scaled_df.asof(t_minus_2tau).values
            x_vector = np.concatenate([genomic_centroid, np.concatenate([e_t, e_tau, e_2tau])])
            fused_features.append(x_vector)
            target_labels.append(case_labels_df.loc[current_week, 'risk_class'])
    return np.array(fused_features, dtype=np.float32), np.array(target_labels, dtype=np.int64)
