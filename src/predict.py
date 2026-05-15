import torch
import numpy as np
from src.model import P2MSAndesNet
from src.data_loader import GenomicFeatureExtractor

def get_live_prediction(fasta_string, environmental_9dim_vector):
    """
    Inputs:
        fasta_string: Raw nucleotide sequence of the circulating strain.
        environmental_9dim_vector: [Temp/Precip/Solar at t, t-4, t-8].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Brain
    model = P2MSAndesNet()
    model.load_state_dict(torch.load("data/processed/global_model.pth"))
    model.to(device).eval()
    
    # 2. Extract Genomic Motif Profile
    extractor = GenomicFeatureExtractor()
    # transform_sequences expects a dict
    genomic_vector = extractor.transform_sequences({"current_strain": fasta_string})[0]
    
    # 3. Fuse into the 1033-dim Vector
    input_vector = np.concatenate([genomic_vector, environmental_9dim_vector])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 4. Predict
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        risk_class = torch.argmax(logits, dim=1).item()
        
    classes = {0: "Low (Baseline)", 1: "Moderate (Monitor)", 2: "High (Urgent Action)"}
    print(f"--- P²MS-Andes Forecasting Result ---")
    print(f"Predicted Risk: {classes[risk_class]}")
    print(f"Confidence: {probabilities[0][risk_class]*100:.2f}%")
    return risk_class