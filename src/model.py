import torch
import torch.nn as nn

class P2MSAndesNet(nn.Module):
    """
    Multi-Scale Decoupled Architecture for Andes Virus Risk Forecasting.
    Splits a 1033-dim vector into 1024-dim Genomic and 9-dim Environmental tracks.
    """
    def __init__(self, genomic_dim=1024, environmental_dim=9, num_classes=3):
        super(P2MSAndesNet, self).__init__()
        self.genomic_dim = genomic_dim
        self.environmental_dim = environmental_dim
        
        # Genomic Feature Matrix Extraction Track
        self.genomic_branch = nn.Sequential(
            nn.Linear(genomic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Environmental Eco-Climatic Lag Integration Track
        self.env_branch = nn.Sequential(
            nn.Linear(environmental_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Unified Multi-Scale Fusion Head
        self.classifier = nn.Sequential(
            nn.Linear(64 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        g_feat = x[:, :self.genomic_dim]
        e_feat = x[:, self.genomic_dim:]
        
        g_out = self.genomic_branch(g_feat)
        e_out = self.env_branch(e_feat)
        
        fused = torch.cat((g_out, e_out), dim=1)
        return self.classifier(fused)
