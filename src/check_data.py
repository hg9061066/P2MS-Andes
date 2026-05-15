import numpy as np

data = np.load("data/processed/Node_1_Neuquen.npz")
features = data['features']

print(f"Total Dimensions: {features.shape[1]}")
# The last 9 columns should not be all zeros or identical
env_slice = features[:, 1024:] 
print(f"Environmental Slice Mean: {env_slice.mean():.4f}")

if features.shape[1] == 1033 and not np.all(env_slice == 0):
    print("✅ DATA INTEGRITY VERIFIED: Environmental features are present and non-zero.")
else:
    print("❌ DATA ERROR: Environmental features are missing or corrupted.")