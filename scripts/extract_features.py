import os
import numpy as np
from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

dataset_path = "data/raw"

features = []
paths = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".png"):
            path = os.path.join(root, file)

            feat = extractor.extract(path)
            features.append(feat)
            paths.append(path)

features = np.array(features)

np.save("embeddings/features.npy", features)

import pickle
pickle.dump(paths, open("embeddings/image_paths.pkl", "wb"))

print("✅ Features extracted:", features.shape)