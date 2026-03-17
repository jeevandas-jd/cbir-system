# scripts/extract_features.py

import os
import numpy as np
from models.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

features = []
paths = []

dataset_path = "data/raw"

for img_name in os.listdir(dataset_path):
    path = os.path.join(dataset_path, img_name)

    feat = extractor.extract(path)
    features.append(feat)
    paths.append(path)

np.save("embeddings/features.npy", features)

import pickle
pickle.dump(paths, open("embeddings/image_paths.pkl", "wb"))