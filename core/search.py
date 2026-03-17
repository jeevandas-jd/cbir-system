# core/search.py

import faiss
import numpy as np
import pickle
from models.feature_extractor import FeatureExtractor

index = faiss.read_index("embeddings/faiss_index.bin")
paths = pickle.load(open("embeddings/image_paths.pkl", "rb"))

extractor = FeatureExtractor()

def search(query_image, k=5):
    q_feat = extractor.extract(query_image)
    q_feat = np.expand_dims(q_feat, axis=0)

    distances, indices = index.search(q_feat, k)

    results = [paths[i] for i in indices[0]]
    return results