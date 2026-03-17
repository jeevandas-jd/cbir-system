# core/index.py

import faiss
import numpy as np

features = np.load("embeddings/features.npy")

d = features.shape[1]

index = faiss.IndexFlatL2(d)
index.add(features)

faiss.write_index(index, "embeddings/faiss_index.bin")