import numpy as np
from numpy.linalg import norm

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-8))

def aggregate_embeddings(list_of_embs):
    return np.mean(np.array(list_of_embs), axis=0).tolist()
