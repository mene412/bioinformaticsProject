from datasketch import MinHashLSH # type: ignore
from config import NUM_PERM
import numpy as np
from clustering.distances import jaccard_distance, compute_minhashes

# O(n*n) in spazio e in tempo. n = numero sample metagenomici
def create_distance_matrix(samples):
    sample_ids = list(samples.keys())
    n = len(sample_ids)
    print("Creating matrix ", n, " x ", n)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jaccard_distance(samples[sample_ids[i]], samples[sample_ids[j]])
            matrix[i, j] = matrix[j, i] = d
    return matrix, sample_ids

def minhash_distance_matrix(minhashes):
    sample_ids = list(minhashes.keys())
    n = len(sample_ids)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = minhashes[sample_ids[i]].jaccard(minhashes[sample_ids[j]])
            d = 1 - sim
            matrix[i, j] = matrix[j, i] = d
    return matrix, sample_ids

def lsh_distance_matrix(samples, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    minhashes = compute_minhashes(samples)
    sample_ids = list(samples.keys())

    for sid in sample_ids:
        lsh.insert(sid, minhashes[sid])

    n = len(sample_ids)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        sid1 = sample_ids[i]
        candidates = lsh.query(minhashes[sid1])
        for sid2 in candidates:
            j = sample_ids.index(sid2)
            if np.isnan(matrix[i, j]):
                sim = minhashes[sid1].jaccard(minhashes[sid2])
                d = 1 - sim
                matrix[i, j] = matrix[j, i] = d
    return matrix, sample_ids
