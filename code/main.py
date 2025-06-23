import time
import numpy as np

from config import FASTA_FOLDER, NUM_CLUSTERS, NUM_LOCAL, MAX_NEIGHBOR_FRAC, LSH_THRESHOLD
from utils.fasta_parser import read_fasta_samples
from clustering.fastclarans import run_fastclarans
from clustering.distance_matrix import create_distance_matrix, compute_minhashes
from utils.helpers import print_cluster_results
from clustering.distance_matrix import minhash_distance_matrix,lsh_distance_matrix

def main():
    #* FIRST TIME PARSING THE INPUT, SAMPLING IT
    #samples = read_fasta_samples_sampled(FASTA_FOLDER, sample_size_mb=10)
    #save_samples_to_fasta_files(samples, '/home/mene/Desktop/CAMIDATASET clean/')
    
    #* SECOND TIME READING THE INPUT, ALREADY PARSED, FASTER :)
    samples = read_fasta_samples(FASTA_FOLDER)

    # 1. Raw set Jaccard distances
    print("\n[1] Raw Jaccard Distance Matrix:")
    t1 = time.time()
    dist_matrix_1, sample_ids_1 = create_distance_matrix(samples)
    t2 = time.time()
    best_labels, best_medoids, best_cost = run_fastclarans(dist_matrix_1, NUM_CLUSTERS, NUM_LOCAL, MAX_NEIGHBOR_FRAC)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_1, best_labels)

    # 2. MinHash distances
    print("\n[2] MinHash Jaccard Distance Matrix:")
    t1 = time.time()
    minhashes = compute_minhashes(samples)
    dist_matrix_2, sample_ids_2 = minhash_distance_matrix(minhashes)
    t2 = time.time()
    best_labels, best_medoids, best_cost = run_fastclarans(dist_matrix_2, NUM_CLUSTERS, NUM_LOCAL, MAX_NEIGHBOR_FRAC)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_2, best_labels)

    # 3. LSH-based partial distance matrix
    print("\n[3] LSH-Filtered Distance Matrix:")
    t1 = time.time()
    dist_matrix_3, sample_ids_3 = lsh_distance_matrix(samples, LSH_THRESHOLD)
    t2 = time.time()
    # fill NaNs with large values to simulate skipping distances
    np.nan_to_num(dist_matrix_3, copy=False, nan=1.0)
    best_labels, best_medoids, best_cost = run_fastclarans(dist_matrix_3, NUM_CLUSTERS, NUM_LOCAL, MAX_NEIGHBOR_FRAC)
    t3 = time.time()
    print(f"Distance matrix (LSH-filtered) created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_3, best_labels)


if __name__ == "__main__":
    main()