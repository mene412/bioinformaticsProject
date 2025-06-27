import time
import numpy as np
import pandas as pd

from config import FASTA_FOLDER, NUM_CLUSTERS, NUM_LOCAL, MAX_NEIGHBOR_FRAC, LSH_THRESHOLD, MAX_ITERATIONS, OUTPUT_SAMPLE_FOLDER
from utils.fasta_parser import read_fasta_samples, read_fasta_samples_sampled, save_samples_to_fasta_files, read_fasta_samples_sampled_lollo
from clustering.fastclarans import run_fastclarans
from clustering.distance_matrix import create_distance_matrix, compute_minhashes
from utils.helpers import print_cluster_results
from clustering.distance_matrix import minhash_distance_matrix,lsh_distance_matrix
from clustering.fastpam import run_fast_pam
from clustering.PAM import run_pam

def main():
    #* FIRST TIME PARSING THE INPUT, SAMPLING IT
    #samples = read_fasta_samples_sampled_lollo(FASTA_FOLDER, sample_size_mb=10)
    #save_samples_to_fasta_files(samples, OUTPUT_SAMPLE_FOLDER)
    
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
    print("Cost for fastCLARANS[Raw]:", best_cost)

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
    print("Cost for fastCLARANS[minhash]:", best_cost)

    '''
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
    '''


    '''
    fastPAM
    
    '''
    t2 = time.time()
    best_labels, best_medoids, best_cost = run_fast_pam(dist_matrix_1, NUM_CLUSTERS, MAX_ITERATIONS)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_1, best_labels)
    print("Cost for fastPAM[Raw]:", best_cost)

    # 2. MinHash distances
    t2 = time.time()
    best_labels, best_medoids, best_cost = run_fast_pam(dist_matrix_2, NUM_CLUSTERS, MAX_ITERATIONS)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_2, best_labels)
    print("Cost for fastPAM[MinHash]:", best_cost)


    '''
    
    PAM
    
    '''

    t2 = time.time()
    best_labels, best_medoids, best_cost = run_pam(dist_matrix_1, NUM_CLUSTERS, MAX_ITERATIONS)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_1, best_labels)
    print("Cost for PAM[Raw]:", best_cost)

    # 2. MinHash distances
    t2 = time.time()
    best_labels, best_medoids, best_cost = run_pam(dist_matrix_2, NUM_CLUSTERS, MAX_ITERATIONS)
    t3 = time.time()
    print(f"Distance matrix created in {t2 - t1:.2f}s; clustering done in {t3 - t2:.2f}s")
    print_cluster_results(sample_ids_2, best_labels)
    print("Cost for PAM[MinHash]:", best_cost)


    df = pd.DataFrame(dist_matrix_1, index=sample_ids_1, columns=sample_ids_1)
    print(df.round(3))  # 3 decimali

    df = pd.DataFrame(dist_matrix_2, index=sample_ids_1, columns=sample_ids_1)
    print(df.round(3))  # 3 decimali

    save_distance_matrix_to_csv(dist_matrix_1, sample_ids_1, "distanze_raw.csv")

def save_distance_matrix_to_csv(matrix, sample_ids, filename="C:\\Users\\Lorenzo Berlese\\Desktop\\metagenomics project\figanap.csv"):
    df = pd.DataFrame(matrix, index=sample_ids, columns=sample_ids)
    df.to_csv(filename)

if __name__ == "__main__":
    main()