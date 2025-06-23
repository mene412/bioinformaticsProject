import os
import time
from collections import defaultdict
from Bio import SeqIO # type: ignore
import numpy as np
from datasketch import MinHash, MinHashLSH # type: ignore
from tqdm import tqdm # type: ignore
import gzip
import random


# --- CONFIG ---
FASTA_FOLDER = "/home/mene/Desktop/CAMIDATASET clean/"  # change to your actual path
K = 21  # k-mer size for MinHash
NUM_PERM = 128  # number of permutations for MinHash
NUM_CLUSTERS = 3
NUM_LOCAL = 3
MAX_NEIGHBOR_FRAC = 0.05
LSH_THRESHOLD = 0.5

def read_fasta_samples(folder_path):
    """
    Reads all FASTA files from a folder and returns their sequences.
    
    Parameters:
    - folder_path: path to folder containing .fasta files
    
    Returns:
    - samples: dictionary with filename as key and set of sequences as value
    """
    samples = {}
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist")
        return samples
    
    # Get all .fasta files in the folder
    fasta_files = [f for f in os.listdir(folder_path) if f.endswith(".fasta")]
    
    if not fasta_files:
        print(f"No .fasta files found in {folder_path}")
        return samples
    
    print(f"Found {len(fasta_files)} FASTA files")
    
    for filename in fasta_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Reading: {file_path}")
        
        # Use filename (without extension) as sample ID
        sample_id = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, "rt") as handle:
                # Fixed: using "fasta" parser for .fasta files
                sequences = set(str(record.seq) for record in SeqIO.parse(handle, "fasta"))
                
                if sequences:
                    samples[sample_id] = sequences
                    print(f"  -> Found {len(sequences)} unique sequences for sample '{sample_id}'")
                else:
                    print(f"  -> Warning: No sequences found in {filename}")
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Successfully read {len(samples)} files")
    return samples

def read_fasta_samples_sampled(fasta_folder, sample_size_mb=10, avg_bytes_per_read=250):
    samples = {}
    reads_to_sample = int((sample_size_mb * 1024 * 1024) / avg_bytes_per_read)

    for subfolder in os.listdir(fasta_folder):
        subfolder_path = os.path.join(fasta_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        reads_path = os.path.join(subfolder_path, "reads", "anonymous_reads.fq.gz")
        if not os.path.exists(reads_path):
            print(f"File non trovato: {reads_path}")
            continue

        sample_id = subfolder
        reservoir = []

        try:
            with gzip.open(reads_path, "rt") as handle:
                print('aperto: ', handle)
                for i, record in enumerate(SeqIO.parse(handle, "fastq")):
                    if i < reads_to_sample:
                        reservoir.append(str(record.seq))
                    else:
                        j = random.randint(0, i)
                        if j < reads_to_sample:
                            reservoir[j] = str(record.seq)

        except EOFError:
            print(f"⚠️ Errore: il file {reads_path} è compresso male o troncato. Campionamento parziale salvato.")
        except Exception as e:
            print(f"⚠️ Errore inaspettato su {reads_path}: {e}")
        finally:
            # Salva i dati raccolti finora, anche se c'è stato un errore
            if reservoir:
                samples[sample_id] = set(reservoir)
                print(f"Sampled {len(reservoir)} reads from {reads_path} (~{sample_size_mb} MB)")
            else:
                print(f"Nessuna lettura campionata da {reads_path}.")

    print('Samples fatti tutti')
    return samples

def save_samples_to_fasta_files(samples, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for sample_id, seqs in samples.items():
        output_path = os.path.join(output_folder, f"{sample_id}.fasta")
        with open(output_path, "w") as f:
            for i, seq in enumerate(seqs):
                f.write(f">{sample_id}_read_{i}\n{seq}\n")
        print(f"Saved {len(seqs)} sequences to {output_path}")

def jaccard_distance(set1, set2):
    # TODO check che faccia giusta l'intersezione e l'unione
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - inter / union if union > 0 else 1

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


# minhashes occupa (n * M) spazio. n array da M
def compute_minhashes(samples):
    minhashes = {}
    for sid, seqs in samples.items(): # sampleID + sample
        mh = MinHash(num_perm=NUM_PERM)
        # O(M), M = numero reads per sample
        for seq in seqs:
            mh.update(seq.encode('utf-8'))
        minhashes[sid] = mh
    return minhashes


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


def run_fastclarans(distance_matrix, k, numlocal=3, maxneighbor=0.05):
    """
    FastCLARANS clustering algorithm implementation.
    
    Parameters:
    - distance_matrix: nxn symmetric matrix of pairwise distances
    - k: number of clusters (medoids)
    - numlocal: number of local searches to perform
    - maxneighbor: fraction determining maximum neighbors to examine
    
    Returns:
    - best_labels: cluster assignment for each point
    - best_medoids: indices of the best medoids found
    - best_cost: total cost of the best clustering
    """
    # Get number of data points
    n = distance_matrix.shape[0]
    
    # Validate inputs
    if k >= n:
        raise ValueError(f"k ({k}) must be less than number of points ({n})")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square")
    
    # Calculate maximum number of neighbors to examine in local search
    num_neighbors = int(maxneighbor * k * (n - k))
    if num_neighbors == 0:
        num_neighbors = 1  # Ensure at least one neighbor is examined
    
    best_labels = None
    best_cost = float("inf")
    best_medoids = None

    # Perform numlocal independent runs
    for run in range(numlocal):
        print(f"Run {run + 1}/{numlocal}: distance matrix shape = {distance_matrix.shape}")
        
        # Initialize: randomly select k medoids
        medoids = np.random.choice(n, k, replace=False).tolist()
        
        # Assign each point to nearest medoid
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        
        # Calculate initial cost (sum of distances from points to their medoids)
        current_cost = sum(distance_matrix[i, medoids[labels[i]]] for i in range(n))

        neighbors_examined = 0
        
        # Local search: try to improve by swapping medoids
        while neighbors_examined < num_neighbors:
            # Randomly select a current medoid to potentially replace
            medoid_idx = np.random.choice(len(medoids))  # Index in medoids list
            m = medoids[medoid_idx]  # Actual medoid point index
            
            # Get all non-medoid points
            non_medoids = [i for i in range(n) if i not in medoids]
            
            # Randomly select a non-medoid as potential replacement
            n_idx = np.random.choice(non_medoids)
            
            # Create new medoids list with the swap
            new_medoids = medoids.copy()
            new_medoids[medoid_idx] = n_idx
            
            # Reassign all points to nearest medoid in new configuration
            new_labels = np.argmin(distance_matrix[:, new_medoids], axis=1)
            
            # Calculate new cost
            new_cost = sum(distance_matrix[i, new_medoids[new_labels[i]]] for i in range(n))

            # If improvement found, accept the change and reset neighbor counter
            if new_cost < current_cost:
                medoids = new_medoids
                labels = new_labels
                current_cost = new_cost
                neighbors_examined = 0  # Reset counter when improvement found
                print(f"  Improvement found: cost {new_cost:.4f}")
            else:
                neighbors_examined += 1

        # Keep track of best solution across all runs
        if current_cost < best_cost:
            best_cost = current_cost
            best_labels = labels.copy()
            best_medoids = medoids.copy()
            print(f"  New best solution: cost {best_cost:.4f}")

    return best_labels, best_medoids, best_cost


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


def print_cluster_results(sample_ids, labels):
    cluster_map = defaultdict(list)
    for sid, label in zip(sample_ids, labels):
        cluster_map[label].append(sid)
    for label, members in cluster_map.items():
        print(f"Cluster {label}: {members}")


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