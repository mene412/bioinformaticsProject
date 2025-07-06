from Bio import SeqIO
from collections import Counter
import numpy as np
import random
import os
import time
from evaluation import compute_evaluation_metrics
from scipy.sparse import csr_matrix


def kmer_to_index(kmer):
    """Converts a k-mer string to a base-4 integer index."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = 0
    for char in kmer:
        idx = idx * 4 + mapping[char]
    return idx

'''
def valid_kmers(seq, k):
    """
    Yields valid k-mers containing only ACGT.
    """
    i = 0
    while i <= len(seq) - k:
        kmer = seq[i:i+k]
        for j in range(k):
            if kmer[j] not in 'ACGT':
                i += j + 1  # Skip to character after invalid base
                break
        else:
            yield kmer
            i += 1
'''

def count_kmers_sparse(seq: str, k: int, vector_size: int) -> csr_matrix:
    """
    Returns a sparse vector (1 x 4^k) for all valid ACGT k-mers in a sequence.
    """
    data = []
    indices = []
    i = 0
    while i <= len(seq) - k:
        kmer = seq[i:i+k]
        for j in range(k):
            if kmer[j] not in 'ACGT':
                i += j + 1
                break
        else:
            idx = kmer_to_index(kmer)
            indices.append(idx)
            data.append(1)
            i += 1

    # Aggregate repeated indices
    if not indices:
        return csr_matrix((1, vector_size), dtype=np.uint16)
    
    counts = Counter(indices)
    idxs, freqs = zip(*counts.items())
    return csr_matrix((freqs, ([0]*len(idxs), idxs)), shape=(1, vector_size), dtype=np.uint16)

def load_true_labels(true_labels_path):
    labels = {}
    with open(true_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                labels[parts[0]] = " ".join(parts[1:]).lstrip(": ")
    return labels

# List of files without sequences but only metadata
#todo try to re-download them with specific single wget after checking the link
skip_files = {'GOS45.fasta', 'GOS43.fasta', 'GOS41.fasta', 'GOS40.fasta', 'GOS37.fasta', 'GOS39.fasta', 'GOS42.fasta', 'GOS38.fasta', 'GOS44.fasta'}

def prepare_all_samples(samples_folder_path, k, true_labels_path):
    # num_kmers = 4 ** k
    vector_size = 4 ** k
    samples = []
    labels = load_true_labels(true_labels_path)

    for filename in os.listdir(samples_folder_path):
        if filename.endswith('.fasta') or filename.endswith('.fa'):
            if filename in skip_files:
                print(f"Skipping file {filename} (in skip list)")
                continue

            full_path = os.path.join(samples_folder_path, filename)
            print(f"Processing file: {filename} ({os.path.getsize(full_path) / 1024:.4f} KB)")

            # sample_array = np.zeros(num_kmers, dtype=np.uint32)
            total_vector = csr_matrix((1, vector_size), dtype=np.uint16)
            start_time = time.time()
            '''
            for record in SeqIO.parse(full_path, "fasta"):
                num_reads += 1
                for kmer in valid_kmers(str(record.seq), k):
                    idx = kmer_to_index(kmer)
                    sample_array[idx] += 1
            '''
            for record in SeqIO.parse(full_path, "fasta"):
                vec = count_kmers_sparse(str(record.seq), k, vector_size)
                total_vector += vec

            end_time = time.time()
            print(f"Document parsed in {end_time - start_time:.4f} seconds")


            id = os.path.splitext(filename)[0]
            ambiental_label = labels.get(id, "")

            
            printfile = id + "_done.txt"
            with open(printfile, "w", encoding="utf-8") as f:
                    f.write(f"Ambiental label for {id}--> {ambiental_label}")

            samples.append({
                'id': id,
                # 'counts': sample_array,
                'vector': total_vector,
                'label': -1,
                'second_label': -1,
                'ambiental_label': ambiental_label
            })
            # numpy arrays with a lot of zeros!!
            # find quantity of kmers counted
            # print(f"Added sample {id} with {len(kmer_counter)} k-mers")

    return samples

'''
def distance(a, b, metric='braycurtis'):
    if metric == 'braycurtis':
        numerator = np.sum(np.minimum(a, b))
        denominator = np.sum(a) + np.sum(b)
        return 1.0 - 2 * (numerator / denominator) if denominator > 0 else 1.0
    elif metric == 'jaccard':
        a_set = set(np.nonzero(a)[0])
        b_set = set(np.nonzero(b)[0])
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return 1.0 - (intersection / union) if union > 0 else 1.0
    else:
        raise ValueError("Unsupported metric: choose 'jaccard' or 'braycurtis'")
'''

def distance(a: csr_matrix, b: csr_matrix, metric='braycurtis') -> float:
    """
    Computes Bray-Curtis or Jaccard distance between two sparse 1D vectors.
    
    Parameters
    ----------
    a, b : csr_matrix (shape (1, N))
        Sparse row vectors
    metric : str
        'braycurtis' or 'jaccard'
    
    Returns
    -------
    float
        Distance between a and b
    """
    if metric == 'braycurtis':
        # Vectorized and sparse-safe
        numerator = a.minimum(b).sum()
        denominator = a.sum() + b.sum()
        return 1.0 - 2 * (numerator / denominator) if denominator > 0 else 1.0

    elif metric == 'jaccard':
        # Jaccard using binary presence
        a_bin = a.astype(bool)
        b_bin = b.astype(bool)
        intersection = (a_bin.multiply(b_bin)).nnz
        union = a_bin.nnz + b_bin.nnz - intersection
        return 1.0 - (intersection / union) if union > 0 else 1.0

    else:
        raise ValueError("Unsupported metric: choose 'braycurtis' or 'jaccard'")

def fastCLARANS(samples, k, numlocal, maxneighbor, metric, distance_matrix):
    file_fastCLARANS = "fasCLARANS_" + str(maxneighbor) + ".txt"
    with open(file_fastCLARANS, "w", encoding="utf-8") as f:
        f.write(f"start")

    n = len(samples)
    mincost = float('inf')
    bestnode = None
    run_usage_matrix = np.zeros((n, n), dtype=np.uint8)  # Local usage matrix
    how_many_cells_used = 0

    def get_cached_distance(i, j, metric):
        nonlocal how_many_cells_used
        if distance_matrix[i, j] < 0:
            d = distance(samples[i]['vector'], samples[j]['vector'], metric)
            distance_matrix[i, j] = distance_matrix[j, i] = d
        if not run_usage_matrix[i, j]:
            run_usage_matrix[i, j] = run_usage_matrix[j, i] = 1
            how_many_cells_used += 1
        return distance_matrix[i, j]

    def assign(centers):
        cost = 0.0
        for idx, sample in enumerate(samples):
            dists = [(get_cached_distance(idx, center_idx, metric), i)
                     for i, center_idx in enumerate(centers)]
            dists.sort()
            # be aware of the indeces!!!!
            sample['label'] = dists[0][1]
            sample['second_label'] = dists[1][1] if len(dists) > 1 else dists[0][1]
            cost += dists[0][0]
        return cost

    for _ in range(numlocal):
        current = random.sample(range(n), k)
        current_cost = assign(current)

        j = 1
        while j <= maxneighbor:
            Op = random.choice([i for i in range(n) if i not in current])
            delta_TS = [0.0] * k
            d_Op_to_center = get_cached_distance(Op, current[samples[Op]['label']], metric)
            for i in range(k):
                delta_TS[i] -= d_Op_to_center

            for idx_Oj, Oj in enumerate(samples):
                d_Oj_Op = get_cached_distance(idx_Oj, Op, metric)

                current_label = Oj['label']
                idx_Oj_nearest = current[current_label]
                d_Oj_Om = get_cached_distance(idx_Oj, idx_Oj_nearest, metric)
                idx_Oj_second = current[Oj['second_label']]
                d_Oj_second = get_cached_distance(idx_Oj, idx_Oj_second, metric)
                delta_TS[current_label] += min(d_Oj_Op, d_Oj_second) - d_Oj_Om
                if d_Oj_Op < d_Oj_Om:
                    for i in range(k):
                        if i != current_label:
                            delta_TS[i] += d_Oj_Op - d_Oj_Om

            min_index = np.argmin(delta_TS)
            if delta_TS[min_index] < 0:
                # best_medoid = current[min_index]
                current[min_index] = Op
                current_cost += delta_TS[min_index]
                assign(current)
                j = 1
            else:
                j += 1

        if current_cost < mincost:
            mincost = current_cost
            bestnode = current.copy()
            print(f"New best cost: {mincost:.5f}")

    print(f"Number of distances computed in this run: {how_many_cells_used}")
    return bestnode


def main():
    dataset_folder_path = "C:/Users/Lorenzo Berlese/Desktop/metagenomics project/alcuni_dataset_gos"
    true_labels_path = "C:/Users/Lorenzo Berlese/Desktop/metagenomics project/true_labels.txt"
    k_mer_size = 5
    num_clusters = 3
    numlocal = 3
    metrics = 'braycurtis'

    print("Preparing samples...")
    samples = prepare_all_samples(dataset_folder_path, k_mer_size, true_labels_path)
    n = len(samples)

    distance_matrix = np.full((n, n), -1.0, dtype=np.float32)

    # for maxneighbor in [10, 20, 30, 40, 50, 60, 65]:
    for maxneighbor in [4]:
        print(f"\nRunning fastCLARANS with maxneighbor = {maxneighbor}")
        start_time = time.time()
        medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor,
                              metrics, distance_matrix)
        end_time = time.time()
        with open(f"cluster{maxneighbor}.txt", "w", encoding="utf-8") as f:
            f.write(f"Medoids: {medoids}, Runtime: {end_time - start_time:.2f}s")

        label_metrics, macro_avg, micro_avg = compute_evaluation_metrics(samples)
        with open(f"cluster{maxneighbor}.txt", "w", encoding="utf-8") as f:
            f.write('- - - RESULTS - - -')
            for label, m in label_metrics.items():
                f.write(f"Label '{label}': TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, "
                    f"Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1score={m['F1score']:.2f}")
            f.write("Macro-Averaged Metrics:", macro_avg)
            f.write("Micro-Averaged Metrics:", micro_avg)


        with open(f"cluster{maxneighbor}.txt", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(f"{s['id']}\n{s['label']}\n{s['second_label']}\n{s['ambiental_label']}\n\n")
        
        
        num_kmers = 4 ** k_mer_size
        a = np.random.randint(low = 0, high = 110000, size = num_kmers, dtype=np.uint32)
        b = np.random.randint(low = 0, high = 110000, size = num_kmers, dtype=np.uint32)

        start = time.time()
        distance_value = distance(a, b, metrics)
        end = time.time()

        print(f"Distance between two zero vectors: {distance_value:.5f}, computed in {end - start:.4f} seconds")


if __name__ == "__main__":
    main()