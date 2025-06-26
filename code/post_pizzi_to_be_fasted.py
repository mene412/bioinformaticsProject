from Bio import SeqIO
from collections import Counter
import numpy as np
import random

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def prepare_all_samples(fasta_file, k):
    """
    Reads sequences from a FASTA file and returns a list of sample dictionaries.

    Each dictionary contains:
        - 'kmers': a set of k-mers
        - 'label': index of the nearest medoid (initially 0)
        - 'second_label': index of the second nearest medoid (initially 0)

    Parameters:
        fasta_file (str): Path to the FASTA file
        k (int): k-mer size

    Returns:
        List[Dict]: List of sample dictionaries
    """
    samples = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        kmers = get_kmers(str(record.seq), k)
        samples.append({
            'kmers': set(kmers),
            'label': 0,
            'second_label': 0
        })
    return samples


def distance(a, b, metric='jaccard'):
    """
    Computes distance between two sets or lists of k-mers.
    """
    if metric == 'jaccard':
        a_set, b_set = set(a), set(b)
        intersection = len(a_set & b_set)
        union = len(a_set | b_set)
        return 1.0 - (intersection / union) if union > 0 else 1.0

    elif metric == 'braycurtis':
        a_counts = Counter(a)
        b_counts = Counter(b)
        all_keys = set(a_counts.keys()).union(b_counts.keys())
        a_vec = np.array([a_counts[k] for k in all_keys])
        b_vec = np.array([b_counts[k] for k in all_keys])
        return np.sum(np.abs(a_vec - b_vec)) / np.sum(a_vec + b_vec)

    else:
        raise ValueError("Unsupported metric: choose 'jaccard' or 'braycurtis'")

def assign(centers, samples, metric='jaccard'):
    """
    Assigns each sample to the nearest and second nearest medoids.

    Parameters:
        centers (List[int]): Indices of current medoid samples in `samples`
        samples (List[Dict]): Samples with 'kmers', 'label', and 'second_label'
        metric (str): Distance metric to use
    """
    for sample in samples:
        dists = [
            (distance(sample['kmers'], samples[center_idx]['kmers'], metric), i)
            for i, center_idx in enumerate(centers)
        ]
        dists.sort()  # sort by distance

        # Set label as nearest medoid's position in `centers`
        sample['label'] = dists[0][1]  # index in centers list
        sample['second_label'] = dists[1][1] if len(dists) > 1 else dists[0][1]



def fastCLARANS(samples, k, numlocal, maxneighbor, metric='jaccard'):
    """
    CLARANS clustering algorithm with in-place label assignment and distance caching.

    Parameters:
        samples (List[Dict]): Each sample has 'kmers', 'label', 'second_label'
        k (int): Number of clusters
        numlocal (int): Number of local minima to search
        maxneighbor (int): Maximum neighbors to explore per local search
        metric (str): Distance metric: 'jaccard' or 'braycurtis'

    Returns:
        List[int]: Indices of best medoids
    """
    n = len(samples)
    mincost = float('inf')
    bestnode = None

    # Initialize distance matrix with -1
    distance_matrix = np.full((n, n), -1.0)

    def get_cached_distance(i, j):
        if distance_matrix[i][j] == -1:
            d = distance(samples[i]['kmers'], samples[j]['kmers'], metric)
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d
        return distance_matrix[i][j]

    for _ in range(numlocal):
        current = random.sample(range(n), k)
        assign(current, samples, metric)

        j = 1
        while j <= maxneighbor:
            Om = random.choice(current)
            Op = random.choice([idx for idx in range(n) if idx not in current])

            TCmp = 0.0

            for idx_Oj, Oj in enumerate(samples):
                label_idx = current[Oj['label']]
                second_label_idx = current[Oj['second_label']]

                d_Oj_Om = get_cached_distance(idx_Oj, Om)
                d_Oj_Op = get_cached_distance(idx_Oj, Op)
                d_Oj_second = get_cached_distance(idx_Oj, second_label_idx)

                if label_idx == Om and d_Oj_second < d_Oj_Op:
                    Cjmp = d_Oj_second - d_Oj_Om
                elif label_idx == Om and d_Oj_Op < d_Oj_second:
                    Cjmp = d_Oj_Op - d_Oj_Om
                elif label_idx != Om and d_Oj_second < d_Oj_Op:
                    Cjmp = 0.0
                else:
                    Cjmp = d_Oj_Op - d_Oj_second

                TCmp += Cjmp

            if TCmp < 0:
                current.remove(Om)
                current.append(Op)
                assign(current, samples, metric)
                j = 1
            else:
                j += 1

        current_cost = sum(
            get_cached_distance(idx, current[sample['label']])
            for idx, sample in enumerate(samples)
        )
        if current_cost < mincost:
            mincost = current_cost
            bestnode = current.copy()

    return bestnode
