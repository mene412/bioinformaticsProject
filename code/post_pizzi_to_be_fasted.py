from Bio import SeqIO
from collections import Counter
import numpy as np
import random
import os
import time
import sys
from evaluation import compute_evaluation_metrics

def get_kmers(seq, k):
    """
    Returns a list of k-mers.

    Parameters
    ----------
        seq : str
            The starting sequence
        k : int
            k-mer size
    
    Returns
    -------
        list
            list of the k-mers in seq
    """
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

skip_files = {'GOS45.fasta', 'GOS43.fasta', 'GOS41.fasta', 'GOS40.fasta', 'GOS37.fasta', 'GOS39.fasta', 'GOS42.fasta', 'GOS38.fasta', 'GOS44.fasta'}

def prepare_all_samples(samples_folder_path, k, true_labels_path):
    """
    Reads all samples from the folder. Each FASTA file correspond to a sample.
    Each sample is the combined set of all k-mers from all the sequences in the file.

    Parameters
    ----------
        folder_path : str 
            Path to a folder containing FASTA files
        k : int
            k-mer size

    Returns
    ----------
        List[Dict]
        A list of sample dictionaries. Each dictionary represents a sample and has the following keys:

        - 'id' : str  
            Identifier for the sample, retrieved from the filename in the format 'GOSXX'.
        - 'kmers' : set of str  
            A set of all unique k-mers found in the sample.
        - 'counts' : dict of {str: int}  
            A dictionary mapping each k-mer to its count in the sample.
        - 'label' : int  
            Default label for the sample. Initialized to -1.
        - 'second_label' : int  
            Secondary label for the sample. Initialized to -1.
        - 'ambiental_label' : str or int  
            Additional label for environmental context, retrieved from `true_labels.txt`.
    """
    samples = []
    for filename in os.listdir(samples_folder_path):
        if filename.endswith('.fasta') or filename.endswith('.fa'):
            if filename in skip_files:
                print(f"Skipping file {filename} (in skip list)")
                continue
            full_path = os.path.join(samples_folder_path, filename)

            kmer_counter = Counter()

            print(f"Processing file: {filename} of dimension {os.path.getsize(full_path) / 1024:.4f} KB")
            
            num_reads = 0
            start_time = time.time()
            for record in SeqIO.parse(full_path, "fasta"):
                num_reads += 1
                # print(".", end="", flush=True)
                if num_reads % 10000 == 0:
                    print(f"Processed {num_reads} reads so far...")
                kmer_counter.update(get_kmers(str(record.seq), k))
            end_time = time.time()


            print(f"Found {len(kmer_counter)} k-mers. Done in {end_time - start_time:.4f} seconds")
            
            start_time = time.time()
            id = os.path.splitext(filename)[0] # Use filename without extension as ID
            ambiental_label = ""
            #todo check true_labels path
            with open(os.path.join(true_labels_path), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:  # <-- evita l'accesso a lista vuota
                        continue
                    label_id = parts[0]
                    if (label_id == id):
                        ambiental_label = " ".join(parts[1:]).lstrip(": ")
                        print(f"Ambiental label for {id}--> {ambiental_label}")
            end_time = time.time()
            print(f"Ambiental label for {id}: {ambiental_label}, found in {end_time - start_time:.4f} seconds")

            size_before = sys.getsizeof(kmer_counter.keys())
            start_time = time.time()
            samples.append({
                'id': id,
                'kmers': set(kmer_counter.keys()),
                'counts': kmer_counter,
                'label': -1,
                'second_label': -1,
                'ambiental_label': ambiental_label
            })
            end_time = time.time()
            print(f"Added sample {id} with {len(kmer_counter)} k-mers. Done in {end_time - start_time:.4f} seconds")

            print(f"Size of added sample in KB: {(sys.getsizeof(kmer_counter.keys()) / 1024 - size_before / 1024):.5f}")
            print()

    return samples

def distance(a, b, metric='braycurtis'):
    """
    Computes the distance between two sets or lists of k-mers.

    Parameters
    ----------
    a : dict of {str: int}  
        The first collection of k-mers. Can be a list (with possible duplicates) or a set (unique k-mers).
    
    b : dict of {str: int}  
        The second collection of k-mers. Same format as `a`.
    
    metric : str, optional
        The distance metric to use. Supported values are:
        
        - `'jaccard'` : computes the Jaccard distance between the two sets.
        - `'braycurtis'` : computes the Bray-Curtis distance, which takes into account the counts of k-mers.
        
        Default is `'braycurtis'`.

    Returns
    -------
    float
        The computed distance value between the two k-mer collections.

    Raises
    ------
    ValueError
        If an unsupported metric is specified.
    """
    if metric == 'jaccard':
        intersection = len(a.keys() & b.keys())
        union = len(a.keys() | b.keys())
        return 1.0 - (intersection / union) if union > 0 else 1.0

    elif metric == 'braycurtis':
        intersection = set(a.keys()).intersection(b.keys())
        numerator = np.sum(min(a[k], b[k]) for k in intersection)
        all_keys = set(a.keys()).union(b.keys())
        a_vec = np.array([a[k] for k in all_keys])
        a_sum = np.sum(a_vec)
        b_vec = np.array([b[k] for k in all_keys])
        b_sum = np.sum(b_vec)
        if a_sum == 0 and b_sum == 0:
            print("Warning: Both sets are empty, returning distance 1.0")
        denominator = a_sum + b_sum
        return 1 - (numerator/denominator) if denominator > 0 else 1.0

    else:
        raise ValueError("Unsupported metric: choose 'jaccard' or 'braycurtis'")


def fastCLARANS(samples, k, numlocal, maxneighbor, metric='braycurtis'):
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
    how_many_cells_used = 0

    def get_cached_distance(i, j, metric):
        nonlocal how_many_cells_used
        if distance_matrix[i][j] == -1:
            d = distance(samples[i]['counts'], samples[j]['counts'], metric)
            distance_matrix[i][j] = d
            distance_matrix[j][i] = d
            how_many_cells_used = how_many_cells_used + 1
        return distance_matrix[i][j]
    
    def assign(centers, samples, metric='jaccard'):
        """
        Assigns each sample to the nearest and second nearest medoids.

        Parameters:
            centers (List[int]): Indices of current medoid samples in `samples`
            samples (List[Dict]): Samples with 'kmers', 'label', and 'second_label'
            metric (str): Distance metric to use
        """
        for sample in range(len(samples)):
            dists = [
                (get_cached_distance(sample, center_idx, metric), i)
                for i, center_idx in enumerate(centers)
            ]
            dists.sort()  # sort by distance --> k logk

            # Set label as nearest medoid's position in `centers`
            samples[sample]['label'] = dists[0][1]  # index in centers list
            samples[sample]['second_label'] = dists[1][1] if len(dists) > 1 else dists[0][1]


    # i = 1
    for _ in range(numlocal):
        print("local n. ", numlocal)

        # current = k centers at random from samples
        current = random.sample(range(n), k)

        # give a center to each sample in samples
        # calculate second medoid most near
        assign(current, samples, metric)

        # step 3) j = 1
        j = 1

        # step 4)
        while j <= maxneighbor:

            # take one center at random and take another from samples
            Om = random.choice(current)
            Op = random.choice([idx for idx in range(n) if idx not in current])

            # define:
            # Oj = sample in samples
            # Om = medoid to be changed
            # Op = new medoid to be added
            # Oj,2 = second medoid most near to Oj
            # Cjmp = cost of changing Om to Op for sample Oj

            TCmp = 0.0

            # calculate for each sample in samples:
            for idx_Oj, Oj in enumerate(samples):
                label_idx = current[Oj['label']]
                second_label_idx = current[Oj['second_label']]

                d_Oj_Om = get_cached_distance(idx_Oj, Om, metric)
                d_Oj_Op = get_cached_distance(idx_Oj, Op, metric)
                d_Oj_second = get_cached_distance(idx_Oj, second_label_idx, metric)

                # case 1: Oj belongs to cluster of Om, Oj more similar to Oj,2 than Op
                if label_idx == Om and d_Oj_second < d_Oj_Op:
                    Cjmp = d_Oj_second - d_Oj_Om

                # case 2: Oj belongs to cluster of Om, Oj more similar to Op than Oj,2
                elif label_idx == Om and d_Oj_Op < d_Oj_second:
                    Cjmp = d_Oj_Op - d_Oj_Om

                # case 3: Oj does not belong to cluster of Om, Oj more similar to Oj,2 than Op
                elif label_idx != Om and d_Oj_second < d_Oj_Op:
                    Cjmp = 0.0

                # case 4: Oj does not belong to cluster of Om, Oj more similar to Op than Oj,2
                else:
                    Cjmp = d_Oj_Op - d_Oj_second

                # TCmp = sum(Cjmp for each sample Oj in samples)
                TCmp += Cjmp

            # step 5)
            if TCmp < 0:
                # change Om to Op
                current.remove(Om)
                current.append(Op)

                # go to step 3)
                assign(current, samples, metric)
                j = 1
            else:
                # j = j + 1
                j += 1

        # if cost(current) < mincost:
        current_cost = sum(
            get_cached_distance(idx, current[sample['label']], metric)
            for idx, sample in enumerate(samples)
        )

        if current_cost < mincost:
            # mincost = cost(current)
            mincost = current_cost
            # bestnode = current
            bestnode = current.copy()
            print("found best node")

    # step 8)
    # i = i+1
    # if i > numlocal:
    #   return bestnode
    # else:
    #   go to step 2

    print(f"Number of distances calculated: {how_many_cells_used}")
    return bestnode


def main():
    dataset_folder_path = "/nfsd/bcb/bcbg/berleselor/datasets"               # Replace with actual path
    true_labels_path = "/nfsd/bcb/bcbg/meneghinma/input/true_labels.txt"     # Replace with actual path
    k_mer_size = 21                        # Adjust k-mer size as needed
    num_clusters = 10                      # Adjust number of medoids (k)
    numlocal = 5                           # Number of local minima to search
    maxneighbor = 10                       # Max neighbors per local search
    metric = 'braycurtis'                  # Or 'jaccard'

    print("Preparing samples...")
    samples = prepare_all_samples(dataset_folder_path, k_mer_size, true_labels_path)

    print(f"Running fastCLARANS on {len(samples)} samples...")
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()

    with open("cluster.txt", "w", encoding="utf-8") as f:
        for sample in samples:
            # Extract only specific fields (excluding 'kmers' and 'counts')
            values = [
                sample["id"],
                sample["label"],
                sample["second_label"],
                sample["ambiental_label"]
            ]
            line = "\n".join(str(v) for v in values)
            f.write(line + "\n\n")
    
    compute_evaluation_metrics(samples)

    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

if __name__ == "__main__":
    main()