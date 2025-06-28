from Bio import SeqIO
from collections import Counter
import numpy as np
import random
import os
import time
import sys
from evaluation import compute_evaluation_metrics
import json

def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

'''
def find_ambiental_label(filename):
    """
    Parses the first header line of a FASTA file and extracts the value of the /region_1 field.

    Parameters:
        filename (str): Path to the FASTA file

    Returns:
        str or None: Value of /region_1 if found, else None
    """
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if '/region_1="' in line:
                    start = line.find('/region_1="') + len('/region_1="')
                    end = line.find('"', start)
                    return line[start:end]
                else:
                    return None  # /region_1 not found
    return None  # No header found

def find_id(filename):
    """
    Extracts the ID from the filename by removing the file extension and any leading directory path.

    Parameters:
        filename (str): Path to the FASTA file

    Returns:
        str: The ID extracted from the filename
    """
    base_name = os.path.basename(filename)
    return os.path.splitext(base_name)[0]  # Remove extension
'''

def prepare_all_samples(folder_path, k):
    """
    Reads one sample from each FASTA file in a folder.
    Each sample is the combined set of all k-mers from all sequences in that file.

    Parameters:
        folder_path (str): Path to a folder containing FASTA files
        k (int): k-mer size

    Returns:
        List[Dict]: List of samples, one per file
    """
    samples = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.fasta') or filename.endswith('.fa'):
            full_path = os.path.join(folder_path, filename)

            all_kmers = []
            
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
            id = os.path.splitext(filename)[0]
            ambiental_label = ""
            with open(os.path.join(folder_path, '..\\bioinformaticsProject\\code\\true_labels.txt'), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    label_id = parts[0]
                    if (label_id == id):
                        ambiental_label = " ".join(parts[1:]).lstrip(": ")
                        print(f"Ambiental label for {id}--> {ambiental_label}")
            end_time = time.time()
            print(f"Ambiental label for {id}: {ambiental_label}, found in {end_time - start_time:.4f} seconds")

            size_before = sys.getsizeof(kmer_counter.keys())
            start_time = time.time()
            samples.append({
                'id': id,  # Use filename without extension as ID
                'kmers': set(kmer_counter.keys()),  # or use the Counter directly if using Bray-Curtis
                'counts': kmer_counter,             # optional: store separately for braycurtis
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
    Computes distance between two sets or lists of k-mers.
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

    # mincost = infinity
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


    print(f"Number of distances calculated: {how_many_cells_used}")  # 3. Print the counter
    return bestnode


def main():
    folder_path = "C:\\Users\\Lorenzo Berlese\\Desktop\\metagenomics project\\alcuni_dataset_gos"  # Replace with actual path
    k_mer_size = 5                       # Adjust k-mer size as needed
    num_clusters = 5                      # Adjust number of medoids (k)
    numlocal = 5                          # Number of local minima to search
    maxneighbor = 10                      # Max neighbors per local search
    metric = 'braycurtis'                 # Or 'jaccard'

    print("Preparing samples...")
    samples = prepare_all_samples(folder_path, k_mer_size)

    '''
    for i, sample in enumerate(samples):
        with open(f"samples{i}.txt", "w", encoding="utf-8") as f:
            output_dict = {
                'id': sample['id'],
                'kmers': list(sample['kmers']),  # Convert set to list
                'counts': dict(sample['counts']),  # Convert Counter to dict
                'label': sample['label'],
                'second_label': sample['second_label'],
                'ambiental_label': sample['ambiental_label']
            }
            json.dump(output_dict, f, indent=2)
    '''

    print(f"Running fastCLARANS on {len(samples)} samples...")
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()

    with open("cluster.txt", "w", encoding="utf-8") as f:
        for sample in samples:
            # Join only the values as strings, tab-separated
            line = "\n".join(str(v) for v in sample.values())
            f.write(line + "\n\n")
    
        

    compute_evaluation_metrics(samples)

    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

if __name__ == "__main__":
    main()