from Bio import SeqIO
from collections import Counter
import numpy as np
import random
import os
import time
import sys
from evaluation import compute_evaluation_metrics
import tracemalloc



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

# List of files without sequences but only metadata
#todo try to re-download them with specific single wget after checking the link
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
        true_labels_path : str
            Path to the `.txt` file with labels in the format `GOSXX : label`, one entry for line

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

    # tracemalloc.start()

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
                if num_reads % 10000 == 0:
                    print(f"Processed {num_reads} reads so far...")
                    # print(tracemalloc.get_traced_memory())
                kmer_counter.update(valid_kmers(str(record.seq), k))
                
            end_time = time.time()


            base_name = os.path.splitext(filename)[0]  # "GOS01"
            printfile = base_name + "_done.txt"
            print(f"Found {len(kmer_counter)} k-mers. Done in {end_time - start_time:.4f} seconds")
            with open(printfile, "w", encoding="utf-8") as f:
                    f.write("done\n")

            
            start_time = time.time()
            id = os.path.splitext(filename)[0] # Use filename without extension as ID
            ambiental_label = ""
            
            with open(os.path.join(true_labels_path), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    label_id = parts[0]
                    if (label_id == id):
                        ambiental_label = " ".join(parts[1:]).lstrip(": ")
                        print(f"Ambiental label for {id}--> {ambiental_label}")
            end_time = time.time()
            print(f"Ambiental label for {id}: {ambiental_label}, found in {end_time - start_time:.4f} seconds")

            '''
            pre_size = sys.getsizeof(kmer_counter)
            for key in kmer_counter:
                pre_size += sys.getsizeof(key) + sys.getsizeof(kmer_counter[key])
            # size_before = sys.getsizeof(kmer_counter.keys())
            '''
            start_time = time.time()
            samples.append({
                'id': id,
                #'kmers': set(kmer_counter.keys()),
                'counts': kmer_counter,
                'label': -1,
                'second_label': -1,
                'ambiental_label': ambiental_label
            })
            end_time = time.time()
            print(f"Added sample {id} with {len(kmer_counter)} k-mers. Done in {end_time - start_time:.4f} seconds")

            # print(f"Size of added sample in KB: {(pre_size):.5f}")
            print()

    return samples

def distance(a, b, metric='braycurtis'):
    """
    Computes the distance between two dicts of k-mers, in the format `{k-mer : counts}`.

    Parameters
    ----------
    a : dict of {str: int}  
        The first collection of k-mers.
    
    b : dict of {str: int}  
        The second collection of k-mers.
    
    metric : str, optional
        The distance metric to use. Supported values are:
        
        - `'jaccard'` : computes the Jaccard distance between the two sets.
        - `'braycurtis'` : computes the Bray-Curtis distance, which takes into account the counts of k-mers.
        
        Default is `'braycurtis'`.

    Returns
    ----------
    float
        The computed distance value between the two k-mer collections.

    Raises
    ----------
    ValueError
        If an unsupported metric is specified.
    """
    if metric == 'jaccard':
        intersection = len(a.keys() & b.keys())
        union = len(a.keys() | b.keys())
        return 1.0 - (intersection / union) if union > 0 else 1.0

    elif metric == 'braycurtis':
        intersection = set(a.keys()).intersection(b.keys())
        numerator = sum(min(a[k], b[k]) for k in intersection)
        all_keys = set(a.keys()).union(b.keys())
        a_vec = np.array([a[k] for k in all_keys])
        a_sum = np.sum(a_vec)
        b_vec = np.array([b[k] for k in all_keys])
        b_sum = np.sum(b_vec)
        if a_sum == 0 and b_sum == 0:
            print("Warning: Both sets are empty, returning distance 1.0")
        denominator = a_sum + b_sum
        return 1 - 2 * (numerator/denominator) if denominator > 0 else 1.0

    else:
        raise ValueError("Unsupported metric: choose 'jaccard' or 'braycurtis'")


def fastCLARANS(samples, k, numlocal, maxneighbor, metric='braycurtis'):
    """
    CLARANS clustering algorithm with in-place label assignment and distance caching.

    Parameters
    ----------
        samples (List[Dict]) 
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
        k (int)
            Number of clusters
        numlocal (int)
            Number of local minima to search
        maxneighbor (int)
            Maximum neighbors to explore per local search
        metric (str)
            Distance metric: 'jaccard' or 'braycurtis'

    Returns
    ----------
        List[int]
            Indices of best medoids
    """
    print(f"Running fastCLARANS with k={k}, numlocal={numlocal}, maxneighbor={maxneighbor}, metric={metric}")
    n = len(samples)
    mincost = float('inf')
    bestnode = None

    # Initialize distance matrix with -1
    distance_matrix = np.full((n, n), -1.0)
    how_many_cells_used = 0

    def get_cached_distance(i, j, metric):
        """
        Retrieves the cached distance between two samples, or computes and caches it if not already computed.

        This function checks if the distance between samples `i` and `j` has already been computed and stored 
        in the `distance_matrix`. If not, it calculates the distance using the provided `metric`, stores it 
        symmetrically in the matrix, and updates the counter `how_many_cells_used`.

        Parameters
        ----------
        i : int
            Index of the first sample in the `samples` list.
        j : int
            Index of the second sample in the `samples` list.
        metric : str
            The distance metric to use ('jaccard' or 'braycurtis').

        Returns
        -------
        float
            The distance between samples `i` and `j`.
        """
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

        Parameters
        ----------
            centers (List[int])
                Indices of current medoid samples in `samples`
            samples (List[Dict])
                Samples with 'kmers', 'label', and 'second_label'
            metric (str)
                Distance metric to use
        """
        cost = 0.0
        for sample in range(len(samples)):
            dists = [
                (get_cached_distance(sample, center_idx, metric), i)
                for i, center_idx in enumerate(centers) # index of medoid in centers, center_idx is the index in `samples`
            ]
            dists.sort()  # sort by distance --> k logk --> could be done in less time

            # Set label as nearest medoid's position in `centers`
            samples[sample]['label'] = dists[0][1]  # index in centers list
            cost += dists[0][0]  # add distance to cost
            samples[sample]['second_label'] = dists[1][1] if len(dists) > 1 else dists[0][1]
        
        return cost


    for _ in range(numlocal):
        # current = k centers at random from samples
        current = random.sample(range(n), k)

        # give a center to each sample in samples
        # calculate second medoid most near
        print(f"Assigning medoids for iteration {_}...")
        current_cost = assign(current, samples, metric)

        # step 3) j = 1
        j = 1

        best_delta_TS = 0.0  # best delta_TS found so far
        best_medoid_to_remove = None  # best medoid to change
        best_new_medoid = None  # best new medoid to add

        # step 4)
        while j <= maxneighbor:
            print("\t maxneighbor:", maxneighbor, "j:", j)

            # define:
            # Oj = sample in samples
            # Om = medoid to be changed
            # Op = new medoid to be added
            # Oj,2 = second medoid most near to Oj
            # Cjmp = cost of changing Om to Op for sample Oj


            delta_TS = [0.0] * k  # delta_TS[i] = loss function if swapping Op with medoid m[i]

            # take one center at random and take another from samples
            # Om = random.choice(current)  # medoid to be changed
            Op = random.choice([idx for idx in range(n) if idx not in current]) # new medoid to be added

            nearest_idx = current[samples[Op]['label']]  # index of the nearest medoid in `samples`

            d_nearest = get_cached_distance(nearest_idx, Op, metric)  # distance from Op to its medoid
            print("Update all delta_TS...")
            for i in range(k):
                delta_TS[i]  = delta_TS[i] - d_nearest # update delta_TS for all medoids

            # calculate for neighbours in samples until maxneighbour:
            for idx_Oj, Oj in enumerate(samples):
                d_Oj_Op = get_cached_distance(idx_Oj, Op, metric) # distance from sample Oj to new medoid Op
                
                label_idx = current[Oj['label']] # index of the nearest medoid in `samples`
                second_label_idx = current[Oj['second_label']] # index of the second nearest medoid in `samples`
                d_Oj_Om = get_cached_distance(idx_Oj, label_idx, metric) # distance from sample Oj to medoid Om
                d_Oj_second = get_cached_distance(idx_Oj, second_label_idx, metric) # distance from sample Oj to second medoid Oj,2

                delta_TS[Oj['label']] += min(d_Oj_Op, d_Oj_second) - d_Oj_Om  # update delta_TS for the medoid of Oj
                
                if d_Oj_Op < d_Oj_Om:
                    for i in range(k):
                        if i != Oj['label']:
                            delta_TS[i] += d_Oj_Op - d_Oj_Om

            min_delta_TS = 0
            min_index = -1
            print("Finding best medoid to change...")
            for i in range(k):
                if delta_TS[i] < min_delta_TS:
                    min_delta_TS = delta_TS[i]
                    min_index = i


            if delta_TS[min_index] < 0:
                print("Found better medoid to change...")
                best_delta_TS = delta_TS[min_index]
                current_cost = current_cost + best_delta_TS

                best_medoid_to_remove = current[min_index]  # index of the medoid to change
                best_new_medoid = Op
                current.remove(best_medoid_to_remove)
                current.append(best_new_medoid)

                assign(current, samples, metric)
                j = 1
            else:
                j += 1
            if best_delta_TS >= 0:
                break

        '''  
        current_cost = sum(
            get_cached_distance(idx, current[sample['label']], metric)
            for idx, sample in enumerate(samples)
        )
        '''

        if current_cost < mincost:
            mincost = current_cost
            bestnode = current.copy()
            print("found new best node with cost:", mincost)  


    print(f"Number of distances calculated: {how_many_cells_used}")    
    return bestnode


def main():
    # dataset_folder_path = "C:\\Users\\Lorenzo Berlese\\Desktop\\metagenomics project\\alcuni_dataset_gos"               # Replace with actual path
    # true_labels_path = "C:\\Users\\Lorenzo Berlese\\Desktop\\metagenomics project\\true_labels.txt"     # Replace with actual path

    dataset_folder_path = "/nfsd/bcb/bcbg/berleselor/datasets"               # Replace with actual path
    true_labels_path = "/nfsd/bcb/bcbg/meneghinma/input/true_labels.txt"     # Replace with actual path

    k_mer_size = 21                        # Adjust k-mer size as needed
    num_clusters = 10                      # Adjust number of medoids (k)
    numlocal = 3                           # Number of local minima to search
    maxneighbor = 10                       # Max neighbors per local search
    metric = 'braycurtis'                  # Or 'jaccard'

    print("Preparing samples...")
    samples = prepare_all_samples(dataset_folder_path, k_mer_size, true_labels_path)

    print(f"Running fastCLARANS on {len(samples)} samples...")
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()

    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")
    
    compute_evaluation_metrics(samples)

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 20
    maxneighbor = 20
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 30
    maxneighbor = 30
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 40
    maxneighbor = 40
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 50
    maxneighbor = 50
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 60
    maxneighbor = 60
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

    # maxneighbours = 65
    maxneighbor = 65
    start_time = time.time()
    medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, metric)
    end_time = time.time()
    compute_evaluation_metrics(samples)
    print(f"\nBest medoids (sample indices): {medoids}")
    print(f"Runtime: {end_time - start_time:.8f} seconds")

    name = "cluster" + str(maxneighbor) + ".txt"
    with open(name, "w", encoding="utf-8") as f:
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

if __name__ == "__main__":
    main()