from Bio import SeqIO
from collections import Counter
import numpy as np
import random
import os
import time
from evaluation import compute_evaluation_metrics
import math
import pandas as pd

def fastCLARANS(samples, k, numlocal, maxneighbor, metric, distance_matrix):
    # print(f"Running fastCLARANS with k={k}, numlocal={numlocal}, maxneighbor={maxneighbor}, metric={metric}")
    file_fastCLARANS = "C:/Users/Lorenzo Berlese/Desktop/the end is near/fastCLARANS_" + str(maxneighbor) + ".txt"
    with open(file_fastCLARANS, "w", encoding="utf-8") as f:
        f.write(f"start\n")
    
    n = len(samples)
    combinations = math.comb(n, k)
    mincost = float('inf')
    bestnode = None

    run_usage_matrix = np.zeros((n, n), dtype=np.uint8)  # Local usage matrix

    how_many_cells_used = 0

    def get_cached_distance(i, j, metric):
        nonlocal how_many_cells_used
        if run_usage_matrix[i, j] == 1:
            return distance_matrix[i, j]
        run_usage_matrix[i, j] = 1
        how_many_cells_used += 1        
        
        return distance_matrix[i, j]
    
    def assign(centers, samples, metric='braycurtis'):
        cost = 0.0
        for idx, sample in enumerate(samples):
            #print(f"Assigning medoids for sample {idx}...")
            dists = []  # list of tuples (distance, index of medoid in centers)
            for i, center_idx in enumerate(centers):
                #print("center_idx = ", center_idx)
                dists.append((get_cached_distance(idx, center_idx, metric), i)) # index of medoid in centers, center_idx is the index in `samples`

            dists.sort()  # sort by distance --> k logk --> could be done in less time
            #print("dists = ", dists)
            # Set label as nearest medoid's position in `centers`
            #print("prima: ", sample['center'])
            sample['center'] = dists[0][1]  # index in centers list
            #print("dopo: ", sample['center'])
            cost += dists[0][0]  # add distance to cost
            sample['second'] = dists[1][1] if len(dists) > 1 else dists[0][1]
        return cost

    start_time = time.time()
    for _ in range(numlocal):   
        # current = k centers at random from samples
        current = random.sample(range(n), k)
        #print(f"Current medoids: {current}")

        # give a center to each sample in samples
        # calculate second nearest medoid
        print(f"Assigning medoids for iteration {_}...")
        with open(file_fastCLARANS, "a", encoding="utf-8") as f:
            f.write("going to assign medoids\n")
        start_time = time.time()
        current_cost = assign(current, samples, metric)
        #print("current_cost = ", current_cost)
        end_time = time.time()

        with open(file_fastCLARANS, "a", encoding="utf-8") as f:
            f.write(f"Medoids assigned in {end_time - start_time:.2f}s\n")
            f.write(f"local n: {_}\n")
        num_swaps = 0

        # step 3) j = 1
        j = 1
        
        # step 4)
        while j <= maxneighbor:
            # define:
            # Oj = sample in samples
            # Om = medoid to be changed
            # Op = new medoid to be added
            # Oj,2 = second medoid most near to Oj
            # Cjmp = cost of changing Om to Op for sample Oj
            count_negatives = [0] * k

            delta_TS = [0.0] * k  # delta_TS[i] = loss function if swapping Op with medoid m[i]

            # take one center at random and take another from samples
            # Om = random.choice(current)  # medoid to be changed
            Op = random.choice([idx for idx in range(n) if idx not in current]) # new medoid to be added

            d_Op_to_center = get_cached_distance(Op, current[samples[Op]['center']], metric)  # distance from Op to its medoid
            # print("Update all delta_TS...")
            for i in range(k):
                delta_TS[i] -= d_Op_to_center # update delta_TS for all medoids

            # calculate for neighbours in samples until maxneighbour:
            for idx_Oj, Oj in enumerate(samples):
                if idx_Oj == Op:
                    continue
                d_Oj_Op = get_cached_distance(idx_Oj, Op, metric) # distance from sample Oj to new medoid Op
                
                current_label = Oj['center'] # index of the nearest medoid in `samples`
                idx_Oj_nearest = current[current_label]
                idx_Oj_second = current[Oj['second']] # index of the second nearest medoid in `samples`
                d_Oj_nearest = get_cached_distance(idx_Oj, idx_Oj_nearest, metric) # distance from sample Oj to medoid Om
                d_Oj_second = get_cached_distance(idx_Oj, idx_Oj_second, metric) # distance from sample Oj to second medoid Oj,2

                value = (min(d_Oj_Op, d_Oj_second) - d_Oj_nearest)
                if value < 0:
                    count_negatives[current_label] += 1
                delta_TS[current_label] += value  # update delta_TS for the medoid of Oj
                
                if d_Oj_Op < d_Oj_nearest:
                    for i in range(k):
                        if i != current_label:
                            value = (d_Oj_Op - d_Oj_nearest)
                            if value < 0:
                                count_negatives[i] += 1
                            delta_TS[i] += value

            min_index = np.argmin(delta_TS)
            #print("count_negatives = ", count_negatives, " delta_TS = ", delta_TS[min_index])

            if delta_TS[min_index] < 0:
                # print(f"Swap {num_swaps}: New medoid {Op} replaces medoid {current[min_index]} with cost change {delta_TS[min_index]:.5f}")
                # best_medoid = current[min_index]
                #print("scambio questo: ", current[min_index], " con questo: ", Op)
                current[min_index] = Op
                #print("ora è: ", current[min_index])
                #print("dentro. current_cost = ", current_cost)
                #print("delta_TS[min_index] = ", delta_TS[min_index])
                current_cost += delta_TS[min_index]
                #print("current_cost after change with delta= ", current_cost)
                #print("current_cost sanity check = ", assign(current, samples, metric))
                #print(f"Current cost: {current_cost:.5f}")
                assign(current, samples, metric)
                j = 1
                num_swaps += 1
                if num_swaps % 100 == 0:
                    with open(file_fastCLARANS, "a", encoding="utf-8") as f:
                        f.write(f"Swap {num_swaps} with cost change {delta_TS[min_index]:.5f}\n")
                        f.write(f"Current cost: {current_cost:.5f}\n")
                if num_swaps > combinations:
                    with open(file_fastCLARANS, "a", encoding="utf-8") as f:
                        f.write(f"reached max num_swaps: {num_swaps}\n")
                    
            else:
                j += 1

        '''  
        current_cost = sum(
            get_cached_distance(idx, current[sample['label']], metric)
            for idx, sample in enumerate(samples)
        )
        '''

        if current_cost < mincost:
            mincost = current_cost
            bestnode = current.copy()
    end_time = time.time()

    with open(file_fastCLARANS, "a", encoding="utf-8") as f:
            f.write(f"Number of distances computed in this run: {how_many_cells_used}\n")
            f.write(f"Runtime: {end_time - start_time:.2f}s\n")
    return bestnode


def main():
    #dataset_folder_path = "C:/Users/Lorenzo Berlese/Desktop/metagenomics project/alcuni_dataset_gos"
    true_labels_path = "C:/Users/Lorenzo Berlese/Desktop/run locale/true_labels.txt"
    distance_matrix_path = "C:/Users/Lorenzo Berlese/Downloads/mat_abundance_braycurtis.csv/mat_abundance_braycurtis.csv"

    #dataset_folder_path = "/nfsd/bcb/bcbg/berleselor/datasets" 
    # true_labels_path = "/nfsd/bcb/bcbg/berleselor/inputs/true_labels.txt"
    #distance_matrix_path = "/nfsd/bcb/bcbg/meneghinma/output_simka/mat_abundance_braycurtis.csv.gz"

    k_mer_size = 6
    num_clusters = 8
    numlocal = 3
    metrics = 'braycurtis'


    df = pd.read_csv(distance_matrix_path, index_col=0, sep=';')

    # Convert to NumPy array
    distance_matrix = df.values.astype(np.float32)

    samples = []

    with open(true_labels_path, 'r') as f:
        for i, line in enumerate(f):
            name = line.strip().split()[0]
            label = line.split(":")[1].strip()
            samples.append({
                'id': name,
                'center': -1,  # or some default value
                'second': -1,  # or some default value
                'ambiental_label': label,
            })
    
    for maxneighbor in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 448]:

        output_file = f"C:/Users/Lorenzo Berlese/Desktop/the end is near/cluster{maxneighbor}.txt"
        #print(f"\nRunning fastCLARANS with maxneighbor = {maxneighbor}")
        start_time = time.time()    
        medoids = fastCLARANS(samples, num_clusters, numlocal, maxneighbor,
                                metrics, distance_matrix)
        end_time = time.time()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Medoids: {medoids}, Runtime: {end_time - start_time:.2f}s\n\n")

        label_metrics, macro_avg, micro_avg = compute_evaluation_metrics(samples)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write('- - - RESULTS - - -\n')
            for label, m in label_metrics.items():
                f.write(f"Label '{label}': TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, "
                    f"Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1score={m['F1score']:.2f}\n")
            f.write(f"Macro-Averaged Metrics: {macro_avg}\n")
            f.write(f"Micro-Averaged Metrics: {micro_avg}\n\n\n")


        with open(output_file, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(f"id of sample: {s['id']}\nCenter given: {s['center']}\nSecond center: {s['second']}\nLabel: {s['ambiental_label']}\n\n")


if __name__ == "__main__":
    main()