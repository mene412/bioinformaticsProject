import numpy as np
import random
import time
from evaluation import compute_evaluation_metrics
import pandas as pd
from scipy import stats

def fastCLARANS(samples, k, numlocal, maxneighbor, distance_matrix):
   
    n = len(samples)
    mincost = float('inf')
    bestnode = None

    run_usage_matrix = np.zeros((n, n), dtype=np.uint8)  # Local usage matrix

    how_many_cells_used = 0

    def get_cached_distance(i, j):
        nonlocal how_many_cells_used
        if run_usage_matrix[i, j] == 1:
            return distance_matrix[i, j]
        run_usage_matrix[i, j] = 1
        how_many_cells_used += 1        
        
        return distance_matrix[i, j]
    
    def assign(centers, samples):
        cost = 0.0
        for idx, sample in enumerate(samples):
            dists = []  # list of tuples (distance, index of medoid in centers)
            for i, center_idx in enumerate(centers):
                dists.append((get_cached_distance(idx, center_idx), i)) # index of medoid in centers, center_idx is the index in `samples`

            dists.sort()  # sort by distance --> k logk --> could be done in less time
            # Set label as nearest medoid's position in `centers`
            sample['center'] = dists[0][1]  # index in centers list
            cost += dists[0][0]  # add distance to cost
            sample['second'] = dists[1][1] if len(dists) > 1 else dists[0][1]
        return cost

    start_time = time.time()
    for _ in range(numlocal):   
        # current = k centers at random from samples
        current = random.sample(range(n), k)
        # give a center to each sample in samples
        # calculate second nearest medoid
        start_time = time.time()
        current_cost = assign(current, samples)
        end_time = time.time()

        num_swaps = 0
        
        # step 3)
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

            d_Op_to_center = get_cached_distance(Op, current[samples[Op]['center']])  # distance from Op to its medoid
            # print("Update all delta_TS...")
            for i in range(k):
                delta_TS[i] -= d_Op_to_center # update delta_TS for all medoids

            # calculate for neighbours in samples until maxneighbour:
            for idx_Oj, Oj in enumerate(samples):
                if idx_Oj == Op:
                    continue
                d_Oj_Op = get_cached_distance(idx_Oj, Op) # distance from sample Oj to new medoid Op
                
                current_label = Oj['center'] # index of the nearest medoid in `samples`
                idx_Oj_nearest = current[current_label]
                idx_Oj_second = current[Oj['second']] # index of the second nearest medoid in `samples`
                d_Oj_nearest = get_cached_distance(idx_Oj, idx_Oj_nearest) # distance from sample Oj to medoid Om
                d_Oj_second = get_cached_distance(idx_Oj, idx_Oj_second) # distance from sample Oj to second medoid Oj,2

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

            if delta_TS[min_index] < 0:
                current[min_index] = Op
                current_cost += delta_TS[min_index]
                assign(current, samples)
                j = 1
                num_swaps += 1
                    
            else:
                j += 1

        if current_cost < mincost:
            mincost = current_cost
            bestnode = current.copy()
    end_time = time.time()

    return bestnode, how_many_cells_used, end_time - start_time


def main():
    true_labels_path = "C:/Users/Lorenzo Berlese/Desktop/run locale/true_labels.txt"
    distance_matrix_path = "C:/Users/Lorenzo Berlese/Downloads/mat_abundance_braycurtis.csv/mat_abundance_braycurtis.csv"

    num_clusters = 8
    numlocal = 3

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

    for maxneighbor in [10, 20, 30]:
        macro_F1s = []
        micro_F1s = []
        weighted_F1s = []
        macro_precisions = []
        micro_precisions = []
        weighted_precisions = []
        macro_recalls = []
        micro_recalls = []
        weighted_recalls = []
        cells_used = []
        runtimes = []

        for iteration in range(100):
            output_file = f"C:/Users/Lorenzo Berlese/Desktop/the end is near/cluster{maxneighbor}.txt"
            start_time = time.time()    
            medoids, cells, runtime = fastCLARANS(samples, num_clusters, numlocal, maxneighbor, distance_matrix)
            end_time = time.time()
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Medoids: {medoids}, Runtime: {end_time - start_time:.2f}s\n\n")

            label_metrics, macro_avg, micro_avg, weighted_avg = compute_evaluation_metrics(samples)
            macro_F1s.append(macro_avg['macro_F1score'])
            micro_F1s.append(micro_avg['micro_F1score'])
            weighted_F1s.append(weighted_avg['weighted_F1score'])
            macro_precisions.append(macro_avg['macro_precision'])
            micro_precisions.append(micro_avg['micro_precision'])
            weighted_precisions.append(weighted_avg['weighted_precision'])
            macro_recalls.append(macro_avg['macro_recall'])
            micro_recalls.append(micro_avg['micro_recall'])
            weighted_recalls.append(weighted_avg['weighted_recall'])
            cells_used.append(cells)
            runtimes.append(runtime)
        
        all_metrics = {
            'macro_F1s': macro_F1s,
            'micro_F1s': micro_F1s,
            'weighted_F1s': weighted_F1s,
            'macro_precisions': macro_precisions,
            'micro_precisions': micro_precisions,
            'weighted_precisions': weighted_precisions,
            'macro_recalls': macro_recalls,
            'micro_recalls': micro_recalls,
            'weighted_recalls': weighted_recalls,
            'cells_used': cells_used,
            'runtimes': runtimes
        }
        
        confidence = 0.95
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n- - - FINAL RESULTS FOR maxneighbor = {maxneighbor} - - -\n")
            for metric_name, metric_values in all_metrics.items():
                data = np.array(metric_values)
                mean_value = np.mean(data)
                std_value = np.std(data, ddof=1)
                if std_value == 0:
                    confidence_interval = (mean_value, mean_value)
                else:
                    df = len(data) - 1
                    confidence_interval = stats.t.interval(confidence, df, loc=mean_value, scale=std_value / np.sqrt(len(data)))
            

                f.write(f"{metric_name} - Mean: {mean_value:.4f}, Std: {std_value:.4f}, "
                        f"Confidence Interval: {confidence_interval[0]:.4f} to {confidence_interval[1]:.4f}\n")

    
if __name__ == "__main__":
    main()