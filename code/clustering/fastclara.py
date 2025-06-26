import numpy as np
from clustering.fastpam import run_fast_pam 

def run_fastCLARA(distance_matrix, k, max_iter=5, sample_size=None):
    """
    CLARA (Clustering LARge Applications) implementation.

    Parameters:
        distance_matrix (np.ndarray): Full distance matrix of all samples (NxN)
        k (int): Number of clusters
        max_iter (int): Number of random samples to draw (default 5)
        sample_size (int): Optional sample size, default is 40 + 2k

    Returns:
        best_labels (np.ndarray): Cluster assignment for each item in the full dataset
        best_medoids (List[int]): List of medoid indices in the full dataset
    """
    n = distance_matrix.shape[0]
    sample_size = sample_size or (40 + 2 * k)
    
    best_cost = float("inf")
    best_labels = None
    best_medoids = None

    for iter_num in range(max_iter):
        print(f"CLARA iteration {iter_num+1}/{max_iter}...")

        # Step 1: Draw sample indices
        sample_indices = np.random.choice(n, min(sample_size, n), replace=False)
        sample_indices = sorted(sample_indices)  # For consistency

        # Create a sample distance matrix. TODO not so needed if we use indices directly
        sample_dist = distance_matrix[np.ix_(sample_indices, sample_indices)]

        # Step 2: Run PAM on sample
        sample_labels, medoids, td = run_fast_pam(sample_dist, k)

        # Convert local indices back to global medoid indices
        sample_medoids = [sample_indices[i] for i in sample_labels]

        # Step 3: Assign each object in the full dataset to closest medoid
        full_labels = np.argmin(distance_matrix[:, sample_medoids], axis=1)

        # Step 4: Compute total cost
        current_cost = sum(distance_matrix[i, sample_medoids[full_labels[i]]] for i in range(n))

        # Keep the best result
        if current_cost < best_cost:
            best_cost = current_cost
            best_labels = full_labels
            best_medoids = sample_medoids

    return best_labels, best_medoids, best_cost
