import numpy as np

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
