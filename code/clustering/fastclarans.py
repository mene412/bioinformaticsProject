import numpy as np
import random
from clustering.fastPAM import fastpam_lab_build

def run_fastCLARANS(distance_matrix, k, numlocal=5, maxneighbor=10, max_iter=100):
    """
    CLARANS (Clustering Large Applications based on RANdomized Search)

    Parameters:
        distance_matrix (np.ndarray): Full distance matrix (NxN)
        k (int): Number of clusters
        numlocal (int): Number of local minima to search
        maxneighbor (int): Maximum number of neighbors to explore per local search
        max_iter (int): Max iterations per local search

    Returns:
        best_labels (np.ndarray): Cluster assignments
        best_medoids (List[int]): Indices of best medoids
        best_cost (float): Total cost (sum of distances to medoids)
    """
    n = distance_matrix.shape[0]
    all_indices = list(range(n))
    best_medoids = None
    best_cost = float('inf')
    best_labels = None

    for local_iter in range(numlocal):
        print(f"CLARANS local search {local_iter+1}/{numlocal}...")

        # --- Step 2: Initialize current medoids using FastPAM BUILD ---
        current_medoids = fastpam_lab_build(distance_matrix, k)
        current_medoids_set = set(current_medoids)

        # --- Initial Assignment ---
        Dp = np.full(n, np.inf)
        Ep = np.full(n, np.inf)
        nearest = np.full(n, -1, dtype=int)

        for i in range(n):
            dists = [(distance_matrix[i][m], m) for m in current_medoids]
            dists.sort()
            Dp[i], nearest[i] = dists[0]
            Ep[i] = dists[1][0] if len(dists) > 1 else dists[0][0]
        TD = np.sum(Dp)

        j = 0
        while j < maxneighbor:
            xj = random.choice(list(set(all_indices) - current_medoids_set))
            delta_TD = {m: -Dp[m] for m in current_medoids}

            for xo in range(n):
                if xo == xj:
                    continue
                doj = distance_matrix[xo][xj]
                dn = Dp[xo]
                ds = Ep[xo]
                mn = nearest[xo]

                if doj < dn:
                    for m in current_medoids:
                        if m != mn:
                            delta_TD[m] += doj - dn
                else:
                    delta_TD[mn] += min(doj, ds) - dn

            mi, min_delta = min(delta_TD.items(), key=lambda x: x[1])

            if min_delta < 0:
                # Accept neighbor
                current_medoids.remove(mi)
                current_medoids.append(xj)
                current_medoids_set = set(current_medoids)

                for i in range(n):
                    dists = [(distance_matrix[i][m], m) for m in current_medoids]
                    dists.sort()
                    Dp[i], nearest[i] = dists[0]
                    Ep[i] = dists[1][0] if len(dists) > 1 else dists[0][0]
                TD += min_delta
                j = 0
            else:
                j += 1

        if TD < best_cost:
            best_cost = TD
            best_medoids = current_medoids.copy()
            best_labels = np.argmin(distance_matrix[:, best_medoids], axis=1)

    return best_labels, best_medoids, best_cost