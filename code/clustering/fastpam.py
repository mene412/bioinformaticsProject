import random
import math
import numpy as np

def fastpam_lab_build(distance_matrix, k, seed=42):
    """
    FastPAM LAB: approximate BUILD initialization using subsampling
    """
    random.seed(seed)
    n = distance_matrix.shape[0]
    m = int(10 + math.sqrt(n))  # subsample size
    all_indices = list(range(n))
    medoids = []

    # Step 1: select first medoid using sum of distances over subsample
    S = random.sample(all_indices, min(m, n))
    TD_min = float('inf')
    m1 = None

    for xj in S:
        TDj = sum(distance_matrix[xj][xo] for xo in S if xo != xj)
        if TDj < TD_min:
            TD_min = TDj
            m1 = xj

    medoids.append(m1)

    # Step 2: select remaining k-1 medoids
    for _ in range(1, k):
        remaining = list(set(all_indices) - set(medoids))
        S = random.sample(remaining, min(m, len(remaining)))
        best_delta = float('inf')
        best_candidate = None

        for xj in S:
            delta = 0
            for xo in S:
                if xo == xj:
                    continue
                current_d = min(distance_matrix[xo][m] for m in medoids)
                new_d = distance_matrix[xo][xj]
                if new_d < current_d:
                    delta += new_d - current_d
            if delta < best_delta:
                best_delta = delta
                best_candidate = xj

        if best_candidate is not None:
            medoids.append(best_candidate)

    return medoids


def run_fast_pam(distance_matrix, k, max_iter=100):
    n = distance_matrix.shape[0]
    all_indices = set(range(n))

    # --- BUILD phase ---
    medoids = fastpam_lab_build(distance_matrix, k)

    # Initial assignment
    Dp = np.full(n, np.inf)
    Ep = np.full(n, np.inf)
    nearest = np.full(n, -1, dtype=int)

    for i in range(n):
        distances = [(distance_matrix[i][m], m) for m in medoids]
        distances.sort()
        Dp[i], nearest[i] = distances[0]
        Ep[i] = distances[1][0] if len(distances) > 1 else distances[0][0]

    # --- OPTIMIZED SWAP phase ---
    TD = np.sum(Dp)
    for _ in range(max_iter):
        delta_TD_best = 0
        m_star, x_star = None, None

        for xj in all_indices - set(medoids):  # Non-medoids
            delta_TD = {m: -Dp[m] for m in medoids}  # line 5

            for xo in range(n):
                if xo == xj:
                    continue
                doj = distance_matrix[xo][xj]  # line 7
                dn = Dp[xo]
                ds = Ep[xo]
                mn = nearest[xo]

                if doj < dn:  # line 10
                    for mi in medoids:
                        if mi != mn:
                            delta_TD[mi] += doj - dn
                else:
                    delta_TD[mn] += min(doj, ds) - dn

            # Select best medoid i to replace
            mi, min_delta = min(delta_TD.items(), key=lambda x: x[1])
            if min_delta < delta_TD_best:
                delta_TD_best = min_delta
                m_star = mi
                x_star = xj

        if delta_TD_best >= 0:
            break  # No improvement
        else:
            # Perform the swap
            medoids.remove(m_star)
            medoids.append(x_star)

            # Recompute Dp, Ep, nearest
            for i in range(n):
                distances = [(distance_matrix[i][m], m) for m in medoids]
                distances.sort()
                Dp[i], nearest[i] = distances[0]
                Ep[i] = distances[1][0] if len(distances) > 1 else distances[0][0]

            TD += delta_TD_best

    # Assign labels
    labels = np.argmin(distance_matrix[:, medoids], axis=1)
    return labels, medoids, TD