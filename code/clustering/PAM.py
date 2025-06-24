import numpy as np

def run_pam(distance_matrix, k, max_iter=100):
    """
    Run the PAM algorithm on a distance matrix.
    
    Parameters:
    - distance_matrix: numpy array (n x n)
    - k: number of clusters
    - max_iter: max number of iterations in the SWAP phase
    
    Returns:
    - medoids: list of medoid indices
    - labels: array of cluster labels for each point
    """
    n = distance_matrix.shape[0]
    all_indices = set(range(n))

    # --- BUILD phase ---
    total_distances = np.sum(distance_matrix, axis=1)
    medoids = [np.argmin(total_distances)]

    while len(medoids) < k:
        U = list(all_indices - set(medoids))
        gains = []
        # O(n * n * k) complexity
        for i in U:
            gain = 0
            for j in U:
                if i == j:
                    continue
                Dj = np.min([distance_matrix[j][m] for m in medoids])
                Cji = max(Dj - distance_matrix[j][i], 0)
                gain += Cji
            gains.append((gain, i))
        _, best_i = max(gains)
        medoids.append(best_i)

    # --- SWAP phase ---
    def compute_D_E(medoids):
        Dp = np.full(n, np.inf)
        Ep = np.full(n, np.inf)
        for i in range(n):
            distances = [distance_matrix[i][m] for m in medoids]
            sorted_d = sorted(distances)
            Dp[i] = sorted_d[0]
            Ep[i] = sorted_d[1] if len(sorted_d) > 1 else sorted_d[0]
        return Dp, Ep

    # Initial Dp and Ep for each point
    Dp, Ep = compute_D_E(medoids)

    
    for _ in range(max_iter): # Not in original paper, but useful for convergence
        best_T = 0
        best_swap = None

        # O(n^2 * k) complexity)
        for i in medoids:
            for h in all_indices - set(medoids):
                Tih = 0
                for j in range(n):
                    if j == h:
                        continue
                    dji = distance_matrix[j][i]
                    djh = distance_matrix[j][h]

                    if dji > Dp[j]:
                        Kjih = min(djh - Dp[j], 0)
                    elif dji == Dp[j]:
                        if djh < Ep[j]:
                            Kjih = djh - Dp[j]
                        else:
                            Kjih = Ep[j] - Dp[j]
                    else:
                        Kjih = 0
                    Tih += Kjih

                if Tih < best_T:
                    best_T = Tih
                    best_swap = (i, h)

        if best_swap:
            i, h = best_swap
            medoids.remove(i)
            medoids.append(h)
            Dp, Ep = compute_D_E(medoids)
        else:
            break

    # Assign clusters
    labels = np.argmin(distance_matrix[:, medoids], axis=1)
    return labels, medoids, best_T
