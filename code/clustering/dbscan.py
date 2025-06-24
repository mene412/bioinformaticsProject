def dbscan(distance_matrix, eps, min_pts):
    """
    Perform DBSCAN clustering on a set of points.

    Parameters:
    ----------
    distance_matrix : 
    eps : float
        The maximum distance between two samples for one to be considered 
        as in the neighborhood of the other.
    min_pts : int
        The minimum number of points required to form a dense region (i.e., a cluster).

    Returns:
    -------
    None
        The function updates the 'cl_id' field of each point in-place.
    """
    set_of_points = []

    # Crea la lista di punti
    for i in range(len(distance_matrix)):
        set_of_points.append({
            'index': i,  # Salva l'indice originale
            'cl_id': None
        })

    cluster_id = next_id(0) #? non sarebbe meglio 0?
    for point in set_of_points:  # Itera direttamente sulla lista
        if point['cl_id'] == None:
            if expand_cluster(distance_matrix, set_of_points, point, cluster_id, eps, min_pts):
                cluster_id = next_id(cluster_id)
    
    return set_of_points

def expand_cluster(distance_matrix, set_of_points, point, cluster_id, eps, min_pts):
    seeds = region_query(distance_matrix, set_of_points, point, eps)
    if len(seeds) < min_pts:
        change_cl_id(set_of_points, [point], 'noise')
        return False
    else:
        change_cl_id(set_of_points, seeds, cluster_id)
        seeds = [p for p in seeds if p['index'] != point['index']]
        while seeds:
            current_p = seeds.pop(0)
            result = region_query(distance_matrix, set_of_points, current_p, eps)
            if len(result) >= min_pts:
                for resultP in result:
                    if resultP['cl_id'] is None or resultP['cl_id'] == 'noise':
                        if resultP['cl_id'] is None:
                            seeds.append(resultP)
                        change_cl_id(set_of_points, [resultP], cluster_id)
        return True


# funzione da rifare
def next_id(value):
    """
    Restituisce il prossimo ID numerico, oppure 'noise' se il valore in input è 'noise'.

    Parameters
    ----------
    value : str or int
        Il valore corrente da cui calcolare il prossimo ID.

    Returns
    -------
    int or str
        Il prossimo ID numerico oppure 'noise'.
    """
    if value == 'noise':
        return 'noise'
    return int(value) + 1

def region_query(distance_matrix, set_of_points, point, eps):
    """
    Returns the indices and cl_id of all points q such that the distance between `point` and `q`
    is less than or equal to `eps`.

    Parameters
    ----------
    distance_matrix : list of list
        A precomputed 2D distance matrix where distance_matrix[i][j] gives the distance between point i and j.
    set_of_points : list
        A list of dictionaries, where each dictionary represents a point with 'index' and 'cl_id'.
        Example: [{'index': 0, 'cl_id': None}, {'index': 1, 'cl_id': None}, ...]
    point : int
        The index of the reference point in the dataset.
    eps : float
        The epsilon threshold distance.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing 'index' and 'cl_id' of the neighboring points.
        Format: [{'index': 0, 'cl_id': None}, {'index': 1, 'cl_id': None}, ...]
    """
    neighbors_with_cl_id = []

    # Create a quick lookup for cl_id by index from set_of_points
    # This avoids iterating through set_of_points repeatedly inside the loop
    point_data_lookup = {p['index']: p for p in set_of_points}

    # Iterate through all possible points (columns in the distance_matrix row for 'point')
    for q_index in range(len(distance_matrix[point['index']])):
        # Check if the distance is within the epsilon
        if distance_matrix[point['index']][q_index] <= eps:
            # If the point exists in our lookup (it should, given how set_of_points is created)
            if q_index in point_data_lookup:
                # Retrieve the full point dictionary including its current cl_id
                # Make a copy to ensure we don't modify the original set_of_points indirectly
                neighbor_data = point_data_lookup[q_index].copy()
                neighbors_with_cl_id.append(neighbor_data)
            else:
                # This case should ideally not happen if set_of_points is correctly built
                # but it's good practice to handle unexpected scenarios or log them.
                print(f"Warning: Index {q_index} not found in set_of_points lookup.")

    return neighbors_with_cl_id

def change_cl_id(set_of_points, points, value):
    """
    For each point in points (in the format: [
        {
            'index': 0,
            'cl_id': None
        },
        {
            'index': 1,
            'cl_id': None
        },
        {
            'index': 2,
            'cl_id': None
        }
    ]) update it's value in set_of_points (through it's index) and set cl_id to value
    """
    for point_to_update in points:
        index_to_find = point_to_update['index']
        # Find the corresponding point in set_of_points using its index
        for main_point in set_of_points:
            if main_point['index'] == index_to_find:
                main_point['cl_id'] = value
                break # Once updated, move to the next point_to_update

def main():
    # - - DBSCAN - - 
    # Add to the list the cluster id, set to none intially
    dist_matrix = [[0,2,2],[2,0,4],[2,4,0]]
    print("- - DBSCAN - -")
    print(dbscan(dist_matrix, 1, 1))

if __name__ == "__main__":
    main()