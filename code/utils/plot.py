import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Example input
# dist_matrix_1 = your distance matrix (numpy 2D array)
# sample_ids_1 = list of sample names
# best_labels = cluster assignments for each sample

def plot_cluster_graph(dist_matrix, sample_ids, labels, threshold=5):
    G = nx.Graph()

    # Add nodes with cluster color as attribute
    for i, sid in enumerate(sample_ids):
        G.add_node(sid, cluster=labels[i])

    # Add edges if distance below threshold (e.g., samples are similar)
    for i in range(len(dist_matrix)):
        for j in range(i + 1, len(dist_matrix)):
            if dist_matrix[i][j] < threshold:
                G.add_edge(sample_ids[i], sample_ids[j], weight=1 - dist_matrix[i][j])

    # Assign colors to clusters
    unique_labels = list(set(labels))
    color_map = [plt.cm.tab10(unique_labels.index(labels[i])) for i in range(len(labels))]

    # Plot
    pos = nx.spring_layout(G, seed=42)  # for reproducible layout
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, edge_color='gray')
    plt.title("Cluster Graph Based on Jaccard Distances")
    plt.show()