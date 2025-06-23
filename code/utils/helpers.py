from collections import defaultdict

def print_cluster_results(sample_ids, labels):
    cluster_map = defaultdict(list)
    for sid, label in zip(sample_ids, labels):
        cluster_map[label].append(sid)
    for label, members in cluster_map.items():
        print(f"Cluster {label}: {members}")