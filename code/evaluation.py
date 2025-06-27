from collections import defaultdict, Counter

def find_majority_label(clustered_samples):
    """
    Based on the label (the cluster id), returns the most present ambient_label present in that cluster
    
    Input:
    - clustered_samples: list of dict, example: 
        [{
            'id': id,  # Use filename without extension as ID
            'kmers': set(all_kmers),
            'label': -1,
            'second_label': -1,
            'ambiental_label': ambiental_label
        }, ...]
    Return:
    - majority_labels: list of dicts, example:
        [
            {
                'cl_id': 1,
                'majority_label': ambiental_label
            },
            ...
        ]
    """
    # Group ambiental_labels by cluster label
    clusters = defaultdict(list)
    for sample in clustered_samples:
        cl_id = sample['label']
        amb_label = sample['ambiental_label']
        clusters[cl_id].append(amb_label)
    
    # Find majority label for each cluster
    majority_labels = []
    for cl_id, labels in clusters.items():
        most_common_label, _ = Counter(labels).most_common(1)[0]
        majority_labels.append({
            'cl_id': cl_id,
            'majority_label': most_common_label
        })
    
    return majority_labels

samples = [
    {'id': 'a', 'kmers': {'A'}, 'label': 0, 'second_label': -1, 'ambiental_label': 'indoor'},
    {'id': 'b', 'kmers': {'B'}, 'label': 0, 'second_label': -1, 'ambiental_label': 'indoor'},
    {'id': 'c', 'kmers': {'C'}, 'label': 1, 'second_label': -1, 'ambiental_label': 'outdoor'},
    {'id': 'd', 'kmers': {'D'}, 'label': 1, 'second_label': -1, 'ambiental_label': 'outdoor'},
    {'id': 'e', 'kmers': {'E'}, 'label': 1, 'second_label': -1, 'ambiental_label': 'indoor'},
]

print(find_majority_label(samples))

