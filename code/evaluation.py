from collections import defaultdict, Counter

def find_majority_label(clustered_samples):
    """
    Determines the most frequent 'ambiental_label' within each cluster.

    Parameters
    ----------
    clustered_samples : list of dict
        A list of sample dictionaries, each containing:
            - 'id': str, identifier of the sample
            - 'kmers': set, set of k-mers
            - 'label': int, cluster ID
            - 'second_label': int, optional secondary label
            - 'ambiental_label': str, the true environmental label (e.g., Caribbean Sea, Galapagos Islands)

    Returns
    -------
    majority_labels : list of dict
        A list of dictionaries where each dictionary contains:
            - 'cl_id': int, the cluster ID
            - 'majority_label': str, the most common 'ambiental_label' in the cluster

        Example:
        [
            {'cl_id': 1, 'majority_label': 'Caribbean Sea'},
            {'cl_id': 2, 'majority_label': 'Galapagos Islands'},
            ...
        ]
    """
    print('- - - FIND MAJORITY LABELS - - -')
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
    print(majority_labels)
    return majority_labels

def compute_evaluation_metrics(clustered_samples):
    """
    Computes per-label (ambiental_label) precision, recall,
    as well as macro and micro averages.

    Parameters
    -----------
        clustered_samples

    Returns
    ---------
        label_metrics: dict per ambiental_label
        macro_avg: dict
        micro_avg: dict
    """
    print('* - - - COMPUTE EVALUATION METRICS - - - *')
    majority_labels = find_majority_label(clustered_samples)
    clid_to_majority = {entry['cl_id']: entry['majority_label'] for entry in majority_labels}

    all_labels = set(sample['ambiental_label'] for sample in clustered_samples)
    label_metrics = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in all_labels}

    # Count TP, FP, FN
    for sample in clustered_samples:
        true_label = sample['ambiental_label']
        cl_id = sample['label']
        predicted_label = clid_to_majority.get(cl_id)

        if predicted_label == true_label:
            label_metrics[true_label]['TP'] += 1
        else:
            label_metrics[predicted_label]['FP'] += 1
            label_metrics[true_label]['FN'] += 1

    total_TP = total_FP = total_FN = 0

    # Compute per-label metrics and sum for micro
    for label, counts in label_metrics.items():
        TP, FP, FN = counts['TP'], counts['FP'], counts['FN']
        total_TP += TP
        total_FP += FP
        total_FN += FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        counts.update({'precision': precision, 'recall': recall, 'F1score': F1score})

    # Macro averaging
    macro_avg = {
        'macro_precision': sum(m['precision'] for m in label_metrics.values()) / len(label_metrics),
        'macro_recall': sum(m['recall'] for m in label_metrics.values()) / len(label_metrics),
        'macro_F1score': sum(m['F1score'] for m in label_metrics.values()) / len(label_metrics),
    }

    # Micro averaging
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_F1score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    micro_avg = {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_F1score': micro_F1score,
    }

    # Print results
    print('- - - RESULTS - - -')
    for label, m in label_metrics.items():
        print(f"Label '{label}': TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, "
              f"Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1score={m['F1score']:.2f}")
    print("Macro-Averaged Metrics:", macro_avg)
    print("Micro-Averaged Metrics:", micro_avg)

    return label_metrics, macro_avg, micro_avg