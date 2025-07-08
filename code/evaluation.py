from collections import defaultdict, Counter

def find_majority_label(clustered_samples):
    """
    Determines the majority (most frequent) ambiental_label for each cluster.

    This function groups samples by their cluster ID (from the 'center' key),
    then finds the most common 'ambiental_label' within each cluster.

    Parameters
    ----------
    clustered_samples : list of dict
        Each dictionary represents a sample and should contain at least:
        - 'center' : the cluster ID the sample belongs to
        - 'ambiental_label' : the true class label of the sample

    Returns
    -------
    majority_labels : list of dict
        Each dictionary contains:
        - 'cl_id' : the cluster ID
        - 'majority_label' : the most frequent ambiental_label within the cluster
    """
    
    # Group ambiental_labels by cluster label
    clusters = defaultdict(list)
    for sample in clustered_samples:
        cl_id = sample['center']
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

def compute_evaluation_metrics(clustered_samples):
    """
    Computes per-label (ambiental_label) precision, recall, and F1-score,
    as well as macro, micro, and weighted averages.

    Parameters
    ----------
    clustered_samples : list of dict
        Each dictionary represents a sample and should contain at least the keys:
        - 'ambiental_label': the ground truth label.
        - 'center': the ID of the cluster to which the sample was assigned.

    Returns
    -------
    label_metrics : dict
        Dictionary keyed by label. For each label, contains:
        - 'TP' : int
            True Positives.
        - 'FP' : int
            False Positives.
        - 'FN' : int
            False Negatives.
        - 'precision' : float
            Precision score for the label.
        - 'recall' : float
            Recall score for the label.
        - 'F1score' : float
            F1-score for the label.

    macro_avg : dict
        Contains:
        - 'macro_precision' : float
        - 'macro_recall' : float
        - 'macro_F1score' : float

    micro_avg : dict
        Contains:
        - 'micro_precision' : float
        - 'micro_recall' : float
        - 'micro_F1score' : float

    weighted_avg : dict
        Contains:
        - 'weighted_precision' : float
        - 'weighted_recall' : float
        - 'weighted_F1score' : float
    """


    majority_labels = find_majority_label(clustered_samples)
    clid_to_majority = {entry['cl_id']: entry['majority_label'] for entry in majority_labels}

    all_labels = set(sample['ambiental_label'] for sample in clustered_samples)
    label_metrics = {label: {'TP': 0, 'FP': 0, 'FN': 0} for label in all_labels}

    # Count TP, FP, FN
    for sample in clustered_samples:
        true_label = sample['ambiental_label']
        cl_id = sample['center']
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
    
    micro_avg = {
        'micro_precision': micro_precision,
        'micro_recall': micro_precision,
        'micro_F1score': micro_precision,
    }

    # Compute total number of samples (denominator n)
    total_support = sum(label_metrics[label]['TP'] + label_metrics[label]['FN'] for label in label_metrics)

    weighted_precision = sum(
        label_metrics[label]['precision'] * (label_metrics[label]['TP'] + label_metrics[label]['FN'])
        for label in label_metrics
    ) / total_support if total_support > 0 else 0.0

    weighted_recall = sum(
        label_metrics[label]['recall'] * (label_metrics[label]['TP'] + label_metrics[label]['FN'])
        for label in label_metrics
    ) / total_support if total_support > 0 else 0.0

    weighted_F1score = sum(
        label_metrics[label]['F1score'] * (label_metrics[label]['TP'] + label_metrics[label]['FN'])
        for label in label_metrics
    ) / total_support if total_support > 0 else 0.0

    weighted_avg = {
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_F1score': weighted_F1score,
    }

    return label_metrics, macro_avg, micro_avg, weighted_avg