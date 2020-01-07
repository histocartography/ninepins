import numpy as np

def overall_IOU(output_map, label):
    """
    Compute overall IOU of model output and label.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        IOU (float)

    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    seg_map = output_map != 0
    seg_label = label != 0

    I = seg_map & seg_label
    U = seg_map | seg_label

    return np.count_nonzero(I) / np.count_nonzero(U)

def average_IOU(output_map, label):
    """
    Compute IOU of model output and label averaged over annotated nuclei.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        avgIOU (float)
        
    Definition of matched nuclei:
        nuclei exist both in model output and label, and have more than 50% overlap.
    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    MAX_LABEL_IDX = int(label.max())
    IOUs = []
    for idx in range(1, MAX_LABEL_IDX + 1):
        overlapped_indices = np.unique(output_map[label == idx])
        if len(overlapped_indices) == 0:
            continue
        overlapped_percentages = []

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
            B = (label == idx)
            I = A & B
            U = A | B

            perc = np.count_nonzero(I) / np.count_nonzero(B)
            IOU = np.count_nonzero(I) / np.count_nonzero(U)

            overlapped_percentages.append((perc, IOU))
        IOUs.append(max(overlapped_percentages, key=lambda x: x[0])[1])

    return sum(IOUs) / len(IOUs)

def pixelwise_stats(output_map, label):
    """
    Compute pixel-wise statistics of model output and label.
    Accuracy, Precision, Sensitivity, and Specificity.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        (dict)
            'TP' (int): True Positive
            'TN' (int): True Negative
            'FP' (int): False Positive
            'FN' (int): False Negative
            'ALL' (int): number of all pixels
            'Accuracy' (float): accuracy
            'Precision' (float): precision
            'Sensitivity (float)': sensitivity
            'Specificity' (float): specificity
        
    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    seg_map = output_map != 0
    seg_label = label != 0

    PP = seg_map
    PN = ~PP
    LP = seg_label
    LN = ~LP

    TP = np.count_nonzero(PP & LP)
    TN = np.count_nonzero(PN & LN)
    FP = np.count_nonzero(PP & LN)
    FN = np.count_nonzero(PN & LP)
    ALL = seg_map.size

    Accuracy = (TP + TN) / ALL
    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (FP + TN)

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'ALL': ALL,
        'Accuracy': Accuracy,
        'Precision': Precision,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity
    }
    
    return stats(TP, TN, FP, FN, ALL)

def nucleuswise_stats(output_map, label):
    """
    Compute nucleus-wise statistics of model output and label.
    Sensitivity, and Specificity.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        (dict)
            'TP' (int): True Positive (annotation)
            'TP_pred' (int): True Positive (prediction)
            'FP' (int): False Positive
            'FN' (int): False Negative
            'Precision' (float): precision
            'Sensitivity (float)': sensitivity

    Under nucleus-wise condition, we can only define TP, FN, and FP (and thus Precision and Sensitivity).
    Definition of TP:
        # of annotated nuclei covered by a predicted nucleus with more than 70% overlap
        NOTE: # is different in model output and label,
            since one nucleus in prediction could cover multiple nuclei in label and vice versa.
    Definition of FN:
        (# of annotated nuclei) - TP (# in label)
    Definitoin of FP:
        (# of predicted nuclei) - TP (# in prediction)
    NOTE: label needs to be cropped first to fit the valid size of the model.
    """

    MAX_LABEL_IDX = int(label.max())
    TP = 0
    hit = [False] * int(output_map.max())

    for idx in range(1, MAX_LABEL_IDX + 1):
        overlapped_indices = np.unique(output_map[label == idx])
        if len(overlapped_indices) == 0:
            continue

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
            B = (label == idx)
            I = A & B

            perc = np.count_nonzero(I) / np.count_nonzero(B)

            if perc >= 0.7:
                TP += 1
                hit[overlapped_index - 1] = True
                break

    TP_pred = np.count_nonzero(hit)

    FN = MAX_LABEL_IDX - TP
    FP = len(hit) - TP_pred

    Precision = TP_pred / len(hit)
    Sensitivity = TP / MAX_LABEL_IDX

    return {
        'TP': TP,
        'TP_pred': TP_pred,
        'FP': FP,
        'FN': FN,
        'Precision': Precision,
        'Sensitivity': Sensitivity
    }

# Mapping from metrics name to computation function
VALID_METRICS = {
    'IOU': overall_IOU,
    'avgIOU': average_IOU,
    'pixelwise': pixelwise_stats,
    'nucleiwise': nucleiwise_stats
}

def score(output_map, label, *metrics):
    """
    Compute requested metrics of model output and label.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
        metrics (list[str]): metrics names
    Returns:
        (dict)
            metrics_name (str): metrics value
            ...
    """
    res = {}
    for metric in metrics:
        try:
            metric_function = VALID_METRICS[metric]
            res[metric] = metric_function(output_map, label)
        except KeyError:
            print('{} is not a valid metric. skipped.'.format(metric))
    
    return res

if __name__ == "__main__":
    from pathlib import Path
    from utils import show, get_valid_view
    IDX = 3
    prefix = 'out_k_0.2'

    output_map = np.load('output/{}_{}.npy'.format(prefix, IDX))

    from dataset_reader import CoNSeP

    dataset = CoNSeP(download=False)
    label, _ = dataset.read_labels(IDX, 'test')
    label = get_valid_view(label)

    print(score(output_map, label, 'nucleiwise', 'IOU'))