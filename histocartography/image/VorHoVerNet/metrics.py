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
        nuclei exist both in model output and label, and have the maximum overlap.
    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    MAX_LABEL_IDX = int(label.max())
    IOUs = []
    for idx in range(1, MAX_LABEL_IDX + 1):
        overlapped_indices = np.unique(output_map[label == idx])
        if len(overlapped_indices) == 0:
            continue
        overlapped_percentages = []
        B = (label == idx)

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
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
        B = (label == idx)

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
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

def DICE2(output_map, label):
    """
    Compute DICE2 score of model output and label.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        DICE2 (float)

    Definition of matched nuclei:
        nuclei exist both in model output and label, and have the maximum overlap.

    For each annotated nucleus (X) and corresponding matched predicted nucleus (Y):
        DICE = 2 * (|X intersect Y|) / (|X| + |Y|)
    DICE2 = average of DICE

    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    MAX_LABEL_IDX = int(label.max())
    DICEs = []
    for idx in range(1, MAX_LABEL_IDX + 1):
        overlapped_indices = np.unique(output_map[label == idx])
        if len(overlapped_indices) == 0:
            continue
        overlapped_percentages = []
        B = (label == idx)

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
            I = A & B

            perc = np.count_nonzero(I) / np.count_nonzero(B)
            DICE = 2 * np.count_nonzero(I) / (np.count_nonzero(A) + np.count_nonzero(B))

            overlapped_percentages.append((perc, DICE))
        DICEs.append(max(overlapped_percentages, key=lambda x: x[0])[1])

    return sum(DICEs) / len(DICEs)

def AJI(output_map, label):
    """
    Compute Aggregated Jaccord Index of model output and label.
    Args:
        output_map (numpy.ndarray): model output (instance map)
        label (numpy.ndarray): label (instance map)
    Returns:
        AJI (float)

    Definition of matched nuclei:
        nuclei exist both in model output and label, and have the maximum overlap.
    
    I = 0
    U = 0
    For each annotated nucleus (X) and corresponding matched predicted nucleus (Y):
        I += |X intersect Y|
        U += |X union Y|
    For each not paired annotated nucleus (X):
        U += |X|
    For each not paired predicted nucleus (Y):
        U += |Y|
    AJI = I / U

    NOTE: label needs to be cropped first to fit the valid size of the model.
    """
    MAX_LABEL_IDX = int(label.max())
    AI = 0
    AU = 0

    hit = [False] * int(output_map.max())
    for idx in range(1, MAX_LABEL_IDX + 1):
        overlapped_indices = np.unique(output_map[label == idx])
        overlapped_percentages = []
        B = (label == idx)

        for overlapped_index in overlapped_indices:
            A = (output_map == overlapped_index)
            I = A & B
            U = A | B

            perc = np.count_nonzero(I) / np.count_nonzero(B)

            overlapped_percentages.append((perc, np.count_nonzero(I), np.count_nonzero(U), overlapped_index))

        if len(overlapped_percentages) > 0:
            _, mI, mU, idx = max(overlapped_percentages, key=lambda x: x[0])
            AI += mI
            AU += mU
            hit[idx - 1] = True
        else:
            AU += np.count_nonzero(B)

    for i, hit_ in enumerate(hit):
        if hit_: continue
        AU += np.count_nonzero(output_map == (i + 1))

    return AI / AU

# Mapping from metrics name to computation function
VALID_METRICS = {
    'IOU': overall_IOU,
    'avgIOU': average_IOU,
    'pixelwise': pixelwise_stats,
    'nucleuswise': nucleuswise_stats,
    'DICE2': DICE2,
    'AJI': AJI
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
    # from utils import show, get_valid_view
    from dataset_reader import CoNSeP
    # IDX = 3
    prefix = 'out'
    # prefix = 'curr_cell'
    dataset = CoNSeP(download=False)
    
    m = 'DICE2'

    for IDX in range(1, 15):
        output_map = np.load('output/{}_{}.npy'.format(prefix, IDX))
        # output_map = np.load('iteration/{}_{}.npy'.format(prefix, IDX))

        label, _ = dataset.read_labels(IDX, 'test')
        # label = get_valid_view(label)

        s = score(output_map, label, m)
        print(s[m])