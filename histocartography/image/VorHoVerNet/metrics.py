import numpy as np

def overall_IOU(output_map, label):
    """
    label needs to be cropped first to fit the valid size of the model.
    """
    seg_map = output_map != 0
    seg_label = label != 0

    I = seg_map & seg_label
    U = seg_map | seg_label

    return np.count_nonzero(I) / np.count_nonzero(U)

def average_IOU(output_map, label):
    """
    label needs to be cropped first to fit the valid size of the model.
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
    label needs to be cropped first to fit the valid size of the model.
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

    ACC = (TP + TN) / ALL
    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)
    Specificity = TN / (FP + TN)

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'ALL': ALL,
        'ACC': ACC,
        'Precision': Precision,
        'Sensitivity': Sensitivity,
        'Specificity': Specificity
    }
    
    return stats(TP, TN, FP, FN, ALL)

def nucleiwise_stats(output_map, label):
    """
    label needs to be cropped first to fit the valid size of the model.

    Under nucleiwise condition, we can only define TP, FN, and FP (and thus Precision and Sensitivity).
    Definition of TP: a nucleus in ground truth is covered by more than 70% overlapping in prediction
    Definition of FN: (# of label nuclei) - TP
    Definitoin of FP: (# of predicted nuclei) - TP
    """

    MAX_LABEL_IDX = int(label.max())
    TP = 0
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
                break

    FN = MAX_LABEL_IDX - TP
    FP = int(output_map.max()) - TP

    Precision = TP / (TP + FP)
    Sensitivity = TP / (TP + FN)

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision': Precision,
        'Sensitivity': Sensitivity
    }

VALID_METRICS = {
    'IOU': overall_IOU,
    'avgIOU': average_IOU,
    'pixelwise': pixelwise_stats,
    'nucleiwise': nucleiwise_stats
}

def score(output_map, label, metrics=['IOU']):
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
    from utils import show
    IDX = 0
    prefix = 'out_k_0.2'

    output_map = np.load('output/{}_{}.npy'.format(prefix, IDX))

    from dataset_reader import CoNSeP

    dataset = CoNSeP(download=False)
    label, _ = dataset.read_labels(IDX + 1, 'test')
    label = label[95: 895, 95: 895]

    print(score(output_map, label, metrics=['nucleiwise']))