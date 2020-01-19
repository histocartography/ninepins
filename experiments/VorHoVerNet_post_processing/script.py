#!/usr/bin/env python3
"""
Script for testing post-processing
"""
import logging
import argparse
import numpy as np
import sys
import os
import mlflow
from skimage.io import imsave
from histocartography.image.VorHoVerNet.post_processing import get_instance_output, DEFAULT_H, DEFAULT_K, get_original_image_from_file
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS
from histocartography.image.VorHoVerNet.dataset_reader import CoNSeP
from histocartography.image.VorHoVerNet.utils import draw_label_boundaries

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::PostProcessing')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

# configure argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--split',
    type=str,
    help='split of dataset to run.',
    default='test',
    required=False
)
parser.add_argument(
    '-o',
    '--output-path',
    type=str,
    help='instance output path.',
    default='../../histocartography/image/VorHoVerNet/output',
    required=False
)
parser.add_argument(
    '-i',
    '--inference-path',
    type=str,
    help='inference file directory path.',
    default='../../histocartography/image/VorHoVerNet/inference',
    required=False
)
parser.add_argument(
    '-d',
    '--dataset-path',
    type=str,
    help='dataset path.',
    default='../../histocartography/image/VorHoVerNet/CoNSeP/',
    required=False
)
parser.add_argument(
    '-p',
    '--prefix',
    type=str,
    help='prefix of files',
    required=True
)
parser.add_argument(
    '-g',
    '--segmentation-threshold',
    type=float,
    help='threshold for segmentation prediction',
    default=DEFAULT_H,
    required=False
)
parser.add_argument(
    '-t',
    '--distancemap-threshold',
    type=float,
    help='threshold for distance map prediction',
    default=DEFAULT_K,
    required=False
)
parser.add_argument(
    '--sobel',
    type=bool,
    help='use sobel filter instead of laplacian filter',
    default=True,
    required=False
)
parser.add_argument(
    '--max',
    type=bool,
    help='use maximum instead of l2_norm',
    default=False,
    required=False
)

def main(arguments):
    """
    Run post-processing on the split of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    SPLIT = arguments.split
    OUT_PATH = arguments.output_path
    IN_PATH = arguments.inference_path
    DATASET_PATH = arguments.dataset_path
    PREFIX = arguments.prefix
    SEG_THRESHOLD = arguments.segmentation_threshold
    DIS_THRESHOLD = arguments.distancemap_threshold
    SOBEL = arguments.sobel
    MAX = arguments.max

    os.makedirs(OUT_PATH, exist_ok=True)

    dataset = CoNSeP(download=False, root=DATASET_PATH)

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}

    for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
        ori = get_original_image_from_file(IDX, root=IN_PATH)
        output_map = get_instance_output(True, IDX, root=IN_PATH, h=SEG_THRESHOLD, k=DIS_THRESHOLD, use_sobel=SOBEL, use_max=MAX)
        out_file_prefix = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}'
        out_npy = out_file_prefix + '.npy'
        out_img = out_file_prefix + '.png'
        np.save(out_npy, output_map)
        image = draw_label_boundaries(ori, output_map.copy())
        imsave(out_img, image.astype(np.uint8))
        mlflow.log_artifact(out_npy)
        mlflow.log_artifact(out_img)

        label, _ = dataset.read_labels(IDX, SPLIT)
        s = score(output_map, label, *metrics)
        for metric in metrics:
            value = s[metric]
            if isinstance(value, dict):
                for key, val in value.items():
                    if not isinstance(val, list):
                        metric_name = metric + '_' + key
                        mlflow.log_metric(metric_name, val, step=(IDX-1))
                        if metric_name not in aggregated_metrics:
                            aggregated_metrics[metric_name] = []
                        aggregated_metrics[metric_name].append(val)
            else:
                mlflow.log_metric(metric, value, step=(IDX-1))
                if metric not in aggregated_metrics:
                    aggregated_metrics[metric] = []
                aggregated_metrics[metric].append(value)

    for metric, score_list in aggregated_metrics.items():
        mlflow.log_metric("average_" + metric, sum(score_list) / len(score_list))


if __name__ == "__main__":
    main(arguments=parser.parse_args())
