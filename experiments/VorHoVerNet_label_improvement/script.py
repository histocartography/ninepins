#!/usr/bin/env python3
"""
Script for testing label improvement
"""
import logging
import argparse
import numpy as np
import sys
import os
import mlflow
from skimage.io import imsave
from skimage.morphology import label as cc, binary_dilation, disk
from histocartography.image.VorHoVerNet.post_processing import improve_pseudo_labels, get_original_image_from_file, get_output_from_file, DEFAULT_TRANSFORM
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS
from histocartography.image.VorHoVerNet.dataset_reader import CoNSeP
from histocartography.image.VorHoVerNet.utils import draw_label_boundaries
from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::LabelImprovement')
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
    help='output path.',
    default='../../histocartography/image/VorHoVerNet/iteration',
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
    '-m',
    '--method',
    type=str,
    help='method to use for generating next iteration pseudolabel',
    default='Voronoi',
    choices=['Voronoi', 'instance'],
    required=False
)
parser.add_argument(
    '-n',
    '--no-improve',
    help='do not perform improvement. (evaluate current label)',
    type=bool,
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
    METHOD = arguments.method
    NO_IMPROVE = arguments.no_improve

    os.makedirs(OUT_PATH, exist_ok=True)

    dataset = CoNSeP(download=False, root=DATASET_PATH)

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}

    for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
        ori = get_original_image_from_file(IDX, root=IN_PATH, split=SPLIT)
        current_seg_mask = dataset.read_pseudo_labels(IDX, SPLIT) > 0
        point_mask = dataset.read_points(IDX, SPLIT)
        if NO_IMPROVE:
            edges = binary_dilation(get_voronoi_edges(point_mask) > 0, disk(1))
            new_cell = current_seg_mask & ~edges
            new_cell = cc(new_cell)
        else:
            seg, hor, vet = get_output_from_file(IDX, transform=DEFAULT_TRANSFORM, root=IN_PATH, split=SPLIT)
            _, new_cell = improve_pseudo_labels(current_seg_mask, point_mask, seg, hor, vet, method=METHOD)
        image = draw_label_boundaries(ori, new_cell.copy())
        out_file_prefix = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}'
        out_npy = out_file_prefix + '.npy'
        out_img = out_file_prefix + '.png'
        np.save(out_npy, new_cell)
        imsave(out_img, image.astype(np.uint8))
        mlflow.log_artifact(out_npy)
        mlflow.log_artifact(out_img)
        
        label, _ = dataset.read_labels(IDX, SPLIT)
        
        s = score(new_cell, label, *metrics)
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
