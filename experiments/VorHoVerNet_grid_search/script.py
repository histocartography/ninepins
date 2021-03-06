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
from histocartography.image.VorHoVerNet.post_processing import get_instance_output, DEFAULT_H
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::GridSearch')
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
    '-x',
    '--index',
    type=int,
    help='index of image.',
    required=True
)
parser.add_argument(
    '-i',
    '--inference-path',
    type=str,
    help='inference file directory path.',
    default='../../histocartography/image/VorHoVerNet/inference',
    required=False
)
# parser.add_argument(
#     '-d',
#     '--dataset-path',
#     type=str,
#     help='dataset path.',
#     default='../../histocartography/image/VorHoVerNet/CoNSeP/',
#     required=False
# )
parser.add_argument(
    '-t',
    '--dataset-root',
    type=str,
    help='root directory containing datasets',
    default='../../histocartography/image/VorHoVerNet/',
    required=False
)
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    help='dataset ([CoNSeP, MoNuSeg])',
    default='CoNSeP',
    choices=['CoNSeP', 'MoNuSeg'],
    required=False
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
    '-r',
    '--range',
    type=str,
    help=r'range for grid search. {st:ed:num}',
    default='1:2:21',
    required=False
)
parser.add_argument(
    '--v2',
    type=bool,
    help='whether to use v2 (dot refinement)',
    default=False,
    required=False
)
parser.add_argument(
    '-c',
    '--ckpt-filename',
    type=str,
    help='filename of the checkpoint.',
    default='model_009_ckpt_epoch_18',
    required=False
)

def main(arguments):
    """
    Run grid search of post-processing threshold on the split of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    SPLIT = arguments.split
    IN_PATH = arguments.inference_path
    DATASET_ROOT = arguments.dataset_root
    DATASET = arguments.dataset
    SEG_THRESHOLD = arguments.segmentation_threshold
    IDX = arguments.index
    RANGE = arguments.range
    V2 = arguments.v2
    CKPT = arguments.ckpt_filename

    try:
        st, ed, num = map(float, RANGE.split(':'))
        num = int(num)
    except:
        log.error('Invalid range')

    # dataset = CoNSeP(download=False, root=DATASET_PATH)
    dataset = getattr(dataset_reader, DATASET)(download=False, root=DATASET_ROOT+DATASET+"/")

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}
    thresholds = []

    for step, k in enumerate(np.linspace(st, ed, num)):
        thresholds.append(k)
        mlflow.log_metric('threshold', k, step=step)
        output_map = get_instance_output(True, IDX, 
                                        root=IN_PATH, split=SPLIT, 
                                        h=SEG_THRESHOLD, k=k, 
                                        ckpt=CKPT, dot_refinement=V2)
        label, _ = dataset.read_labels(IDX, SPLIT)
        s = score(output_map, label, *metrics)
        for metric in metrics:
            value = s[metric]
            if isinstance(value, dict):
                for key, val in value.items():
                    if not isinstance(val, list):
                        metric_name = metric + '_' + key
                        mlflow.log_metric(metric_name, val, step=step)
                        if metric_name not in aggregated_metrics:
                            aggregated_metrics[metric_name] = []
                        aggregated_metrics[metric_name].append(val)
            else:
                mlflow.log_metric(metric, value, step=step)
                if metric not in aggregated_metrics:
                    aggregated_metrics[metric] = []
                aggregated_metrics[metric].append(value)

    for metric, score_list in aggregated_metrics.items():
        mlflow.log_metric("average_" + metric, sum(score_list) / len(score_list))

    mlflow.log_metric("best_threshold", thresholds[np.argmax(aggregated_metrics['DQ_point'])])


if __name__ == "__main__":
    main(arguments=parser.parse_args())
