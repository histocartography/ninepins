#!/usr/bin/env python3
"""
Script for evaluating HoverNet performance
"""
import logging
import argparse
import numpy as np
import sys
import os
import mlflow
import scipy.io as sio
from skimage.io import imsave
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::HoverNet')
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
    '-r',
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
    '-p',
    '--prediction-dir',
    type=str,
    help='root directory containing prediction npy files',
    default='../../git/hover_net/src/',
    required=False
)
parser.add_argument(
    '-m',
    '--model-dir',
    type=str,
    help='root directory containing certain model\'s prediction npy files',
    default='output/',
    required=False
)
parser.add_argument(
    '-v',
    '--version',
    type=int,
    help='version of model',
    default=1,
    required=False
)
def main(arguments):
    """
    Run HoverNet evaluation on the split of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    DATASET_ROOT = arguments.dataset_root
    DATASET = arguments.dataset
    PREDICTION_DIR = arguments.prediction_dir
    MODEL_DIR = arguments.model_dir
    VERSION = arguments.version

    dataset = getattr(dataset_reader, DATASET)(download=False, root=DATASET_ROOT+DATASET+"/")

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}

    for IDX in range(1, dataset.IDX_LIMITS['test'] + 1):
        filename = dataset.get_path(IDX, 'test', 'label').split('/')[-1]
        # output_map = np.load(PREDICTION_DIR + filename)
        output_map = sio.loadmat(PREDICTION_DIR + MODEL_DIR + '/v{}.0/np_hv/_proc/'.format(VERSION) + filename.replace('npy', 'mat'))['inst_map']
        label, _ = dataset.read_labels(IDX, 'test')
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
