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
from histocartography.image.VorHoVerNet.post_processing import get_instance_output, get_instance_output_v2, DEFAULT_H, DEFAULT_K, get_original_image_from_file
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS, mark_nuclei
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
    '--v2',
    type=bool,
    help='whether to use v2 (data-dependent threshold)',
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
    V2 = arguments.v2
    CKPT = arguments.ckpt_filename

    os.makedirs(OUT_PATH, exist_ok=True)

    dataset = CoNSeP(download=False, root=DATASET_PATH)

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}

    for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
        ori = get_original_image_from_file(IDX, root=IN_PATH, split=SPLIT, ckpt=CKPT)
        if V2:
            output_map = get_instance_output_v2(IDX, root=IN_PATH, split=SPLIT, h=SEG_THRESHOLD, ckpt=CKPT)
        else:
            output_map = get_instance_output(True, IDX, root=IN_PATH, split=SPLIT, h=SEG_THRESHOLD, k=DIS_THRESHOLD, ckpt=CKPT)
        out_file_prefix = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}'
        out_npy = out_file_prefix + '.npy'
        out_img = out_file_prefix + '.png'
        out_b_img = out_file_prefix + '_both.png'
        out_p_img = out_file_prefix + '_pred.png'
        out_l_img = out_file_prefix + '_label.png'
        np.save(out_npy, output_map)
        image = draw_label_boundaries(ori, output_map.copy())
        imsave(out_img, image.astype(np.uint8))
        mlflow.log_artifact(out_npy)
        mlflow.log_artifact(out_img)

        label, _ = dataset.read_labels(IDX, SPLIT)
        s = score(output_map, label, *metrics)

        for img, p in zip(mark_nuclei(ori, output_map, label, s['nucleuswise']), [out_b_img, out_p_img, out_l_img]):
            imsave(p, img)
            mlflow.log_artifact(p)

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
