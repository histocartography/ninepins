#!/usr/bin/env python3
"""
Script for 10-fold cross-validation
"""
import logging
import argparse
import numpy as np
import sys
import os
import mlflow
from skimage.io import imsave
from histocartography.image.VorHoVerNet.post_processing import get_instance_output, DEFAULT_H, DEFAULT_K, get_original_image_from_file, get_output_from_file
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS, mark_nuclei, dot_pred_stats
from histocartography.image.VorHoVerNet.utils import draw_label_boundaries
from histocartography.image.VorHoVerNet.constants import SHUFFLED_IDX
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::10Fold')
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
# parser.add_argument(
#     '-d',
#     '--dataset-path',
#     type=str,
#     help='dataset path.',
#     default='../../histocartography/image/VorHoVerNet/CoNSeP/',
#     required=False
# )
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
    help='whether to use v2 (dot refinement)',
    default=False,
    required=False
)
parser.add_argument(
    '--strong-discard',
    type=bool,
    help='whether to use strong criteria when discarding FP',
    default=False,
    required=False
)
parser.add_argument(
    '--extra-watershed',
    type=bool,
    help='whether to do extra watershed when a predicted nucleus covers multiple points',
    default=True,
    required=False
)
parser.add_argument(
    '--part',
    type=int,
    help='part of dataset to analyze',
    required=True
)

def main(arguments):
    """
    Run post-processing on the part of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    OUT_PATH = arguments.output_path
    IN_PATH = arguments.inference_path
    DATASET_ROOT = arguments.dataset_root
    DATASET = arguments.dataset
    PREFIX = arguments.prefix
    SEG_THRESHOLD = arguments.segmentation_threshold
    DIS_THRESHOLD = arguments.distancemap_threshold
    V2 = arguments.v2
    STRONG_DISCARD = arguments.strong_discard
    EXTRA_WATERSHED = arguments.extra_watershed
    PART = arguments.part
    SPLIT = 'test'
    CKPT = f'model_k{PART}_ckpt'

    os.makedirs(OUT_PATH, exist_ok=True)

    # dataset = CoNSeP(download=False, root=DATASET_PATH)
    dataset = getattr(dataset_reader, DATASET)(download=False, root=DATASET_ROOT+DATASET+"/")

    aggregated_metrics = {}

    # for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
    for i, IDX in enumerate(SHUFFLED_IDX[(PART - 1) * 3: PART * 3]):
        metrics = list(VALID_METRICS.keys())
        ori = get_original_image_from_file(i+1, root=IN_PATH, split=SPLIT, ckpt=CKPT)
        output_map = get_instance_output(True, i+1, root=IN_PATH, split=SPLIT,
                                        h=SEG_THRESHOLD, k=DIS_THRESHOLD,
                                        ckpt=CKPT, dot_refinement=V2, 
                                        strong_discard=STRONG_DISCARD, extra_watershed=EXTRA_WATERSHED)
        if V2:
            seg, hor, vet, dot = get_output_from_file(i+1, root=IN_PATH, split=SPLIT,
                                        ckpt=CKPT, read_dot=True)
        out_file_prefix = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}_p{PART}'
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

        label, _ = dataset.read_labels(IDX, 'all')
        point_mask = dataset.read_points(IDX, 'all')
        s = score(output_map, label, *metrics)

        if V2:
            metrics += ['dot_pred', 'DQ_dot']
            ss = dot_pred_stats(dot > 0.5, label)
            s['dot_pred'] = ss
            s['DQ_dot'] = ss['TP'] / (ss['TP'] + 0.5 * ss['FN'] + 0.5 * ss['FP'])

        for img, p in zip(mark_nuclei(ori, output_map, label, stats=s['nucleuswise_point'], dot_pred=dot if V2 else None), [out_b_img, out_p_img, out_l_img]):
            img[point_mask] = [255, 255, 0]
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
