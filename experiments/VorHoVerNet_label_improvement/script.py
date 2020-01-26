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
from histocartography.image.VorHoVerNet.post_processing import improve_pseudo_labels, get_original_image_from_file, get_output_from_file, DEFAULT_TRANSFORM, DEFAULT_K
from histocartography.image.VorHoVerNet.metrics import score, VALID_METRICS, mark_nuclei
from histocartography.image.VorHoVerNet.utils import draw_label_boundaries
from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader

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
    '-c',
    '--ckpt-filename',
    type=str,
    help='filename of the checkpoint.',
    default='model_009_ckpt_epoch_18',
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
    '--pseudolabel-version',
    type=int,
    help='version of pseudo label',
    default=1,
    required=False
)

def main(arguments):
    """
    Run label improvement on the split of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    SPLIT = arguments.split
    OUT_PATH = arguments.output_path
    IN_PATH = arguments.inference_path
    DATASET_ROOT = arguments.dataset_root
    DATASET = arguments.dataset
    PREFIX = arguments.prefix
    METHOD = arguments.method
    NO_IMPROVE = arguments.no_improve
    DIS_THRESHOLD = arguments.distancemap_threshold
    V2 = arguments.v2
    CKPT = arguments.ckpt_filename
    STRONG_DISCARD = arguments.strong_discard
    VERSION = arguments.pseudolabel_version

    os.makedirs(OUT_PATH, exist_ok=True)

    # dataset = CoNSeP(download=False, root=DATASET_PATH)
    dataset = getattr(dataset_reader, DATASET)(download=False, root=DATASET_ROOT+DATASET+"/", ver=VERSION)

    metrics = VALID_METRICS.keys()

    aggregated_metrics = {}

    for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
        ori = get_original_image_from_file(IDX, root=IN_PATH, split=SPLIT, ckpt=CKPT)
        current_seg_mask = dataset.read_pseudo_labels(IDX, SPLIT) > 0
        point_mask = dataset.read_points(IDX, SPLIT)
        if NO_IMPROVE:
            edges = binary_dilation(get_voronoi_edges(point_mask) > 0, disk(1))
            new_cell = current_seg_mask & ~edges
            new_cell = cc(new_cell)
        else:
            preds = get_output_from_file(IDX, transform=DEFAULT_TRANSFORM, root=IN_PATH, split=SPLIT, ckpt=CKPT, read_dot=V2)
            _, new_cell = improve_pseudo_labels(ori, current_seg_mask, point_mask, preds, method=METHOD, k=DIS_THRESHOLD, strong_discard=STRONG_DISCARD)
        image = draw_label_boundaries(ori, new_cell.copy())
        out_file_prefix = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}'
        out_npy = out_file_prefix + '.npy'
        out_img = out_file_prefix + '.png'
        out_b_img = out_file_prefix + '_both.png'
        out_p_img = out_file_prefix + '_pred.png'
        out_l_img = out_file_prefix + '_label.png'
        np.save(out_npy, new_cell)
        imsave(out_img, image.astype(np.uint8))
        mlflow.log_artifact(out_npy)
        mlflow.log_artifact(out_img)
        
        label, _ = dataset.read_labels(IDX, SPLIT)        
        s = score(new_cell, label, *metrics)

        for img, p in zip(mark_nuclei(ori, new_cell, label, stats=s['nucleuswise_point'], dot_pred=dot if V2 else None), [out_b_img, out_p_img, out_l_img]):
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
