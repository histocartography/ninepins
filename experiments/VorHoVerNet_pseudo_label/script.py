#!/usr/bin/env python3
"""
Script for testing pseudo label generation
"""
import logging
import argparse
import numpy as np
import sys
import os
import re
from time import time
import mlflow
from skimage.io import imsave
from skimage.morphology import label as cc
from histocartography.image.VorHoVerNet.metrics import score
from histocartography.image.VorHoVerNet.utils import draw_boundaries, get_point_from_instance
from histocartography.image.VorHoVerNet.pseudo_label import gen_pseudo_label
# from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges
from histocartography.image.VorHoVerNet.performance import OutTime as OutTime_
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader
# import histocartography.image.VorHoVerNet.color_label as color_label

# setup logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('Histocartography::PseudoLabel')
h1 = logging.StreamHandler(sys.stdout)
log.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
h1.setFormatter(formatter)
log.addHandler(h1)

class OutTime(OutTime_):
    def __exit__(self, type, value, traceback):
        # print execution time when scope is closed.
        if self.switchOff: return
        t = time() - self.st
        log.info("time passed: %5f s" % t)
        self.onExit(t)

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
    default='../../histocartography/image/VorHoVerNet/pseudo',
    required=False
)
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
    '-f',
    '--features',
    type=str,
    help='features for clustering (sequence of characters in [C, S, D]) C: RGB, D: distance, S: HE',
    default='CD',
    required=False
)
parser.add_argument(
    '-k',
    '--num-clusters',
    type=int,
    help='number of clusters',
    default=3,
    required=False
)
parser.add_argument(
    '-t',
    '--out-time',
    type=bool,
    help='whether to print out execution time',
    default=True,
    required=False
)
parser.add_argument(
    '-l',
    '--filter',
    type=bool,
    help='whether to filter out connected-components not covering any point',
    default=True,
    required=False
)

def main(arguments):
    """
    Generate pseudo label on the split of dataset
    Args:
        arguments (Namespace): parsed arguments.
    """
    # create aliases
    SPLIT = arguments.split
    OUT_PATH = arguments.output_path
    DATASET_ROOT = arguments.dataset_root
    DATASET = arguments.dataset
    PREFIX = arguments.prefix
    NUM_CLUSTERS = arguments.num_clusters
    FILTER = arguments.filter

    OutTime.switchOff = not arguments.out_time

    FEATURES = re.sub("[^CDS]", "", arguments.features)

    os.makedirs(OUT_PATH, exist_ok=True)

    # dataset = CoNSeP(download=False, root=DATASET_PATH)
    dataset = getattr(dataset_reader, DATASET)(download=False, root=DATASET_ROOT+DATASET+"/")

    IOUs = []

    for IDX in range(1, dataset.IDX_LIMITS[SPLIT] + 1):
        log.info(f'Generating pseudo label {IDX}/{dataset.IDX_LIMITS[SPLIT]}')

        image = dataset.read_image(IDX, SPLIT)
        label, _ = dataset.read_labels(IDX, SPLIT)
        point_mask = get_point_from_instance(label, binary=True)

        with OutTime():
            # color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], edges, k=NUM_CLUSTERS)
            pseudo, edges = gen_pseudo_label(image, point_mask, return_edge=True, k=NUM_CLUSTERS, features=FEATURES)

        pseudo = pseudo & (edges == 0)

        if FILTER:
            pseudo = cc(pseudo, connectivity=1)
            for seg_idx in np.unique(pseudo):
                if seg_idx == 0: continue
                if np.count_nonzero((pseudo == seg_idx) & point_mask) == 0:
                    pseudo[pseudo == seg_idx] = 0
            pseudo = pseudo > 0

        draw_boundaries(image, pseudo.copy())
        image[point_mask] = [255, 0, 0]
        
        out_file = f'{OUT_PATH}/mlflow_{PREFIX}_{IDX}.png'
        imsave(out_file, image.astype(np.uint8))
        mlflow.log_artifact(out_file)
        
        s = score(1 * pseudo, label, 'IOU')

        IOU = s['IOU']
        IOUs.append(IOU)
        mlflow.log_metric('IOU', IOU, step=IDX-1)

    mlflow.log_metric("average_IOU", sum(IOUs) / len(IOUs))

if __name__ == "__main__":
    main(arguments=parser.parse_args())
