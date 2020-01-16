import numpy as np
from skimage.morphology import binary_dilation, dilation, disk, label
from histocartography.image.VorHoVerNet.color_label import get_cluster_label
from histocartography.image.VorHoVerNet.performance import OutTime
from histocartography.image.VorHoVerNet.utils import (draw_boundaries,
                                                    get_point_from_instance)
from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges


def gen_pseudo_label(image, point_mask, return_edge=False):
    """
    Generate pseudo label from image and instance label.
    1. Generate distance based label.
    2. Generate color based label.
    3. Combine these two labels.
    Args:
        image (numpy.ndarray[uint8]): image.
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        return_edge (bool): whether to also return edges.
    Returns:
        pseudo_label (numpy.ndarray[bool]) /
        (tuple)
            pseudo_label (numpy.ndarray[bool])
            edges (numpy.ndarray[int])
            if return_edge
    """

    out_dict = {}

    distance_based_label = get_voronoi_edges(point_mask, extra_out=out_dict)
    color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], distance_based_label)

    nuclei = (color_based_label == [0, 255, 0]).all(axis=2)

    # draw_boundaries(image, nuclei, color=[0, 255, 255])

    color_based_label[nuclei] = [0, 0, 0]
    
    labeled_nuclei = label(nuclei)
    labeled_point_mask = label(point_mask)
    dilated_point_mask = dilation(labeled_point_mask, disk(5))
    mask = np.zeros_like(nuclei)
    for point_index in np.unique(labeled_point_mask):
        if point_index == 0: continue
        overlapped_nuclei = np.unique(labeled_nuclei[dilated_point_mask == point_index])
        for overlapped_nucleus in overlapped_nuclei:
            if overlapped_nucleus == 0: continue
            mask = mask | (labeled_nuclei == overlapped_nucleus)

    point_mask = binary_dilation(point_mask, disk(3))

    mask = mask | point_mask

    return (mask, distance_based_label) if return_edge else mask

if __name__ == "__main__":
    from dataset_reader import CoNSeP
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Cluster Label Generator (Experiment)")
    parser.add_argument("-n", "--name", default="test",
                        help="experiment name")
    parser.add_argument("-i", "--index", default=4, type=int,
                        help="index of the image in dataset")
    parser.add_argument("-s", "--split", default="test", choices=["test", "train"],
                        help="split of the dataset ([test, train])")
    args = parser.parse_args()

    IDX = args.index
    SPLIT = args.split
    EXP_NAME = args.name

    dataset = CoNSeP(download=False)
    ori = dataset.read_image(IDX, SPLIT)
    point_mask = dataset.read_points(IDX, SPLIT)
    # lab, type_ = dataset.read_labels(IDX, SPLIT)
    # point_mask = get_point_from_instance(lab, binary=True)

    from skimage.io import imsave

    with OutTime():
        label = gen_pseudo_label(ori, point_mask)
        draw_boundaries(ori, label)
    point_mask = binary_dilation(point_mask, disk(3))
    # point_mask = np.where(point_mask, 255, 0)
    ori[point_mask] = [255, 255, 0]

    # imsave("points.png", point_mask.astype(np.uint8))
    imsave("pseudo_label{}.png".format(EXP_NAME), ori)
