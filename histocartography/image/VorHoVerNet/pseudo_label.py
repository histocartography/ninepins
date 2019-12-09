from utils import get_point_from_instance, draw_boundaries
from color_label import get_cluster_label, get_cluster_label_v2
from Voronoi_label import get_voronoi_edges
from skimage.morphology import label, dilation, disk, binary_dilation
from performance import OutTime
import numpy as np

def gen_pseudo_label(image, point_mask, typed_point_map, v2=False, return_edge=False):
    """
    Generate pseudo label from image and instance label.
    1. Generate distance based label.
    2. Generate color based label.
    3. Combine these two labels.
    @image: image.
    @point_mask: point mask. (True at nuclear point, False at background)
    @Return: pseudo label.
    """

    out_dict = {}

    distance_based_label = get_voronoi_edges(point_mask, extra_out=out_dict)
    if v2:
        color_based_label = get_cluster_label_v2(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], distance_based_label, typed_point_map)
    else:
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
    parser.add_argument("-m", "--masked-clustering", default=False, action="store_true",
                        help="use masked images to do clustering")
    args = parser.parse_args()

    IDX = args.index
    SPLIT = args.split
    EXP_NAME = args.name

    dataset = CoNSeP(download=False)
    ori = dataset.read_image(IDX, SPLIT)
    # point_mask = dataset.read_points(IDX, SPLIT)
    lab, type_ = dataset.read_labels(IDX, SPLIT)
    point_mask = get_point_from_instance(lab, binary=True)

    from skimage.io import imsave

    with OutTime():
        label = gen_pseudo_label(ori, point_mask, np.where(point_mask, type_, 0), v2=args.masked_clustering)
        draw_boundaries(ori, label)
    point_mask = binary_dilation(point_mask, disk(3))
    # point_mask = np.where(point_mask, 255, 0)
    ori[point_mask] = [255, 255, 0]

    # imsave("points.png", point_mask.astype(np.uint8))
    imsave("pseudo_label{}.png".format(EXP_NAME), ori)
