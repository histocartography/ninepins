import multiprocessing as mp
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2hed
from skimage.morphology import *
from sklearn.cluster import KMeans
from histocartography.image.VorHoVerNet.performance import OutTime
from histocartography.image.VorHoVerNet.utils import Cascade, draw_boundaries, get_point_from_instance, get_gradient, show

CLUSTER_FEATURES = "CD"
# C: rgb, D: distance, S: he

def get_cluster_label(image, distance_map, point_mask, cells, edges, k=3):
    """
    Compute color-based label from original image, distance map, and point mask.
    Args:
        image (numpy.ndarray[uint8]): original image.
        distance_map (numpy.ndarray[float]): distance map.
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        cells (numpy.ndarray[int]): cells mask.
        edges (numpy.ndarray[int]): voronoi edges. (255 indicates edge, 0 indicates background)
    """
    clusters = get_clusters(image, distance_map, k=k)
    from skimage.io import imsave
    nuclear_index, background_index = find_nuclear_cluster(image, clusters, point_mask)
    # imsave("test cases/cluster.png", np.where(clusters == nuclear_index, 255, 0).astype("uint8"))
    return refine_cluster(clusters == nuclear_index, clusters == background_index, cells, point_mask, edges)

def concat_normalize(*features):
    """
    Concatenate features after normalization.
    Args:
        features (list[tuple[numpy.ndarray[any]{h, w, n_features}, number]]):
            feature matrix and maximum pairs.
            e.g. [(feature, 255), ...] where feature: numpy.ndarray[any]{h, w, n_features}
    Returns:
        concat_features (numpy.ndarray[any]{h, w, total_n_features}): resulted feature matrix.
    """
    features_ = []
    for feature, maximum in features:
        if len(feature.shape) == 2:
            feature = feature[..., None]
        features_.append(feature / maximum)
    return np.concatenate(features_, axis=2)

def find_nuclear_cluster(image, clstrs, point_mask):
    """
    Find the cluster with maximum overlaps with point labels.
    Args:
        clstrs (numpy.ndarray[int]): clusters. (integer indicates the cluster index)
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
    Returns:
        (tuple):
            nclr_clstr_idx (int): nuclear cluster index
            bkgrd_clstr_idx (int): background cluster index.
    """
    # dilated_pt_mask = binary_dilation(point_mask, disk(5))
    # overlaps = [np.count_nonzero(dilated_pt_mask & (clstrs == i)) for i in range(int(clstrs.max()) + 1)]
    # nclr_clstr_idx = np.argmax(overlaps)
    # bkgrd_clstr_idx = np.argmin(overlaps)
    image = rgb2hed(image)
    h_means = [image[clstrs == i][..., 0].mean() for i in range(int(clstrs.max()) + 1)]
    nclr_clstr_idx = np.argmax(h_means)
    bkgrd_clstr_idx = np.argmin(h_means)
    assert nclr_clstr_idx != bkgrd_clstr_idx, "Image is invalid"
    return nclr_clstr_idx, bkgrd_clstr_idx

def get_features(image, distance_map):
    """
    Compute features from original image and distance map.
    Args:
        image (numpy.ndarray[uint8]): original image.
        distance_map (numpy.ndarray[float]): distance map.
    Returns:
        feature_map (numpy.ndarray[float]): feature maps.
    """
    features = []
    for cluster_feature in CLUSTER_FEATURES:
        if cluster_feature == 'C':
            features.append((image, 10))
        if cluster_feature == 'D':
            features.append((np.clip(distance_map, a_min=0, a_max=20), 1))
        if cluster_feature == 'S':
            feature = get_gradient(rgb2hed(image)[..., 0] * 255)
            M, m = feature.max(), feature.min()
            features.append((feature - m, (M - m) / 20))
    features = concat_normalize(*features)
    return features

def get_clusters(image, distance_map, k=3):
    """
    Compute clusters from original image and distance map.
    Args:
        image (numpy.ndarray[uint8]): original image.
        distance_map (numpy.ndarray[float]): distance map.
    Returns:
        cluster_map (numpy.ndarray[int]): cluster map.
    """
    features = get_features(image, distance_map)
    KM = KMeans(n_clusters=k, random_state=0)
    clstrs = KM.fit_predict(features.reshape(-1, features.shape[2])).reshape(image.shape[:2])
    return clstrs

def refine_cluster(nuclei, background, cells, point_mask, edges):
    """
    Refine clustering result with information from Voronoi diagram.
    Args:
        nuclei (numpy.ndarray[bool]): nuclei mask.
        background (numpy.ndarray[bool]): background mask.
        cells: cell indices (numpy.ndarray[int]). (integer indicates the cell index)
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        edges (numpy.ndarray[int]): voronoi edges. (255 indicates edge, 0 indicates background)
    """

    """refine nuclei"""
    # intermediate_prefix="test cases/cluster"
    refined_nuclei = Cascade()\
                        .append(remove_small_objects, 30)\
                        .append("__or__", binary_dilation(point_mask, disk(5)))\
                        .append(binary_dilation, disk(1))\
                        .append(binary_fill_holes)\
                        .append(binary_erosion, disk(1))\
                        (nuclei)
    # .append("__and__", edges == 0)\ 152
    # .append(binary_erosion, disk(2))\
    """refine background"""

    refined_background = background & (~refined_nuclei) & (~binary_dilation(point_mask, disk(10)))

    res = np.zeros(nuclei.shape + (3,), dtype=np.uint8)

    res[refined_nuclei] = [0, 255, 0]
    res[refined_background] = [255, 0, 0]

    return res

def main():
    import dataset_reader
    from Voronoi_label import get_voronoi_edges
    from argparse import ArgumentParser
    import re
    import matplotlib.pyplot as plt

    parser = ArgumentParser(description="Cluster Label Generator (Experiment)")
    parser.add_argument("-f", "--features", default="CD",
                        help="features for clustering (sequence of characters in [C, S, D])\
                            C: RGB, D: distance, S: HE")
    parser.add_argument("-n", "--name", default="test",
                        help="experiment name")
    parser.add_argument("-i", "--index", default=4, type=int,
                        help="index of the image in dataset")
    parser.add_argument("-s", "--split", default="test", choices=["test", "train"],
                        help="split of the dataset ([test, train])")
    parser.add_argument("-d", "--dataset", default="CoNSeP", choices=["CoNSeP", "MoNuSeg"],
                        help="dataset ([CoNSeP, MoNuSeg])")
    parser.add_argument("-k", "--num-clusters", default=3, type=int,
                        help="number of clusters")
    args = parser.parse_args()

    global CLUSTER_FEATURES
    features = re.sub("[^CDS]", "", args.features)
    CLUSTER_FEATURES = CLUSTER_FEATURES if features == "" else features

    IDX = args.index
    SPLIT = args.split
    EXP_NAME = args.name

    dataset = getattr(dataset_reader, args.dataset)(download=False)
    image = dataset.read_image(IDX, SPLIT)
    lab, _ = dataset.read_labels(IDX, SPLIT)
    point_mask = get_point_from_instance(lab, binary=True)
    # point_mask = dataset.read_points(IDX, SPLIT)

    out_dict = {}

    edges = get_voronoi_edges(point_mask, extra_out=out_dict)

    with OutTime():
        color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], edges, k=args.num_clusters)

    mask = (color_based_label == [0, 255, 0]).all(axis=2)
    draw_boundaries(image, mask)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[1].imshow(color_based_label)
    # plt.show()
    
    from skimage.io import imsave

    imsave("img{}.png".format(EXP_NAME), image)
    imsave("cl{}.png".format(EXP_NAME), color_based_label)

if __name__ == "__main__":
    main()
