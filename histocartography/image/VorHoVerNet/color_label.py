import multiprocessing as mp
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2hed
from skimage.morphology import *
from sklearn.cluster import KMeans
from performance import OutTime
from utils import Cascade, draw_boundaries, get_point_from_instance, get_gradient

CLUSTER_FEATURES = "CD" # C: rgb, D: distance, S: he

def get_cluster_label(image, distance_map, point_mask, cells, edges):
    """
    Compute color-based label from original image, distance map, and point mask.
    @image: original image.
    @distance_map: distance map.
    @point_mask: point mask. (True at nuclear point, False at background)
    @cells: cells mask. (np.ndarray<bool>)
    @edges: voronoi edges. (255 indicates edge, 0 indicates background)
    """
    clusters = get_clusters(image, distance_map)
    from skimage.io import imsave
    nuclear_index, background_index = find_nuclear_cluster(clusters, point_mask)
    imsave("test cases/cluster.png", np.where(clusters == nuclear_index, 255, 0).astype("uint8"))
    return refine_cluster(clusters == nuclear_index, clusters == background_index, cells, point_mask, edges)

def get_cluster_label_v2(image, distance_map, point_mask, cells, edges, typed_point_map):
    """
    Compute color-based label from original image, distance map, and point mask.
    For each image masked by dilated point mask do clustering.
    @image: original image.
    @distance_map: distance map.
    @point_mask: point mask. (True at nuclear point, False at background)
    @cells: cells mask. (np.ndarray<bool>)
    @edges: voronoi edges. (255 indicates edge, 0 indicates background)
    """
    circle_mask = distance_map < 1000
    types = np.unique(typed_point_map)
    FULL = len(types) - 1
    lock = mp.Lock()
    queue = mp.Queue(FULL)

    for type_ in types:
        if type_ == 0: continue
        cell_indices = cells[typed_point_map == type_]
        mask = circle_mask & np.logical_or.reduce([cells == cell_index for cell_index in np.unique(cell_indices)])
        p = mp.Process(target=_get_cluster_label_v2, args=(mask, image, distance_map, typed_point_map == type_, lock, queue))
        p.daemon = True
        p.start()

    fnsd = 0
    all_mask = np.zeros_like(point_mask)

    while mp.active_children():
        nuclear = queue.get()
        all_mask = all_mask | nuclear
        fnsd += 1
        if fnsd == FULL:
            break

    return refine_cluster(all_mask, ~all_mask, cells, point_mask, edges)

def _get_cluster_label_v2(mask, image, distance_map, type_points, lock, queue):
    mask3d = mask[..., None]
    mask3d = np.repeat(mask3d, 3, axis=2)
    masked_image = np.where(mask3d, image, 0)
    masked_distance_map = np.where(mask, distance_map, -1)
    clstrs = get_clusters(masked_image, masked_distance_map, k=4)
    nuclear_index, _ = find_nuclear_cluster(clstrs, type_points)
    nuclear = clstrs == nuclear_index
    lock.acquire()
    queue.put(nuclear)
    lock.release()

def concat_normalize(*features):
    """
    Concatenate features after normalization.
    @features: feature matrix and maximum pairs. List[Tuple[np.ndarray [h, w, n_features]], Number]
    @Return: resulted feature matrix.
    """
    features_ = []
    for feature, maximum in features:
        if len(feature.shape) == 2:
            feature = feature[..., None]
        features_.append(feature / maximum)
    return np.concatenate(features_, axis=2)

def find_nuclear_cluster(clstrs, point_mask):
    """
    Find the cluster with maximum overlaps with point labels.
    @clstrs: clusters. (integer indicates the cluster index)
    @point_mask: point mask. (True at nuclear point, False at background)
    @Return: nuclear cluster index, background cluster index.
    """
    overlaps = [np.count_nonzero(point_mask & (clstrs == i)) for i in range(int(clstrs.max()) + 1)]
    nclr_clstr_idx = np.argmax(overlaps)
    dilated_pt_mask = binary_dilation(point_mask, disk(5))
    overlaps = [np.count_nonzero(dilated_pt_mask & (clstrs == i)) for i in range(int(clstrs.max()) + 1)]
    bkgrd_clstr_idx = np.argmin(overlaps)
    assert nclr_clstr_idx != bkgrd_clstr_idx, "Image is invalid"
    return nclr_clstr_idx, bkgrd_clstr_idx

def get_features(image, distance_map):
    """
    Compute features from original image and distance map.
    @image: original image.
    @distance_map: distance map.
    @Return: feature maps.
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
    @image: original image.
    @distance_map: distance map.
    @Return: cluster map.
    """
    features = get_features(image, distance_map)
    KM = KMeans(n_clusters=k, random_state=0)
    clstrs = KM.fit_predict(features.reshape(-1, features.shape[2])).reshape(image.shape[:2])
    return clstrs

def refine_cluster(nuclei, background, cells, point_mask, edges):
    """
    Refine clustering result with information from Voronoi diagram.
    @nuclei: nuclei mask. (np.ndarray<bool>)
    @background: background mask. (np.ndarray<bool>)
    @cells: cell indices. (integer indicates the cell index)
    @point_mask: point mask. (True at nuclear point, False at background)
    @edges: voronoi edges. (255 indicates edge, 0 indicates background)
    """
    """refine nuclei"""

    refined_nuclei = Cascade(intermediate_prefix="test cases/cluster")\
                        .append(remove_small_objects, 30)\
                        .append(binary_dilation, disk(3))\
                        .append(binary_fill_holes)\
                        .append(binary_erosion, disk(3))\
                        .append("__or__", binary_dilation(point_mask, disk(10)))\
                        (nuclei)
    # .append("__and__", edges == 0)\ 152
#     .append(binary_erosion, disk(2))\
    """refine background"""

    refined_background = background & (~refined_nuclei) & (~binary_dilation(point_mask, disk(10)))

    res = np.zeros(nuclei.shape + (3,), dtype=np.uint8)

    res[refined_nuclei] = [0, 255, 0]
    res[refined_background] = [255, 0, 0]

    return res

def main():
    from dataset_reader import CoNSeP
    from Voronoi_label import get_voronoi_edges
    from argparse import ArgumentParser
    import re

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
    parser.add_argument("-m", "--masked-clustering", default=False, action="store_true",
                        help="use masked images to do clustering")
    args = parser.parse_args()

    global CLUSTER_FEATURES
    features = re.sub("[^CDS]", "", args.features)
    CLUSTER_FEATURES = CLUSTER_FEATURES if features == "" else features

    IDX = args.index
    SPLIT = args.split
    EXP_NAME = args.name

    dataset = CoNSeP(download=False)
    image = dataset.read_image(IDX, SPLIT)
    lab, type_ = dataset.read_labels(IDX, SPLIT)
    point_mask = get_point_from_instance(lab, binary=True)
    typed_point_map = np.where(point_mask, type_, 0)
    # point_mask = dataset.read_points(IDX, SPLIT)

    out_dict = {}

    edges = get_voronoi_edges(point_mask, extra_out=out_dict)

    with OutTime():
        if args.masked_clustering:
            color_based_label = get_cluster_label_v2(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], edges, typed_point_map)
        else:
            color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], edges)

    import matplotlib.pyplot as plt

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
