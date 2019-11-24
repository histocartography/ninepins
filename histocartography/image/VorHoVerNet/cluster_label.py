import numpy as np
from skimage.morphology import *
from scipy.ndimage.morphology import binary_fill_holes
from sklearn.cluster import KMeans
from utils import cascade_funcs, booleanize_point_labels, get_point_from_instance
from performance import OutTime

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
    """
    overlaps = [np.count_nonzero(point_mask & (clstrs == i)) for i in range(int(clstrs.max()) + 1)]
    nclr_clstr_idx = np.argmax(overlaps)
    dilated_pt_mask = binary_dilation(point_mask, disk(5))
    overlaps = [np.count_nonzero(dilated_pt_mask & (clstrs == i)) for i in range(int(clstrs.max()) + 1)]
    bkgrd_clstr_idx = np.argmin(overlaps)
    assert nclr_clstr_idx != bkgrd_clstr_idx, "Image is invalid"
    return nclr_clstr_idx, bkgrd_clstr_idx

def get_clusters(image, distance_map):
    """
    Compute clusters from original image and distance map.
    @image: original image.
    @distance_map: distance map.
    @Return: cluster map, nuclear cluster index, background index
    """
    distance_map = np.clip(distance_map, a_min=0, a_max=20)
    features = concat_normalize((image, 10), (distance_map, 1))
    KM = KMeans(n_clusters=3, random_state=0)
    clstrs = KM.fit_predict(features.reshape(-1, features.shape[2])).reshape(image.shape[:2])
    return clstrs

def refine_cluster(nuclei, background, cells, point_mask):
    """
    """
    # refine nuclei

    nuclei = remove_small_objects(nuclei, 30)
    cells = dilation(cells, disk(2))
    refined_nuclei = np.zeros(nuclei.shape, dtype=bool)

    max_cell_index = cells.max()
    for i in range(1, max_cell_index + 1):
        cell_nuclei = nuclei & (cells == i)
        cell_nuclei = cascade_funcs(cell_nuclei, binary_dilation, binary_fill_holes, binary_erosion, arg_lists=[[disk(5)], [], [disk(7)]])
        refined_nuclei = refined_nuclei | cell_nuclei

    # refine background

    refined_background = background & (~refined_nuclei) & (~binary_dilation(point_mask, disk(10)))

    res = np.zeros(nuclei.shape + (3,), dtype=np.uint8)

    res[refined_nuclei] = [0, 255, 0]
    res[refined_background] = [255, 0, 0]

    return res

def get_cluster_label(image, distance_map, point_mask, cells):
    clusters = get_clusters(image, distance_map)
    nuclear_index, background_index = find_nuclear_cluster(clusters, point_mask)
    return refine_cluster(clusters == nuclear_index, clusters == background_index, cells, point_mask)

def main():
    from dataset_reader import CoNSeP
    from Voronoi_label import get_voronoi_edges
    IDX = 4
    SPLIT = 'test'
    dataset = CoNSeP()
    image = dataset.read_image(IDX, SPLIT)
    lab, _ = dataset.read_labels(IDX, SPLIT)
    point_labels = get_point_from_instance(lab)
    point_mask = booleanize_point_labels(point_labels)

    out_dict = {}

    get_voronoi_edges(point_mask, extra_out=out_dict)

    with OutTime():
        color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"])

    import matplotlib.pyplot as plt

    nuclei = np.where((color_based_label == [0, 255, 0]).all(axis=2), 255, 0)
    gx, gy = np.gradient(nuclei)
    gradient = (gx**2 + gy**2)**(0.5)
    
    image[gradient > 0] = [0, 255, 0]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(color_based_label)
    plt.show()

if __name__ == "__main__":
    main()