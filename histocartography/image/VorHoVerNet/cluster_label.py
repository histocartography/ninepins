import numpy as np
# import matplotlib.pyplot as plt
# from skimage.color import rgb2hed
# from skimage.measure import find_contours
from skimage.morphology import closing, opening, binary_dilation, disk
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
    # nclr = np.where(clstrs == nclr_clstr_idx, 255, 0)
    # bkgrd = np.where(clstrs == bkgrd_clstr_idx, 255, 0)
    # clstrs[pt_lbls > 0] = 128
    # nclr = cascade_func(nclr, opening, closing)
    # contours = find_contours(clstrs, 0.5)
    
    # for c in contours:
    #     for pt in c:
    #         y, x = map(int, pt)
    #         image[y, x] = [0, 255, 0]

    # imsave("cluster.png", (clstrs).astype(np.uint8))
    # imsave("testtestst.png", image)
    return clstrs

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
    _, distance_map = get_voronoi_edges(point_mask)

    with OutTime():
        clstrs = get_clusters(image, distance_map)
        nclr_clstr_idx, bkgrd_clstr_idx = find_nuclear_cluster(clstrs, point_mask)

if __name__ == "__main__":
    main()