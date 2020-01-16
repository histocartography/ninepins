import numpy as np
from histocargraphy.image.VorHoVerNet.performance import OutTime

def get_voronoi_edges(point_mask, view=False, extra_out=None, l2_norm=True):
    '''
    Compute Voronoi edges from point mask.
    Args:
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        view (bool): whether to display output in plot window.
        extra_out (dict / None): dictionary to store additional output or no additional output would be stored if set to None.
            Results:
            'dist_map' (numpy.ndarray[float]): distance map.
            'Voronoi_cell' (numpy.ndarray[int]): cell colormap. (positive integer indicates instance index.)
        l2_norm (bool): whether to use L2 norm as distance.
    Returns:
        edges (numpy.ndarray[int]): voronoi edges. (255 indicates edge, 0 indicates background)
    Note: 'point' means one pixel per point.
    Reference: some codes from https://gist.github.com/bert/1188638
    '''
    if l2_norm:
        def dist_func(row, col):
            nonlocal r, c
            return (row - r)**2 + (col - c)**2
    else:
        def dist_func(row, col):
            nonlocal r, c
            return abs(row - r) + abs(col - c)
    
    # create voronoi diagram (color_map)
    shape = point_mask.shape[:2]
    color_map = np.zeros(shape, np.int32)
    depth_map = np.ones(shape, np.float32) * 1e308
    points = np.argwhere(point_mask)
    points = np.random.permutation(points) if view else points
    for i, (r, c) in enumerate(points):
        dist_map = np.fromfunction(dist_func, shape)
        color_map = np.where(dist_map < depth_map, i + 1, color_map)
        depth_map = np.where(dist_map < depth_map, dist_map, depth_map)

    # get voronoi edges
    edges = np.zeros(shape, dtype=np.int32)
    edges[:-1, :][color_map[1:, :] != color_map[:-1, :]] = 255
    edges[:, :-1][color_map[:, 1:] != color_map[:, :-1]] = 255

    if view:
        import matplotlib.pyplot as plt
        import cv2
        # global ori
        fig, ax = plt.subplots(1, 2)
        for r, c in points:
            cv2.circle(color_map, (c, r), 2, -255, -1)
        # ax[0].imshow(ori)
        ax[0].imshow(color_map)
        ax[1].imshow(edges)
        plt.show()

    if extra_out is not None:
        assert isinstance(extra_out, dict), "{} is not a dictionary.".format(extra_out)
        extra_out["dist_map"] = depth_map
        extra_out["Voronoi_cell"] = color_map

    return edges

def main():
    from dataset_reader import CoNSeP
    IDX = 1
    SPLIT = 'train'
    dataset = CoNSeP()
    ori = dataset.read_image(IDX, SPLIT)
    
    with OutTime():
        point_mask = dataset.read_points(IDX, SPLIT)
    
    with OutTime():
        get_voronoi_edges(point_mask, view=True)

if __name__ == "__main__":
    main()
