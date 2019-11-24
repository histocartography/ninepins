import numpy as np
from utils import get_point_from_instance, booleanize_point_labels
from performance import OutTime

def get_voronoi_edges(point_mask, view=False, extra_out=None):
    '''
    Compute Voronoi edges from point mask.
    @point_mask: point mask. (True at nuclear point, False at background)
    @view: whether to display output in plot window.
    @extra_out: dictionary to store additional output. (must be a dictionary.)
    @Return: voronoi edges (255 indicates edge, 0 indicates background), distance map
    Note: 'point' means one pixel per point.
    Reference: some codes from https://gist.github.com/bert/1188638
    '''
    def dist_func(row, col):
        nonlocal r, c
        return (row - r)**2 + (col - c)**2
    
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
    lab, _ = dataset.read_labels(IDX, SPLIT)

    with OutTime():
        mass_map = booleanize_point_labels(get_point_from_instance(lab, ignore_size=6))
    
    with OutTime():
        get_voronoi_edges(mass_map, view=True)

if __name__ == "__main__":
    main()
