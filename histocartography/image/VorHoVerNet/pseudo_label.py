from utils import get_point_from_instance, draw_boundaries
from color_label import get_cluster_label
from Voronoi_label import get_voronoi_edges

def gen_pseudo_label(image, point_mask):
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
    color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"], distance_based_label)

    nuclei = (color_based_label == [0, 255, 0]).all(axis=2)

    draw_boundaries(image, nuclei, color=[0, 255, 255])

    color_based_label[nuclei] = [0, 0, 0]
    from skimage.morphology import watershed, label
    import numpy as np
    from utils import get_gradient
    labeled_nuclei = label(nuclei)
    mask = np.zeros_like(nuclei)
    for point in np.argwhere(point_mask):
        point = tuple(point)
        if labeled_nuclei[point] == 0: continue
        mask = mask | (labeled_nuclei == labeled_nuclei[point])

    draw_boundaries(image, mask)
    
    from skimage.io import imsave

    imsave("img.png", image)

    return None

if __name__ == "__main__":
    from dataset_reader import CoNSeP
    IDX = 6
    SPLIT = 'test'
    dataset = CoNSeP(download=False)
    ori = dataset.read_image(IDX, SPLIT)
    point_mask = dataset.read_points(IDX, SPLIT)
    # point_mask = booleanize_point_labels(get_point_from_instance(dataset.read_labels(IDX, SPLIT)[0]))
    from skimage.morphology import binary_dilation, disk
    from skimage.io import imsave
    import numpy as np

    gen_pseudo_label(ori, point_mask)
    point_mask = binary_dilation(point_mask, disk(3))
    point_mask = np.where(point_mask, 255, 0)

    imsave("points.png", point_mask)