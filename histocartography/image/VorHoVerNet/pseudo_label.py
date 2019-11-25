from utils import booleanize_point_labels, get_point_from_instance
from color_label import get_cluster_label
from Voronoi_label import get_voronoi_edges

def gen_pseudo_label(image, instance_label):
    """
    Generate pseudo label from image and instance label.
    1. Generate distance based label.
    2. Generate color based label.
    3. Combine these two labels.
    @image: image.
    @instance_label: instance label.
    @Return: pseudo label.
    """
    point_mask = booleanize_point_labels(get_point_from_instance(instance_label))

    out_dict = {}

    distance_based_label = get_voronoi_edges(point_mask, extra_out=out_dict)
    color_based_label = get_cluster_label(image, out_dict["dist_map"], point_mask, out_dict["Voronoi_cell"])

    # TODO: combine the labels

    return None
