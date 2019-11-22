from utils import get_point_from_instance, booleanize_point_labels
from Voronoi_label import get_voronoi_edges
from cluster_label import get_clusters, find_nuclear_cluster

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

    distance_based_label, distance_map = get_voronoi_edges(point_mask)
    
    clusters = get_clusters(image, distance_map)
    nuclear_index, background_index = find_nuclear_cluster(clusters, point_mask)

    color_based_label = np.zeros_like(clusters)
    color_based_label[clusters == nuclear_index] = 255
    color_based_label[clusters == background_index] = 128
    color_based_label = color_based_label.astype(np.uint8)

    # TODO: combine the labels

    return None