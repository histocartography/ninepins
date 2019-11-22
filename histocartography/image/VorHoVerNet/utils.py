import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from collections.abc import Mapping, Iterable

def cascade_funcs(x, *funcs, arg_lists=[], kwarg_lists=[]):
    """
    Make @x go through list of functions.
    Useful when conducting morphological operations.
    @x: input to go through the functions.
    @funcs: list of functions. List[Callable]
    @arg_lists: args of each function. Iterable[Iterable]
    @kwarg_lists: kwargs of each function. Iterable[Mapping]
    """
    assert isinstance(arg_lists, Iterable), "{} is not iterable".format(arg_lists)
    assert isinstance(kwarg_lists, Iterable), "{} is not iterable".format(kwarg_lists)
    if len(arg_lists) < len(funcs):
        arg_lists += [tuple()] * (len(funcs) - len(arg_lists))
    if len(kwarg_lists) < len(funcs):
        kwarg_lists += [dict()] * (len(funcs) - len(kwarg_lists))
    for func, args, kwargs in zip(funcs, arg_lists, kwarg_lists):
        assert isinstance(args, Iterable), "{} is not iterable".format(args)
        assert isinstance(kwargs, Mapping), "{} is not a mapping".format(kwargs)
        x = func(x, *args, **kwargs)
    return x

def get_point_from_instance(inst, ignore_size=5):
    """
    @inst: instance map of nuclei. (integers larger than 0 indicates the nuclei instance)
    @Return: point map of nuclei. (integers larger than 0 indicates the nuclei instance)
    """
    max_inst_idx = int(inst.max())
    point_map = np.zeros_like(inst)
    for inst_idx in range(1, max_inst_idx + 1):
        coords = np.argwhere(inst == inst_idx)
        if coords.shape[0] <= ignore_size:
            continue
        point_map[get_center(coords)] = inst_idx
    return point_map

def booleanize_point_labels(pt_lbls):
    """
    @pt_lbls: point labels. (integers larger than 0 indicates the nuclei instance)
    @Return: point mask. (True at nuclear point, False at background)
    """
    return pt_lbls > 0

def get_center(coords, mode="extrema"):
    """
    Find center from list of coordinates. (mean of extreme values)
    @coords: list of coordinates. np.ndarray [n_coordinates, n_dimensions]
    @Return: coordinate of center. Tuple[int]
    """
    if mode == "extrema":
        center = []
        for d in range(coords.shape[1]):
            coords_d = coords[:, d]
            M, m = np.max(coords_d), np.min(coords_d)
            center.append(int((M + m) / 2))
        return tuple(center)
    elif mode == "mass":
        return coords.sum(axis=0) / coords.shape[0]
    else:
        raise ValueError("Unknown mode")

def gen_distance_map(pt_lbls):
    """
    TODO: deprecated: distance map already computed in Voronoi label
    Generate distance map from point_labels.
    @pt_lbls: point labels. (integers larger than 0 indicates the nuclei instance)
    @Return: distance map.
    """
    return distance_transform_edt(pt_lbls == 0)

def gen_distance(inst):
    """
    TODO: deprecated: distance map already computed in Voronoi label
    Workflow of generating distance map from instance labels.
    @inst: instance labels. (integers larger than 0 indicates the nuclei instance)
    @Return: distance map.
    """
    pt_lbls = get_point_from_instance(inst)
    return gen_distance_map(pt_lbls)

def normalize_image(image):
    """
    Normalize an image for saving as an image file.
    @image: the image to normalize.
    @Return: normalized image.
    """
    M, m = image.max(), image.min()
    image = (image - m) / (M - m)
    return image