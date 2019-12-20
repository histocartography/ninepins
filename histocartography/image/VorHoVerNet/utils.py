import numpy as np
from skimage.io import imsave
from scipy.ndimage.morphology import distance_transform_edt
from collections.abc import Mapping, Iterable, Callable

class Cascade:
    """
    Used to create a sequential function.
    Useful when conducting morphological operations.
    """
    def __init__(self, intermediate_prefix=None):
        self.funcs = []
        self.save_intermediate = intermediate_prefix is not None
        self.intermediate_prefix = intermediate_prefix

    def append(self, func, *args, **kwargs):
        """
        Append a function with arguments (additional to first argument).
        If @func is a string, call a method of the input instead of calling @func with the input as its first argument.
        """
        self.funcs.append((func, args, kwargs))
        return self

    def __call__(self, x):
        """
        Conduct sequence of functions on the input @x.
        """
        if self.save_intermediate:
            i = 0
        for func, args, kwargs in self.funcs:
            if isinstance(func, str):
                x = getattr(x, func)(*args, **kwargs)
            elif isinstance(func, Callable):
                x = func(x, *args, **kwargs)
            if self.save_intermediate:
                try: 
                    imsave("{}_{}.png".format(self.intermediate_prefix, i), image_to_save(x))
                    i += 1
                except Exception as e:
                    print(str(e))
        return x

def image_to_save(im):
    if im.dtype == bool:
        return np.where(im, 255, 0).astype("uint8")
    else:
        return im.astype("uint8")

def get_point_from_instance(inst, ignore_size=10, binary=False):
    """
    @inst: instance map of nuclei. (integers larger than 0 indicates the nuclei instance)
    @ignore_size: minimum size of a nuclei
    @Return: point map of nuclei. (integers larger than 0 indicates the nuclei instance)
    """
    max_inst_idx = int(inst.max())
    point_map = np.zeros_like(inst)
    if binary:
        point_map = point_map.astype(bool)
    for inst_idx in range(1, max_inst_idx + 1):
        coords = np.argwhere(inst == inst_idx)
        if coords.shape[0] <= ignore_size:
            continue
        if binary:
            point_map[get_center(coords)] = True
        else:
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

def get_gradient(image):
    """
    Compute the magnitude of gradient of the image.
    @image: the image to compute gradient.
    @Return: the gradient magnitude.
    """
    gx, gy = np.gradient(image)
    gradient = (gx**2 + gy**2)**(0.5)
    return gradient

def draw_boundaries(image, mask, color=[0, 255, 0]):
    """
    Draw boundaries of mask with given color on image.
    @image: the image to draw.
    @mask: the mask.
    @color: the color used to indicate the boundaries.
    @Return: <None>
    """
    if isinstance(color, (int, float)):
        assert 0 <= color <= 255, "Invalid color."
    elif isinstance(color, (list, tuple)):
        assert len(color) == 3 or len(color) == 1, "Invalid color length."
        for c in color:
            assert 0 <= c <= 255, "Invalid color."
    mask = np.where(mask, 255, 0)
    gradient = get_gradient(mask)
    image[gradient > 0] = color

def scale(img, vmax, vmin):
    max_ = img.max()
    min_ = img.min() if img.min() != 0 else 1
    img[img > 0] *= (vmax / max_)
    img[img < 0] *= (vmin / min_)
    return img
    