from collections.abc import Callable, Iterable, Mapping
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.color import label2rgb
from skimage.io import imsave
from skimage.morphology import dilation, disk
from constants import LABEL_COLOR_LIST

class Cascade:
    """
    Used to create a sequential function.
    Useful when conducting morphological operations.
    """
    def __init__(self, intermediate_prefix=None):
        """
        Args:
            intermediate_prefix (str): prefix to use when saving intermediate results.
        """
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

def broadcastable(func):
    def func_(*args, **kwargs):
        return [func(arg, **kwargs) for arg in args]
    return func_

def image_to_save(im):
    """
    Prepare an image for saving.
    """
    if im.dtype == bool:
        return np.where(im, 255, 0).astype("uint8")
    else:
        return im.astype("uint8")

def get_point_from_instance(inst, ignore_size=10, binary=False):
    """
    Args:
        inst (numpy.ndarray[int]): instance map of nuclei. (integers larger than 0 indicates the nuclear instance)
        ignore_size (int): minimum size of a nuclei
        binary (bool): annotate point with binary value? Otherwise, with integer value.
    Returns:
        point_map (numpy.ndarray[int / bool if binary]): point map of nuclei. (integers larger than 0 indicates the nuclear instance)
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
    Args:
        pt_lbls (numpy.ndarray[int]): point labels. (integers larger than 0 indicates the nuclear instance)
    Returns:
        point mask (numpy.ndarray[bool]). (True at nuclear point, False at background)
    """
    return pt_lbls > 0

def get_center(coords, mode="extrema"):
    """
    Find center from list of coordinates.
    Args:
        coords (numpy.ndarray[int]{n_coordinates, n_dimensions}): list of coordinates.
        mode (str): mode to compute center.
            'extrema': mean of extreme values.
            'mass': mass center.
    Returns:
        center (tuple[int]): coordinate of center.
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

def gen_distance(inst):
    """
    TODO: deprecated: distance map already computed in Voronoi label
    Generate distance map from instance labels.
    Args:
        inst (numpy.ndarray[int]): instance labels. (integers larger than 0 indicates the nuclear instance)
    Returns:
        distance_map (numpy.ndarray[float]): distance map.
    """
    pt_lbls = get_point_from_instance(inst)
    return distance_transform_edt(pt_lbls == 0)

def normalize_image(image):
    """
    Normalize an image for saving as an image file.
    Args:
        image (numpy.ndarray[uint8]): the image to normalize.
    Returns:
        normalized_image (numpy.ndarray[float]): normalized image.
    """
    M, m = image.max(), image.min()
    image = (image - m) / (M - m)
    return image

def get_gradient(image):
    """
    Compute the magnitude of gradient of the image.
    Args:
        image (numpy.ndarray[any]): the image to compute gradient.
    Returns:
        gradient (numpy.ndarray[float]): the gradient magnitude.
    """
    gx, gy = np.gradient(image)
    gradient = (gx**2 + gy**2)**(0.5)
    return gradient

def draw_boundaries(image, mask, color=[0, 255, 0]):
    """
    Draw boundaries of mask with given color on image.
    Args:
        image (numpy.ndarray[uint8]): the image to draw.
        mask (numpy.ndarray[float]): the mask.
        color (number, sequence[number]): the color used to indicate the boundaries.
    """
    if isinstance(color, (int, float)):
        assert 0 <= color <= 255, "Invalid color."
    elif isinstance(color, (list, tuple)):
        assert len(color) == 3 or len(color) == 1, "Invalid color length."
        for c in color:
            assert 0 <= c <= 255, "Invalid color."
    if len(image.shape) == 2:
        image = image[..., None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=-1)
    mask = np.where(mask, 255, 0)
    gradient = get_gradient(mask)
    image[gradient > 0] = color
    return image

def get_label_boundaries(labeled_seg, d=1):
    gd = get_gradient(labeled_seg)
    labeled_seg[gd == 0] = 0
    labeled_seg = dilation(labeled_seg, disk(d))
    return labeled_seg

def random_sub_list(l, bound=3):
    length = len(l)
    st = np.random.randint(0, bound)
    return l[st:]

def draw_label_boundaries(ori, labeled_seg, d=1):
    img = ori.copy()
    labeled_seg = get_label_boundaries(labeled_seg, d=d)
    colors = random_sub_list(LABEL_COLOR_LIST)
    rgb_labeled_seg = (label2rgb(labeled_seg, colors=colors) * 255).astype(np.uint8)
    labeled_seg = labeled_seg == 0
    labeled_seg = labeled_seg[..., None]
    labeled_seg = np.concatenate((labeled_seg, labeled_seg, labeled_seg), axis=2)
    labeled_seg = np.where(labeled_seg, img, rgb_labeled_seg)
    return labeled_seg

def get_valid_view(image, patch_size=270, valid_size=80, requested_size=1000):
    """
    Crop the image into only valid region according to the patch size and valid size.
    Args:
        image (numpy.ndarray[any]): the image.
        patch_size (int): the patch size.
        valid_size (int): the valid size.
    Returns:
        cropped_image (numpy.ndarray[any])
    """
    h, w = image.shape[:2]
    offset = (patch_size - valid_size) // 2
    rows = int((h - patch_size) / valid_size + 1)
    cols = int((w - patch_size) / valid_size + 1)
    y_end = offset + rows * valid_size
    x_end = offset + cols * valid_size
    return image[offset: y_end, offset: x_end, ...][:requested_size, :requested_size, ...]

def scale(img, vmax, vmin):
    """
    Scale an image into [vmin, vmax]. (Positive and negative values are scaled independently.)
    ----- img no copy -----
    Args:
        img (numpy.ndarray[any]): the image.
        vmax (number): maximum value.
        vmin (number): minimum value.
    Returns:
        scaled_image (numpy.ndarray[any])
    """
    # img = img.copy()
    max_ = img.max() 
    min_ = img.min() 
    if max_ != 0:
        img[img > 0] *= (vmax / max_)
    if min_ != 0: 
        img[img < 0] *= (vmin / min_)
    return img

def shift_and_scale(img, vmax, vmin):
    """
    Scale an image into [vmin, vmax]. (with shifting)
    Args:
        img (numpy.ndarray[any]): the image.
        vmax (number): maximum value.
        vmin (number): minimum value.
    Returns:
        scaled_image (numpy.ndarray[any])
    """
    img = img.copy()
    max_ = img.max()
    min_ = img.min()
    rang = max_ - min_
    vrang = vmax - vmin
    img -= (min_ - vmin)
    img *= (vrang / rang)
    return img

@broadcastable
def show(img):
    """
    Plot an image.
    Args:
        img (numpy.ndarray[any]): the image.
    """
    plt.imshow(img)
    plt.show()
