from collections.abc import Callable, Iterable, Mapping
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage.color import label2rgb
from skimage.io import imsave
from skimage.morphology import dilation, disk
from histocartography.image.VorHoVerNet.constants import LABEL_COLOR_LIST

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
        if len(args) > 1:
            return [func(arg, **kwargs) for arg in args]
        else:
            return func(*args, **kwargs)
    return func_

def image_to_save(im):
    """
    Prepare an image for saving.
    """
    if im.dtype == bool:
        return np.where(im, 255, 0).astype("uint8")
    else:
        return im.astype("uint8")

def neighbors(shape, coord):
    h, w = shape[:2]
    y, x = coord
    res = []
    if y - 1 >= 0:
        res.append((y-1, x))
    if y + 1 < h:
        res.append((y+1, x))
    if x - 1 >= 0:
        res.append((y, x-1))
    if x + 1 < w:
        res.append((y, x+1))
    return res

def BFS(map_, seed, tar_val):
    visited = np.zeros_like(map_).astype(bool)
    Q = deque()
    Q.append(seed)
    visited[seed] = True
    while Q:
        cur = Q.popleft()
        if map_[cur] == tar_val:
            return cur
        else:
            for n in neighbors(map_.shape, cur):
                if not visited[n]:
                    Q.append(n)
                    visited[n] = True
    raise ValueError(f"This map doesn't contain such value: {tar_val}")

def get_point_from_instance(inst, ignore_size=10, binary=False, center_mode="centroid", perturb_func=None, **kwargs):
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
        center = get_center(coords, mode=center_mode)
        if inst[center] != inst_idx:
            center = BFS(inst, center, inst_idx)
        if perturb_func is not None:
            center = perturb_func(center, inst, inst_idx, coords, **kwargs)
        if binary:
            point_map[center] = True
        else:
            point_map[center] = inst_idx
    return point_map

def random_normal_shift(center, inst, inst_idx, coords, max_retry=10, eta=0.8, verbose=False):
    y, x = center
    horizontal_line = [c[1] for c in coords if c[0] == y]
    x_min, x_max = min(horizontal_line), max(horizontal_line)
    vertical_line = [c[0] for c in coords if c[1] == x]
    y_min, y_max = min(vertical_line), max(vertical_line)
    x_perturb = min(x_max - x, x - x_min)
    y_perturb = min(y_max - y, y - y_min)
    x_perturb = eta * x_perturb
    y_perturb = eta * y_perturb
    retry_count = 0
    while retry_count < max_retry:
        # probability of samples lying beyond 3 stds from center is almost 0.
        new_x = np.random.normal(x, x_perturb / 3)
        new_y = np.random.normal(y, y_perturb / 3)
        if abs(new_x - x) < x_perturb and abs(new_y - y) < y_perturb and inst[int(new_y), int(new_x)] == inst_idx:
            # if the sampled set of coordinates is beyond the perturbation range or outside the instance region, re-draw.
            new_x = int(new_x)
            new_y = int(new_y)
            break
        retry_count += 1
    else:
        if verbose:
            print(f"too many retries, give up perturbating on instance: {inst_idx}")
        new_x = x
        new_y = y
    return new_y, new_x

def get_random_shifted_point_from_instance(inst, eta, ignore_size=10, binary=False, center_mode="centroid", max_retry=10, verbose=False):
    """
    Args:
        inst (numpy.ndarray[int]): instance map of nuclei. (integers larger than 0 indicates the nuclear instance)
        eta (float): perturbation ratio.
        ignore_size (int): minimum size of a nuclei.
        binary (bool): annotate point with binary value? Otherwise, with integer value.
    Returns:
        point_map (numpy.ndarray[int / bool if binary]): point map of nuclei. (integers larger than 0 indicates the nuclear instance)
    """
    if eta == 0:
        return get_point_from_instance(inst, ignore_size=ignore_size, binary=binary, center_mode=center_mode)
    assert 0 < eta <= 1, "Perturbation ratio should lie in [0, 1]."
    return get_point_from_instance(inst, ignore_size=ignore_size, binary=binary, center_mode=center_mode, perturb_func=random_normal_shift, eta=eta, max_retry=max_retry, verbose=verbose)

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
    elif mode == "centroid":
        return tuple((coords.sum(axis=0) / coords.shape[0]).astype(int))
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
def show(img, save_name=None, figure_settings={}, save_settings={}):
    """
    Plot an image.
    Args:
        img (numpy.ndarray[any]): the image.
    """
    plt.figure(**figure_settings)
    plt.imshow(img)
    if save_name is not None:
        plt.savefig(save_name, **save_settings)
    plt.show()
