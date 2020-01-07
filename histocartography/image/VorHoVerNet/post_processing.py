from pathlib import Path
import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import label2rgb
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.io import imread, imsave
from skimage.morphology import *
from inference import shift_and_scale
from utils import Cascade, show
from Voronoi_label import get_voronoi_edges


def get_instance_output(from_file, *args, h=0.5, k=0.1, **kwargs):
    """
    Combine model output values from three branches into instance segmentation.
    Args:
        from_file (bool): read model output from file? Otherwise, obtain by feeding input into model.
        args (list[any]): arguments for get_output_from_*.
        h (float): threshold for the output map.
        k (float): threshold for the distance map.
        kwargs (dict[str: any]): keyword arguments for get_output_from_*.
    Returns:
        instance_map (numpy.ndarray[int]): instance map
    """
    """read files or inference to get individual output"""
    if from_file:
        seg, vet, hor = get_output_from_file(*args, **kwargs)
    else:
        seg, vet, hor = get_output_from_model(*args, **kwargs)

    """pre-process the output maps"""
    vet = shift_and_scale(vet, 1, 0)
    hor = shift_and_scale(hor, 1, 0)
    th_seg = seg > h

    """combine the output maps"""

    """ - generate distance map"""
    grad_vet = shift_and_scale(np.abs(sobel_v(vet)), 1, 0)
    grad_hor = shift_and_scale(np.abs(sobel_h(hor)), 1, 0)
    sig_diff = np.maximum(grad_vet, grad_hor)
    sig_diff = Cascade() \
                    .append(maximum_filter, size=3) \
                    .append(median_filter, size=3) \
                    (sig_diff)
    sig_diff[~th_seg] = 0
    
    # show(sig_diff)

    """ - generate markers"""
    markers = th_seg & (sig_diff <= k)
    # intermediate_prefix='markers/m'
    markers = Cascade() \
                    .append(binary_dilation, disk(3)) \
                    .append(binary_fill_holes) \
                    .append(binary_erosion, disk(3)) \
                    .append(remove_small_objects, min_size=5) \
                    (markers)
    markers = label(markers)
    
    # show(markers)

    """ - run watershed on distance map with markers"""
    res = watershed(sig_diff, markers=markers, mask=th_seg)

    """ - re-fill regions in thresholded nuclei map which do not have markers"""
    lbl_th_seg = label(th_seg)
    offset = int(markers.max())
    lbl_th_seg += offset
    lbl_th_seg[lbl_th_seg == offset] = 0
    
    res = np.where((res == 0) & (lbl_th_seg != 0), lbl_th_seg, res)

    return res

def get_output_from_file(idx, use_patch_idx=False, root='./inference', ckpt='model_003_ckpt_epoch_43', prefix='patch', patchsize=270, validsize=80, imagesize=1000):
    """
    Read output from file.
    Args:
        idx (int): index of image in dataset.
        use_patch_idx (bool): treat the index as patch index instead.
        root (str): root directory of the output files.
        ckpt (str): name of the checkpoint used to generate the output.
        prefix (str): prefix of the file names.
        patchsize (int): size of the input patch of model.
        validsize (int): size of the output patch of model.
        imagesize (int): size of the whole image in dataset.
    Returns:
        (tuple):
            seg (numpy.ndarray[float]): segmentation output.
            vet (numpy.ndarray[float]): vertical distance map output.
            hor (numpy.ndarray[float]): horizontal distance map output.
    """
    if isinstance(root, str):
        root = Path(root)
    assert isinstance(root, Path), "{} is not a Path object or string".format(root)

    if use_patch_idx:
        from_dir = root / ckpt / "{}{:04d}".format(prefix, idx)
        return np.load(str(from_dir / "seg.npy")), \
                np.load(str(from_dir / "dist1.npy")), \
                np.load(str(from_dir / "dist2.npy"))
    else:
        if isinstance(imagesize, int):
            imagesize = (imagesize, imagesize)
        h, w = imagesize[:2]

        rows = int((h - patchsize) / validsize + 1)
        cols = int((w - patchsize) / validsize + 1)
        base = (idx - 1) * rows * cols

        wholesize = (validsize * rows, validsize * cols)

        seg = np.zeros(wholesize, dtype=float)
        vet = np.zeros(wholesize, dtype=float)
        hor = np.zeros(wholesize, dtype=float)

        for i in range(rows):
            for j in range(cols):
                from_dir = root / ckpt / "{}{:04d}".format(prefix, base + i * cols + j + 1)
                seg_ = np.load(str(from_dir / "seg.npy"))
                vet_ = np.load(str(from_dir / "dist1.npy"))
                hor_ = np.load(str(from_dir / "dist2.npy"))
                
                seg[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = seg_
                vet[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = vet_
                hor[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = hor_

        return seg, vet, hor

def get_output_from_model(img, model):
    """
    Get output by feeding an image to a model.
    Args:
        img (torch.Tensor): the input image.
        model (torch.nn.Module): 
    Returns:
        (tuple):
            seg (numpy.ndarray[float]): segmentation output.
            vet (numpy.ndarray[float]): vertical distance map output.
            hor (numpy.ndarray[float]): horizontal distance map output.
    """
    model.eval()
    with torch.no_grad():
        pred = model(img)
    pred = pred.squeeze(0).detach().cpu().numpy()
    return pred[..., 0], \
            pred[..., 1], \
            pred[..., 2]

def get_original_image_from_file(idx, use_patch_idx=False, root='./inference', ckpt='model_003_ckpt_epoch_43', prefix='patch', patchsize=270, validsize=80, imagesize=1000):
    """
    Read original image from file.
    Args:
        idx (int): index of image in dataset.
        use_patch_idx (bool): treat the index as patch index instead.
        root (str): root directory of the output files.
        ckpt (str): name of the checkpoint used to generate the output.
        prefix (str): prefix of the file names.
        patchsize (int): size of the input patch of model.
        validsize (int): size of the output patch of model.
        imagesize (int): size of the whole image in dataset.
    Returns:
        img (numpy.ndarray[uint8]): original image
    """

    if isinstance(root, str):
        root = Path(root)
    assert isinstance(root, Path), "{} is not a Path object or string".format(root)
    if use_patch_idx:
        from_dir = root / ckpt / "{}{:04d}".format(prefix, idx)
        return imread(str(from_dir / "ori.png"))
    else:
        if isinstance(imagesize, int):
            imagesize = (imagesize, imagesize)
        h, w = imagesize[:2]

        rows = int((h - patchsize) / validsize + 1)
        cols = int((w - patchsize) / validsize + 1)
        base = (idx - 1) * rows * cols

        wholesize = (validsize * rows, validsize * cols, 3)

        img = np.zeros(wholesize, dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                from_dir = root / ckpt / "{}{:04d}".format(prefix, base + i * cols + j + 1)
                img_ = imread(str(from_dir / "ori.png"))
                
                img[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = img_

        return img

def improve_pseudo_labels(current_seg_mask, point_mask, pred_seg, pred_vet, pred_hor, h=0.5):
    """
    Improve the pseudo labels with current pseudo labels and model output.
    Args:
        current_seg_mask (numpy.ndarray[bool]): current segmentation mask.
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        pred_seg (numpy.ndarray[float]): predicted segmentation mask.
        pred_vet (numpy.ndarray[float]): predicted vertical distance map.
        pred_hor (numpy.ndarray[float]): predicted horizontal distance map.
    Returns:
        (tuple)
            new_seg (numpy.ndarray[bool]): new segmentation mask.
            new_cell (numpy.ndarray[int]): new cell colormap.
                Can be further used in distance_maps.get_distancemaps with use_full_mask set to True.
    """
    """initialization"""
    pred_seg = pred_seg > h
    out_dict = {}
    edges = get_voronoi_edges(point_mask, extra_out=out_dict)
    color_map = out_dict['Voronoi_cell']

    """improve segmentation label"""
    new_seg = np.zeros_like(pred_seg)

    labeled_pred_seg = label(pred_seg)
    labeled_curr_seg = np.where(current_seg_mask, color_map, 0)
    point_label = label(point_mask)
    MAX_POINT_IDX = int(point_label.max())
    for idx in range(1, MAX_POINT_IDX + 1):
        pred_cc_on_idx = labeled_pred_seg[point_label == idx][0]
        if pred_cc_on_idx != 0:
            new_seg = new_seg | (labeled_pred_seg == pred_cc_on_idx)
        else:
            curr_cc_on_idx = labeled_curr_seg[point_label == idx][0]
            if curr_cc_on_idx != 0:
                new_seg = new_seg | (labeled_curr_seg == curr_cc_on_idx)

    """=> new_seg"""

    """======================================================"""

    """improve cell colormap (for distance map label)"""

    """ - pre-process the output maps"""
    vet = shift_and_scale(pred_vet, 1, 0)
    hor = shift_and_scale(pred_hor, 1, 0)

    """ - generate distance map"""
    grad_vet = shift_and_scale(np.abs(sobel_v(vet)), 1, 0)
    grad_hor = shift_and_scale(np.abs(sobel_h(hor)), 1, 0)
    sig_diff = np.maximum(grad_vet, grad_hor)
    sig_diff = Cascade() \
                    .append(maximum_filter, size=3) \
                    .append(median_filter, size=3) \
                    (sig_diff)
    sig_diff[~pred_seg] = 0
    
    # show(sig_diff)
    # show(point_label)
    dilated_point_mask = binary_dilation(point_mask, disk(1))

    """ - run watershed on distance map with markers"""
    new_cell = watershed(new_seg, markers=point_label, mask=new_seg)

    # rgb_new_cell = (label2rgb(new_cell, bg_label=0) * 255).astype(np.uint8)
    # imsave('sig_diff.png', (sig_diff * 127).astype(np.uint8))
    # imsave('new_seg.png', new_seg.astype(np.uint8) * 255)
    # imsave('pred_seg.png', pred_seg.astype(np.uint8) * 255)
    # imsave('new_cell.png', rgb_new_cell)

    """ - re-fill the segmentation parts not in new_cell but in new segmentation"""

    """ ==== get (new segmentation mask) - (new_cell)"""
    not_in_new_cell = new_seg & (new_cell == 0)

    """ ==== separate nuclei with Voronoi edges and label them"""
    not_in_new_cell[edges == 255] = False
    not_in_new_cell = label(not_in_new_cell)

    """ ==== add back to new_cell"""
    offset = int(new_cell.max())
    not_in_new_cell += offset
    not_in_new_cell[not_in_new_cell == offset] = 0
    new_cell = new_cell + not_in_new_cell

    """ - filter out connected components which do not cover any point"""
    for cell in range(1, int(new_cell.max()) + 1):
        if np.count_nonzero((new_cell == cell) & point_mask) == 0:
            new_cell[new_cell == cell] = 0

    """=> new_cell"""

    # rgb_new_cell = (label2rgb(new_cell, bg_label=0) * 255).astype(np.uint8)
    # rgb_new_cell[dilated_point_mask] = [255, 255, 255]
    # imsave('new_cell1.png', rgb_new_cell)

    return new_seg, new_cell

def _test_instance_output():
    from skimage.io import imsave

    prefix = 'output/out_k_0.2'

    for IDX in range(1, 15):
        res = get_instance_output(True, IDX, k=0.2, use_patch_idx=False)
        img = get_original_image_from_file(IDX, use_patch_idx=False)
        np.save('{}_{}.npy'.format(prefix, IDX), res)

        gd = np.gradient(res)
        gd = (gd[0] ** 2 + gd[1] ** 2) ** (0.5)

        res[gd == 0] = 0

        rgbres = (label2rgb(res, bg_label=0) * 255).astype(np.uint8)
        
        res = res[..., None]
        res = np.concatenate((res, res, res), axis=2)
        img = np.where(res == 0, img, rgbres).astype(np.uint8)

        imsave('{}_{}.png'.format(prefix, IDX), img)

def _test_improve_pseudo_labels():
    from dataset_reader import CoNSeP
    from utils import get_valid_view

    IDX = 2
    SPLIT = 'test'

    dataset = CoNSeP(download=False)

    current_seg_mask = dataset.read_pseudo_labels(IDX, SPLIT)
    current_seg_mask = current_seg_mask > 0
    current_seg_mask = get_valid_view(current_seg_mask)

    point_mask = dataset.read_points(IDX, SPLIT)
    point_mask = get_valid_view(point_mask)

    seg, vet, hor = get_output_from_file(IDX)

    new_seg, new_cell = improve_pseudo_labels(current_seg_mask, point_mask, seg, vet, hor)

if __name__ == "__main__":
    # _test_instance_output()
    _test_improve_pseudo_labels()
