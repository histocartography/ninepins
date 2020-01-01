from pathlib import Path
import matplotlib.pyplot as plt  # for debugging
import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import label2rgb
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.morphology import *
from inference import shift_and_scale
from utils import Cascade


def show(img): # for debugging
    plt.imshow(img)
    plt.show()

def get_instance_output(from_file, *args, h=0.5, k=0.1, **kwargs):
    # read files or inference to get individual output
    if from_file:
        seg, vet, hor = get_instance_output_from_file(*args, **kwargs)
    else:
        seg, vet, hor = get_instance_output_from_model(*args, **kwargs)

    # pre-process the output maps
    vet = shift_and_scale(vet, 1, 0)
    hor = shift_and_scale(hor, 1, 0)
    th_seg = seg > h

    # combine the output maps

    #   generate distance map
    grad_vet = shift_and_scale(np.abs(sobel_v(vet)), 1, 0)
    grad_hor = shift_and_scale(np.abs(sobel_h(hor)), 1, 0)
    sig_diff = np.maximum(grad_vet, grad_hor)
    sig_diff = Cascade() \
                    .append(maximum_filter, size=3) \
                    .append(median_filter, size=3) \
                    (sig_diff)
                    # .append(gaussian) \
    sig_diff[~th_seg] = 0
    
    # show(sig_diff)

    #   generate markers
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

    #   run watershed on distance map with markers
    res = watershed(sig_diff, markers=markers, mask=th_seg)

    #   re-fill regions in thresholded nuclei map which do not have markers
    lbl_th_seg = label(th_seg)
    offset = int(markers.max())
    lbl_th_seg += offset
    lbl_th_seg[lbl_th_seg == offset] = 0
    
    res = np.where((res == 0) & (lbl_th_seg != 0), lbl_th_seg, res)

    return res

def get_instance_output_from_file(idx, use_patch_idx=False, root='./inference', ckpt='model_003_ckpt_epoch_43', prefix='patch', patchsize=270, validsize=80, imagesize=1000):
    """
    idx: index of image in dataset
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
        base = idx * rows * cols

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

def get_instance_output_from_model(img, model):
    model.eval()
    with torch.no_grad():
        pred = model(img)
    pred = pred.squeeze(0).detach().cpu().numpy()
    return pred[..., 0], \
            pred[..., 1], \
            pred[..., 2]

def get_original_image_from_file(idx, use_patch_idx=False, root='./inference', ckpt='model_003_ckpt_epoch_43', prefix='patch', patchsize=270, validsize=80, imagesize=1000):
    """
    idx: index of image in dataset
    """
    from skimage.io import imread

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
        base = idx * rows * cols

        wholesize = (validsize * rows, validsize * cols, 3)

        img = np.zeros(wholesize, dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                from_dir = root / ckpt / "{}{:04d}".format(prefix, base + i * cols + j + 1)
                img_ = imread(str(from_dir / "ori.png"))
                
                img[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = img_

        return img

if __name__ == "__main__":
    from skimage.io import imsave

    prefix = 'output/out_k_0.2'

    for IDX in range(14):
        res = get_instance_output(True, IDX, k=0.2, use_patch_idx=False)
        img = get_original_image_from_file(IDX, use_patch_idx=False)
        np.save('{}_{}.npy'.format(prefix, IDX), res)

        gd = np.gradient(res)
        gd = (gd[0] ** 2 + gd[1] ** 2) ** (0.5)

        res[gd == 0] = 0

        rgbres = (label2rgb(res) * 255).astype(np.uint8)
        
        res = res[..., None]
        res = np.concatenate((res, res, res), axis=2)
        img = np.where(res == 0, img, rgbres).astype(np.uint8)

        imsave('{}_{}.png'.format(prefix, IDX), img)
