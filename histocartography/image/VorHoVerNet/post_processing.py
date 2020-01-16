from pathlib import Path
import numpy as np
import torch
from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import label2rgb
from skimage.filters import gaussian, sobel_h, sobel_v
from skimage.io import imread, imsave
from skimage.morphology import *
from histocartography.image.VorHoVerNet.dataset import get_pseudo_masks
from histocartography.image.VorHoVerNet.utils import *
from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges

DEFAULT_H = 0.5
DEFAULT_K = 1.7

def masked_scale(image, seg, th=0.5, vmax=1, vmin=-1):
    image[seg <= th] = 0
    image = scale(image, vmax, vmin)
    return image

DEFAULT_TRANSFORM = lambda im, seg: masked_scale(im, seg)

def _get_instance_output(seg, hor, vet, h=DEFAULT_H, k=DEFAULT_K):
    """
    Combine model output values from three branches into instance segmentation.
    Args:
        seg (numpy.ndarray[float]): segmentation output.
        hor (numpy.ndarray[float]): horizontal distance map output.
        vet (numpy.ndarray[float]): vertical distance map output.
    Returns:
        instance_map (numpy.ndarray[int]): instance map
    """
    """pre-process the output maps"""
    th_seg = seg > h
    th_seg = Cascade() \
                .append(remove_small_objects, min_size=5) \
                (th_seg)
                # .append(binary_opening, disk(3)) \

    # show(th_seg)

    """combine the output maps"""

    """ - generate distance map"""
    grad_hor = np.exp(shift_and_scale(np.abs(np.gradient(hor)[1]), 1, 0) * 5)
    grad_vet = np.exp(shift_and_scale(np.abs(np.gradient(vet)[0]), 1, 0) * 5)
    sig_diff = (grad_hor ** 2 + grad_vet ** 2) ** 0.5
    sig_diff = Cascade() \
                    .append(median_filter, size=5) \
                    (sig_diff)
    # show(sig_diff * (seg > h))
    sig_diff[~th_seg] = 0


    # grad_hor = shift_and_scale(np.abs(sobel_h(hor)), 1, 0)
    # grad_vet = shift_and_scale(np.abs(sobel_v(vet)), 1, 0)
    # sig_diff = np.maximum(grad_hor, grad_vet)
    # sig_diff = Cascade() \
    #                 .append(maximum_filter, size=3) \
    #                 .append(median_filter, size=3) \
    #                 (sig_diff)
    #                 # .append(gaussian) \
    # sig_diff[~th_seg] = 0

    # show(hor)
    # show(vet)
    # show(sig_diff)

    """ - generate markers"""
    if k == 'adapt':
        k = sig_diff.mean()
    markers = th_seg & (sig_diff <= k)
    # intermediate_prefix='markers/m'
    markers = Cascade() \
                    .append(binary_opening, disk(1)) \
                    .append(remove_small_objects, min_size=10) \
                    (markers)

    # markers = Cascade() \
    #                 .append(binary_dilation, disk(3)) \
    #                 .append(binary_fill_holes) \
    #                 .append(binary_erosion, disk(3)) \
    #                 .append(remove_small_objects, min_size=5) \
    #                 (markers)
    markers = label(markers)
    
    # show(th_seg)
    # show(markers)

    """ - run watershed on distance map with markers"""
    res = watershed(sig_diff, markers=markers, mask=th_seg)

    """ - re-fill regions in thresholded nuclei map which do not have markers"""
    not_in_watershed = th_seg & (res == 0)
    lbl_th_seg = label(not_in_watershed)
    offset = int(res.max())
    lbl_th_seg += offset
    lbl_th_seg[lbl_th_seg == offset] = 0
    
    # res = np.where((res == 0) & (lbl_th_seg != 0), lbl_th_seg, res)
    res += lbl_th_seg

    """debug"""
    # imsave("output/sig_diff.png", (shift_and_scale(sig_diff, 1, 0) * 255).astype(np.uint8))
    # imsave("output/markers.png", (label2rgb(markers, bg_label=0) * 255).astype(np.uint8))

    return res

def get_instance_output(from_file, *args, h=DEFAULT_H, k=DEFAULT_K, **kwargs):
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
        seg, hor, vet = get_output_from_file(*args,
                            transform=lambda im, seg: masked_scale(im, seg, th=h), **kwargs)
    else:
        seg, hor, vet = get_output_from_model(*args,
                            transform=lambda im, seg: masked_scale(im, seg, th=h), **kwargs)

    return _get_instance_output(seg, hor, vet, h=h, k=k)

def get_output_from_file(idx,
                        transform=None, use_patch_idx=False,
                        root='./inference', ckpt='model_009_ckpt_epoch_18',
                        split='test', prefix='patch',
                        patchsize=270, validsize=80, inputsize=1230, imagesize=1000):
    """
    Read output from file.
    Args:
        idx (int): index of image in dataset.
        transform (callable{
            [numpy.ndarray(float), numpy.ndarray(float)]
            => numpy.ndarray(float)
        } / None):
            the transformation function to apply on distance maps after reading.
            Nothing would be done if set to None.
        use_patch_idx (bool): treat the index as patch index instead.
        root (str): root directory of the output files.
        ckpt (str): name of the checkpoint used to generate the output.
        prefix (str): prefix of the file names.
        split (str): split of dataset to use. 
        patchsize (int): size of the input patch of model.
        validsize (int): size of the output patch of model.
        inputsize (int): size of the padded whole image.
        imagesize (int): size of the whole image in dataset.
    Returns:
        (tuple):
            seg (numpy.ndarray[float]): segmentation output.
            hor (numpy.ndarray[float]): horizontal distance map output.
            vet (numpy.ndarray[float]): vertical distance map output.
    """
    if isinstance(root, str):
        root = Path(root)
    assert isinstance(root, Path), "{} is not a Path object or string".format(root)

    if use_patch_idx:
        from_dir = root / ckpt / split / "{}{:04d}".format(prefix, idx)
        seg = np.load(str(from_dir / "seg.npy"))
        hor = np.load(str(from_dir / "dist1.npy"))
        vet = np.load(str(from_dir / "dist2.npy"))
        if transform is not None:
            hor = transform(hor, seg)
            vet = transform(vet, seg)
        return seg, hor, vet
    else:
        if isinstance(inputsize, int):
            inputsize = (inputsize, inputsize)
        if isinstance(imagesize, int):
            imagesize = (imagesize, imagesize)
        h, w = inputsize[:2]

        rows = int((h - patchsize) / validsize + 1)
        cols = int((w - patchsize) / validsize + 1)
        base = (idx - 1) * rows * cols

        wholesize = (validsize * rows, validsize * cols)

        seg = np.zeros(wholesize, dtype=float)
        hor = np.zeros(wholesize, dtype=float)
        vet = np.zeros(wholesize, dtype=float)

        for i in range(rows):
            for j in range(cols):
                index = base + i * cols + j + 1
                from_dir = root / ckpt / split / "{}{:04d}".format(prefix, index)
                seg_ = np.load(str(from_dir / "seg.npy"))
                hor_ = np.load(str(from_dir / "dist1.npy"))
                vet_ = np.load(str(from_dir / "dist2.npy"))

                if transform is not None:
                    hor_ = transform(hor_, seg_)
                    vet_ = transform(vet_, seg_)
                
                seg[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = seg_
                hor[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = hor_
                vet[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = vet_

        ih, iw = imagesize

        return seg[:ih, :iw, ...], hor[:ih, :iw, ...], vet[:ih, :iw, ...]

def get_output_from_model(img, model, transform=None):
    """
    Get output by feeding an image to a model.
    Args:
        img (torch.Tensor): the input image.
        model (torch.nn.Module): 
    Returns:
        (tuple):
            seg (numpy.ndarray[float]): segmentation output.
            hor (numpy.ndarray[float]): horizontal distance map output.
            vet (numpy.ndarray[float]): vertical distance map output.
    """
    with torch.no_grad():
        pred = model(img)
    pred = pred.squeeze(0).detach().cpu().numpy()

    seg = pred[..., 0]
    hor = pred[..., 1]
    vet = pred[..., 2]

    if transform is not None:
        hor = transform(hor, seg)
        vet = transform(vet, seg)
    return seg, hor, vet

def get_original_image_from_file(idx,
                                use_patch_idx=False, root='./inference',
                                ckpt='model_009_ckpt_epoch_18', prefix='patch',
                                split='test',
                                patchsize=270, validsize=80,
                                inputsize=1230, imagesize=1000):
    """
    Read original image from file.
    Args:
        idx (int): index of image in dataset.
        use_patch_idx (bool): treat the index as patch index instead.
        root (str): root directory of the output files.
        ckpt (str): name of the checkpoint used to generate the output.
        prefix (str): prefix of the file names.
        split (str): split of dataset to use.
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
        from_dir = root / ckpt / split / "{}{:04d}".format(prefix, idx)
        return imread(str(from_dir / "ori.png"))
    else:
        if isinstance(inputsize, int):
            inputsize = (inputsize, inputsize)
        if isinstance(imagesize, int):
            imagesize = (imagesize, imagesize)
        h, w = inputsize[:2]

        rows = int((h - patchsize) / validsize + 1)
        cols = int((w - patchsize) / validsize + 1)
        base = (idx - 1) * rows * cols

        wholesize = (validsize * rows, validsize * cols, 3)

        img = np.zeros(wholesize, dtype=np.uint8)

        for i in range(rows):
            for j in range(cols):
                from_dir = root / ckpt / split / "{}{:04d}".format(prefix, base + i * cols + j + 1)
                img_ = imread(str(from_dir / "ori.png"))
                
                img[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = img_

        ih, iw = imagesize

        return img[:ih, :iw, ...]

def improve_pseudo_labels(current_seg_mask, point_mask, pred_seg, pred_hor, pred_vet, h=DEFAULT_H, method='Voronoi'):
    """
    Improve the pseudo labels with current pseudo labels and model output.
    Args:
        current_seg_mask (numpy.ndarray[bool]): current segmentation mask.
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        pred_seg (numpy.ndarray[float]): predicted segmentation mask.
        pred_hor (numpy.ndarray[float]): predicted horizontal distance map.
        pred_vet (numpy.ndarray[float]): predicted vertical distance map.
        h (float): threshold for segmentation mask.
        method (str): method to use. [Voronoi, instance]
    Returns:
        (tuple)
            new_seg (numpy.ndarray[bool]): new segmentation mask.
            new_cell (numpy.ndarray[int]): new cell colormap.
                Can be further used in distance_maps.get_distancemaps with use_full_mask set to True.
    """
    """initialization"""
    pred_seg = pred_seg > h
    pred_seg = Cascade() \
                .append(remove_small_objects, min_size=5) \
                (pred_seg)
    hor = pred_hor
    vet = pred_vet
    out_dict = {}
    edges = get_voronoi_edges(point_mask, extra_out=out_dict, l2_norm=True)
    color_map = out_dict['Voronoi_cell']
    dilated_point_mask = binary_dilation(point_mask, disk(1))
    # dilated_point_label = dilation(point_label, disk(2))

    """improve segmentation label"""
    if method == 'Voronoi':
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

        new_cell = color_map * new_seg

    elif method == 'instance':
        new_seg = np.zeros_like(pred_seg).astype(int)
        # new_seg_curr = np.zeros_like(pred_seg).astype(int)
        # new_seg_pred = new_seg_curr.copy()
        labeled_pred_seg = _get_instance_output(pred_seg, pred_hor, pred_vet)
        labeled_curr_seg = np.where(current_seg_mask, color_map, 0)
        point_label = label(point_mask)
        MAX_POINT_IDX = int(point_label.max())
        for idx in range(1, MAX_POINT_IDX + 1):
            pred_cc_on_idx = labeled_pred_seg[point_label == idx][0]
            if pred_cc_on_idx != 0:
                # new_seg = new_seg | (labeled_pred_seg == pred_cc_on_idx)
                new_seg[labeled_pred_seg == pred_cc_on_idx] = idx
            else:
                curr_cc_on_idx = labeled_curr_seg[point_label == idx][0]
                # new_seg = new_seg | (labeled_curr_seg == curr_cc_on_idx)
                assert curr_cc_on_idx != 0, "current segmentation mask should cover all points."
                connected_components = label(labeled_curr_seg == curr_cc_on_idx)
                cc_on_idx = connected_components[point_label == idx][0]
                new_seg[(connected_components == cc_on_idx) & (new_seg == 0)] = idx
                new_seg = new_seg | (connected_components == cc_on_idx)

        base = new_seg.max()
        for idx in np.unique(new_seg):
            seg = new_seg == idx
            if np.count_nonzero(seg & point_mask) > 1:
                seg_cell = watershed(seg, markers=point_label, mask=seg)
                step = seg_cell.max()
                seg_cell[seg_cell > 0] += base
                base += step
                new_seg[seg] = seg_cell[seg]

        new_seg = label(new_seg)
        new_cell = new_seg
        new_seg = new_cell > 0
    else:
        raise ValueError('Invalid method')

    return new_seg, new_cell

    """!!!!below is discarded for now!!!!"""

    # show(label2rgb(new_seg_pred, bg_label=0))

    # offset = new_seg_pred.max()
    # new_seg_curr += offset
    # new_seg_curr[new_seg_curr == offset] = 0
    # new_seg = new_seg_pred + new_seg_curr

    # return new_seg

    """=> new_seg"""

    """======================================================"""

    """improve cell colormap (for distance map label)"""

    """ - generate distance map"""
    # grad_hor = shift_and_scale(np.abs(sobel_h(hor)), 1, 0)
    # grad_vet = shift_and_scale(np.abs(sobel_v(vet)), 1, 0)
    # sig_diff = np.maximum(grad_hor, grad_vet)
    # sig_diff = Cascade() \
    #                 .append(median_filter, size=3) \
    #                 .append(gaussian) \
    #                 (sig_diff)
    #                 # .append(maximum_filter, size=3) \
    #                 # .append(median_filter, size=3) \
    # sig_diff[~new_seg] = 0

    grad_hor = np.exp(shift_and_scale(np.abs(np.gradient(hor)[1]), 1, 0) * 5)
    grad_vet = np.exp(shift_and_scale(np.abs(np.gradient(vet)[0]), 1, 0) * 5)
    # show(grad_hor, grad_vet)
    # sig_diff = np.maximum(grad_hor, grad_vet)
    sig_diff = (grad_hor ** 2 + grad_vet ** 2) ** 0.5
    sig_diff = Cascade() \
                    .append(median_filter, size=5) \
                    (sig_diff)
                    # .append(gaussian) \
                    # .append(maximum_filter, size=3) \
    sig_diff[~new_seg] = 0

    # sig_diff_ = sig_diff.copy()
    # sig_diff_[point_mask] = 5
    # show(sig_diff_)
    # show(point_label)

    """ - run watershed on distance map with markers"""
    # show(sig_diff)
    # b_sig_diff = draw_boundaries(sig_diff.copy(), new_seg, color=[0, 1, 0])
    # b_sig_diff[dilated_point_mask] = [1, 0, 0]
    # show(b_sig_diff)
    
    # new_cell = watershed(sig_diff, markers=point_label, mask=new_seg)
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
            # new_cell[new_cell > cell] -= 1
        # else:
        #     if np.count_nonzero(new_cell == cell) < 30:
        #         point = point_mask & (new_cell == cell)
        #         new_cell[binary_dilation(point, disk(5)) & new_seg] = cell

    # """force every point to has minimum size of disk(3)"""
    new_cell = label(new_cell)

    """=> new_cell"""

    return new_seg, new_cell

def gen_next_iteration_labels(curr_iter, ckpt, split, inputsize=1230, patchsize=270, validsize=80, imagesize=1000):
    import os
    from skimage.io import imsave
    from torch.utils.data import DataLoader
    
    from dataset_reader import CoNSeP
    from dataset import CoNSeP_cropped, data_reader
    from model.vorhover_net import Net

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load datasets
    dataset = CoNSeP(download=False, itr=curr_iter)
    torch_dataset = CoNSeP_cropped(*data_reader(root='CoNSeP/', split=split, itr=curr_iter, doflip=False, contain_both=False, part=None))
    data_loader = DataLoader(torch_dataset, batch_size=1, shuffle=False)
    pseudo_path = '/'.join(dataset.get_path(1, split, 'pseudo', itr=curr_iter+1).split('/')[:-1])
    os.makedirs(pseudo_path, exist_ok=True)

    # load model
    checkpoint = torch.load(ckpt, map_location=device)
    print('model_name: {}'.format(ckpt))
    model = Net()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # compute dimensions
    rows = int((inputsize - patchsize) / validsize + 1)
    cols = int((inputsize - patchsize) / validsize + 1)
    step = rows * cols
    wholesize = (validsize * rows, validsize * cols)
    total_images = dataset.IDX_LIMITS[split]
    total_patches = total_images*step

    # run
    current_seg_mask = np.zeros(wholesize, dtype=bool)
    pred_seg = np.zeros(wholesize, dtype=np.float32)
    pred_hor = np.zeros(wholesize, dtype=np.float32)
    pred_vet = np.zeros(wholesize, dtype=np.float32)

    for idx, (img, gt) in enumerate(data_loader):
        gt = gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()[..., 0]
        i = (idx % step) // cols
        j = (idx % step) % cols
        current_seg_mask[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = gt

        seg, hor, vet = get_output_from_model(img.to(device), model, transform=DEFAULT_TRANSFORM)

        pred_seg[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = seg
        pred_hor[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = hor
        pred_vet[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = vet

        image_idx = (idx // step) + 1
        print(f'image: {image_idx}/{total_images}, patch: {idx+1}/{total_patches}', end='\r')
        if (idx + 1) % step == 0:
            current_seg_mask_ = current_seg_mask[:imagesize, :imagesize]
            pred_seg_ = pred_seg[:imagesize, :imagesize]
            pred_hor_ = pred_hor[:imagesize, :imagesize]
            pred_vet_ = pred_vet[:imagesize, :imagesize]
            point_mask = dataset.read_points(image_idx, split)
            new_seg, new_cell = improve_pseudo_labels(current_seg_mask_, point_mask, pred_seg_, pred_hor_, pred_vet_)
            assert new_seg.shape == (1000, 1000) and new_cell.shape == (1000, 1000), "I fucked up."
            # get voronoi edges
            edges = np.zeros_like(new_seg).astype(bool)
            edges[:-1, :][new_cell[1:, :] != new_cell[:-1, :]] = True
            edges[:, :-1][new_cell[:, 1:] != new_cell[:, :-1]] = True
            new_seg = new_seg & ~edges
            pseudo_masks = get_pseudo_masks(new_seg, point_mask, new_cell)
            np.save(dataset.get_path(image_idx, split, 'pseudo', itr=curr_iter+1), pseudo_masks)
            imsave(dataset.get_path(image_idx, split, 'pseudo', itr=curr_iter+1).replace('.npy', '.png'), (pseudo_masks[..., 0] * 255).astype(np.uint8))

def _test_instance_output():
    from skimage.io import imsave

    os.makedirs('output', exist_ok=True)

    prefix = 'output/temp_test'

    use_patch_idx = False
    # use_patch_idx = True

    for IDX in range(1, 15):
    # for IDX in range(1, 2367):
    # observe_idx = 2
    # for IDX in range(observe_idx, observe_idx + 1):
        # res = get_instance_output(True, IDX, k=0.1, use_patch_idx=use_patch_idx)
        res = get_instance_output(True, IDX, use_patch_idx=use_patch_idx)
        img = get_original_image_from_file(IDX, use_patch_idx=use_patch_idx)
        np.save('{}_{}.npy'.format(prefix, IDX), res)

        img = draw_label_boundaries(img, res)

        imsave('{}_{}.png'.format(prefix, IDX), img)

        # show(img)

def _test_improve_pseudo_labels():
    from dataset_reader import CoNSeP
    # from utils import get_valid_view
    from skimage.io import imsave
    from utils import draw_label_boundaries

    # IDX = 2
    SPLIT = 'test'
    dataset = CoNSeP(download=False)

    for IDX in range(1, 2):
        ori = get_original_image_from_file(IDX)

        current_seg_mask = dataset.read_pseudo_labels(IDX, SPLIT)
        # current_seg_mask = get_valid_view(current_seg_mask)
        current_seg_mask = current_seg_mask > 0
        # current_seg_mask = get_valid_view(current_seg_mask)

        point_mask = dataset.read_points(IDX, SPLIT)
        # point_mask = get_valid_view(point_mask)

        seg, hor, vet = get_output_from_file(IDX, transform=DEFAULT_TRANSFORM)

        new_seg, new_cell = improve_pseudo_labels(current_seg_mask, point_mask, seg, hor, vet)

        # new_cell = improve_pseudo_labels(current_seg_mask, point_mask, seg, hor, vet)

        # show(current_seg_mask)
        # show(seg > 0.5)
        # show(new_seg)

        dilated_point_mask = binary_dilation(point_mask, disk(1))
        # rgb_new_cell = (label2rgb(new_cell, bg_label=0) * 255).astype(np.uint8)
        rgb_new_cell = draw_label_boundaries(ori, new_cell)
        rgb_new_cell[dilated_point_mask] = [255, 255, 255]
        # imsave('new_cell1.png', rgb_new_cell)

        show(rgb_new_cell)
        # show(draw_label_boundaries(ori, new_cell))

if __name__ == "__main__":
    _test_instance_output()
    # _test_improve_pseudo_labels()
