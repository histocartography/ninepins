import os
import torch
import numpy as np
from random import sample
from skimage.io import imread, imsave
from skimage.morphology import binary_dilation, disk, label
import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader
from histocartography.image.VorHoVerNet.distance_maps import get_distancemaps
from histocartography.image.VorHoVerNet.pseudo_label import gen_pseudo_label
from histocartography.image.VorHoVerNet.utils import get_point_from_instance, get_random_shifted_point_from_instance
from histocartography.image.VorHoVerNet.gaussian_filter import get_gaussian_mask


def padninvert(img, pad_width=((0, 40), (0, 40), (0, 0))):
    """
    Pad input DISTANCE MAP and fix the rule error of value due to padding operations.

    Args:
        img (numpy.ndarray): input DISTANCE MAP (horizontal or vertical maps, which values are from -1.0 to 1.0)
        pad_width (tuple): see pad_width in numpy.pad function
    Returns:
        (numpy.ndarray): result image
    """
    _channels = 3
    assert isinstance(pad_width, tuple), 'pad_width should be tuples in tuple'
    assert img.shape[-1] >= _channels, f'img channel must equal or greater than {_channels}, got {img.shape[-1]}'

    ori_h, ori_w = img.shape[:2]
    padded = np.pad(img, pad_width=pad_width, mode='reflect')
    padded[:, ori_w:, 1] = -padded[:, ori_w:, 1]
    padded[ori_h:, :, 2] = -padded[ori_h:, :, 2]
    return padded

def flip_image(img, flip, mode='normal', contain_both=False):
    """
    Return three other image copies through flip operation.

    Args:
        img (numpy.ndarray): target image to be flipped
        flip (int): 0 to 3, determine which kind of flip to apply
        mode (str): 'normal' or 'dist', 'dist' with value processing for distance map
    Returns:
        (numpy.ndarray): flipped image or distance map
    """
    _channels = 3
    assert mode in ('normal', 'dist'), 'mode must be either normal or dist'
    if mode == 'dist':
        assert img.shape[-1] >= _channels, f'img channel must equal or greater than {_channels}, got {img.shape[-1]}'

    res = img.copy()
    if flip == 1:
        if mode == 'dist':
            res[..., 2] = -res[..., 2]
        return np.flip(res, 0)
    elif flip == 2:
        if mode == 'dist':
            res[..., 1] = -res[..., 1]
        return np.flip(res, 1)
    elif flip == 3:
        if mode == 'dist':
            res[..., (1, 2)] = -res[..., (1, 2)]
        return np.flip(res, (0, 1))
    return res

def get_pseudo_masks(seg_mask, point_mask, inst, contain_both=False):
    """
    Generate pseudo labels and distamce map.

    Args:
        seg_mask (numpy.ndarray): clustered mask
        point_mask (numpy.ndarray): point mask
        inst (numpy.ndarray): instance map
        contain_both (bool): whether generate distance maps from exhausted labels
    Returns:
        pseudo_mask (and full_mask) (numpy.ndarray)
    """
    # get distance map
    h_map, v_map = get_distancemaps(point_mask, seg_mask)
    pseudo_mask = np.stack((seg_mask.astype(np.float32), h_map, v_map), axis=-1)
    if contain_both:
        seg_gt = inst > 0
        h_map, v_map = get_distancemaps(None, inst, use_full_mask=True)
        full_mask = np.stack((seg_gt.astype(np.float32), h_map, v_map), axis=-1)
        return pseudo_mask, full_mask
    return pseudo_mask

def gen_pseudo_masks(dataset='CoNSeP', root=None, split='train', ver=0, itr=0, contain_both=False):
    """
    Generate pseudo labels, including clusters, vertical and horizontal maps.

    Args:
        root (str): path to the top of the dataset
        split (str): data types, 'train' or 'test'
        contain_both (bool): if contain exhausted masks
    """
    
    dataset_class = getattr(dataset_reader, dataset)
    data_reader = dataset_class(root=root, download=False, ver=ver) if root is not None else dataset_class(download=False, ver=ver)
    IDX_LIMITS = data_reader.IDX_LIMITS

    root = data_reader.root.split("/")[0]
    
    for i in range(1, IDX_LIMITS[split] + 1):
        print(f'Generating {split}ing dataset (version {ver})... {i:02d}/{IDX_LIMITS[split]:02d}', end='\r')
        
        ori = data_reader.read_image(i, split)
        lab, type_ = data_reader.read_labels(i, split)

        if ver < 100:
            point_mask = data_reader.read_points(i, split)
        else:
            point_mask = get_random_shifted_point_from_instance(lab, 0.8, binary=True)
        
        # get dot masks
        dot_mask = binary_dilation(point_mask, selem=disk(2))
        dot_mask = np.expand_dims(dot_mask, axis=-1)

        if ver == 2:
            # use true label
            seg_mask_edge = lab > 0

        elif ver == 3:
            # use dilated point as label
            seg_mask_edge = binary_dilation(point_mask, selem=disk(4))

        else:
            # get cluster masks
            seg_mask, edges = gen_pseudo_label(ori, point_mask, return_edge=True)

            # version adjestments
            if ver:
                # do nothing with edges
                # ver == 5 to test if gaussian mask is useful
                seg_mask_edge = seg_mask
            else: # ver == 1 or ver >= 100
                # set edges as background
                seg_mask_edge = seg_mask & (edges == 0)
        
        seg_mask_edge = label(seg_mask_edge, connectivity=1)
        for seg_idx in np.unique(seg_mask_edge):
            if seg_idx == 0: continue
            if np.count_nonzero((seg_mask_edge == seg_idx) & point_mask) == 0:
                seg_mask_edge[seg_mask_edge == seg_idx] = 0
        seg_mask_edge = seg_mask_edge > 0

        # generate distance maps
        # # mirror padding (1000x1000 to 1230x1230, 95 at left and top, 135 at right and bottom)
        # lab_, seg_mask_edge_, point_mask_ = [np.pad(tar, ((95, 135), (95, 135)), mode='reflect') for tar in [lab_, seg_mask_edge_, point_mask_]]

        # get gaussian masks
        gaussian_mask = get_gaussian_mask(point_mask, sigma=5, mag=5.0, constant=1.0)
        gaussian_mask = np.expand_dims(gaussian_mask, axis=-1)

        if contain_both:
            pseudo_mask, full_mask = get_pseudo_masks(seg_mask_edge, point_mask, lab, contain_both=True)
        else:
            pseudo_mask = get_pseudo_masks(seg_mask_edge, point_mask, lab, contain_both=False)

        # save npy file (and png file for visualization)
        path_pseudo = f'{root}/{split.capitalize()}/version_{ver:02d}/PseudoLabels_{itr:02d}'
        pseudo_mask = np.concatenate((pseudo_mask, dot_mask, gaussian_mask), axis=-1)
        os.makedirs(path_pseudo, exist_ok=True)
        rgb_mask = np.stack((seg_mask_edge*255,)*3, axis=-1).astype(np.uint8)
        rgb_mask[dot_mask[..., 0] == 1] = (255, 0, 0)

        # imsave(f'{path_pseudo}/{split}_{i}.png', seg_mask_edge.astype(np.uint8) * 255)
        imsave(f'{path_pseudo}/{split}_{i}.png', rgb_mask)
        np.save(f'{path_pseudo}/{split}_{i}.npy', pseudo_mask)
        if contain_both:
            seg_gt = lab > 0
            path_full = f'{root}/{split.capitalize()}/version_{ver:02d}/FullLabels'
            os.makedirs(path_full, exist_ok=True)
            rgb_mask = np.stack((seg_gt*255,)*3, axis=-1).astype(np.uint8)
            rgb_mask[dot_mask[..., 0] == 1] = (255, 0, 0)
            imsave(f'{path_full}/{split}_{i}.png', rgb_mask)
            np.save(f'{path_full}/{split}_{i}.npy', full_mask)
    print('')

def data_reader(dataset='CoNSeP', root=None, split='train', channel_first=True, ver=0, itr=0, doflip=False, contain_both=False, part=None, norm=True):
    """
    Return images and labels according to the type from split

    Args:
        root (str): path to the top of the dataset
        split (str): data types, 'train' or 'test'
        channel_first (bool): if channel is first or not
        itr (int): which iteration of dataset
        doflip (bool): whether to flip dataset or not
        contain_both (bool): if contain exhausted masks
        part (tuple, list): selected indice of data
    
    Returns:
        (tuple):
            images (numpy.ndarray)
            labels (numpy.ndarray)
    """
    dataset_class = getattr(dataset_reader, dataset)
    data_reader = dataset_class(root=root, download=False, ver=ver) if root is not None else dataset_class(download=False, ver=ver)
    IDX_LIMITS = data_reader.IDX_LIMITS
    # select indice from dataset if customization is needed
    indice = range(1, IDX_LIMITS[split] + 1) if part is None else part
    images = []
    labels = []
    # fulllabels = []
    # pseudolabels = []
    for i, idx in enumerate(indice):
        print(f'Loading {split} dataset... {i+1:02d}/{len(indice):02d}', end='\r')
        # load original image
        image = data_reader.read_image(idx, split) / 255
        # load pseudo labels
        # label_path = data_reader.get_path(idx, split, 'label')
        # pseudolabel_path = label_path.replace('Labels', f'PseudoLabels_{itr:02d}')
        pseudolabels = np.load(data_reader.get_path(idx, split, 'pseudo'))
        ori_h, ori_w = pseudolabels.shape[:2]
        # load full labels (optional)
        if contain_both:
            # fulllabel_path = label_path.replace('Labels', 'FullLabels')
            fulllabels = np.load(data_reader.get_path(idx, split, 'full'))

        flip_idx = range(4) if doflip else (0, )
        for flip in flip_idx:
            # flip and pad
            image_ = flip_image(image, flip, mode='normal')
            image_ = np.pad(image_, ((95, 135), (95, 135), (0, 0)), mode='reflect')
            images.append(image_)

            pseudolabels_ = flip_image(pseudolabels, flip, mode='dist')
            pseudolabels_ = padninvert(pseudolabels_, pad_width=((0, 40), (0, 40), (0, 0)))
            if contain_both:
                fulllabels_ = flip_image(fulllabels, flip, mode='dist')
                fulllabels_ = padninvert(fulllabels_, pad_width=((0, 40), (0, 40), (0, 0)))
                labels.append(np.concatenate((pseudolabels_, fulllabels_), axis=-1))
                # labels.append(np.concatenate((fulllabels_, np.expand_dims(pseudolabels_[..., -1], axis=-1), fulllabels_), axis=-1))
            else:
                labels.append(pseudolabels_)
    print('')
    return images, labels

def dataset_numpy_to_tensor(dataset, batch_size=8):
    imgs, gts = [], []
    for i in range(batch_size):
        img, gt = dataset[i]
        imgs.append(img)
        gts.append(gt)
    new_imgs = np.stack(imgs, axis=0)
    new_gts = np.stack(gts, axis=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.from_numpy(new_imgs).to(device), torch.from_numpy(new_gts).to(device)

class MixedDataset(torch.utils.data.Dataset):
    """
    Mixed dataset of pseudo and true labels.

    Args:
        dataset (torch.utils.data.Dataset): cropped target dataset, better be shuffled
        rate (float): rate of PSEUDO labels
    """
    def __init__(self, dataset, rate=1.0, channel_first=True):
        self.dataset = dataset
        self.rate = rate
        self.channel_first = channel_first

        self.mixed_images = []
        self.mixed_labels = []

        # check input
        _channels = 7
        self.t_ch = 0 if self.channel_first else -1
        assert self.dataset[0][1].shape[self.t_ch] == 7, f"label channel should be greater than {_channels}, got {self.dataset[0][1].shape[self.t_ch]}"
        assert 0 <= self.rate <= 1.0, f"rate must be between 0 and 1.0, got {self.rate}"

        self.mix()

    def mix(self):
        len_ = len(self.dataset)
        true_num = int(len_ * (1 - self.rate))
        true_idx = sample(range(len_), true_num)

        # if self.rate < 1.0:
        print(f"Mixing dataset ({self.rate * 100}% pseudo labels)...")
        for i, (images, labels) in enumerate(self.dataset):
            if i in true_idx:
                # split into 3, 1, 3 channels, mix and restruct
                pseudo, point, full = np.split(labels, (3, 4),  axis=self.t_ch)
                new_labels = np.concatenate((full, point, pseudo), axis=self.t_ch)
                self.mixed_labels.append(new_labels) 
            else:
                self.mixed_labels.append(labels)
            self.mixed_images.append(images)

    # def shape(self):
    #     pass

    def __getitem__(self, index):
        return self.mixed_images[index], self.mixed_labels[index]
    
    def __len__(self):
        return len(self.mixed_images)


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Apply augmentation on the input dataset according to transform and target transform.

    Args:
        dataset (torch.utils.data.Dataset): the dataset to be augmented
        transform, target_transform (torchvision.transforms.Compose): transform methods to be applied.
            if target_transform is None, it will use whatever in transform
    """
    def __init__(self, dataset, transform=None, target_transform=None, channel_first=True):
        self.dataset = dataset
        self.channel_first = channel_first
        self.transform = transform
        self.target_transform = target_transform
        self.check_trans()

    def check_trans(self):    
        if self.transform is not None:
            print('transform on image')
        if self.target_transform is not None:
            print('transform on labels')
        if self.transform is None and self.target_transform is None:
            print('empty transforms')
        
    def __getitem__(self, index):
        img, gts = self.dataset[index]
        img = (img * 255).astype(np.uint8)
        gts = (gts * 255).astype(np.uint8)

        # print('a', img.shape, gts.shape, img.max(), img.min())
        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))
            gts = np.transpose(gts, (1, 2, 0))
        
        # print('b', img.shape, gts.shape)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            if gts.shape[-1] > 3:
                gt02, gt35 = np.split(gts, 2, axis=-1)
                gt02 = self.target_transform(gt02)
                gt35 = self.target_transform(gt35)
                gts = np.concatenate((gt02, gt35), axis=-1)
            else:
                gts = self.target_transform(gts)
        img = np.array(img).astype(np.float32) / 255
        gts = np.array(gts).astype(np.float32) / 255
        # print('c', img.shape, gts.shape)
        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
            gts = np.transpose(gts, (2, 0, 1))
        # print('d', img.shape, gts.shape, img.max(), img.min())
        return img, gts

    def __len__(self):
        return len(self.dataset)

class CoNSeP_cropped(torch.utils.data.Dataset):
    """
    Read dataset and crop images and labels to certain size.
    """
    def __init__(self, images, labels, patchsize=270, validsize=80, channel_first=True):
        self.images = images
        self.labels = labels
        self.patchsize = patchsize
        self.validsize = validsize
        
        # crop patches
        self.crop_patches()
        if channel_first:
            self.transpose((0, 3, 1, 2))

    def crop_patches(self):
        self.crop_images = []
        self.crop_labels = []
        gap = (self.patchsize - self.validsize) // 2
        imgs_append = self.crop_images.append
        lbls_append = self.crop_labels.append
        for i, (img, lbls) in enumerate(zip(self.images, self.labels)):
            print('Cropping dataset... {:02d}/{:02d}'.format(i + 1, len(self.images)), end='\r')
            height, width = img.shape[:2]
            # img = np.pad(img, ((gap, gap), (gap, gap), (0, 0)), mode='reflect')
            # height, width = img.shape[:2]
            for h in range(0, height, self.validsize):
                for w in range(0, width, self.validsize):
                    if h + self.patchsize <= height and w + self.patchsize <= width:
                        imgs_append(img[h:h+self.patchsize, w:w+self.patchsize, :3])
                        lbls_append(lbls[h:h+self.validsize, w:w+self.validsize, :])
        print('')
        self.crop_images = np.array(self.crop_images, dtype=np.float32)
        self.crop_labels = np.array(self.crop_labels, dtype=np.float32)
            
    def transpose(self, dims):
        self.crop_images = np.transpose(self.crop_images, dims)
        self.crop_labels = np.transpose(self.crop_labels, dims)
        
    def patches_per_image(self):
        return len(self.crop_images) // len(self.images)
        
    def __getitem__(self, index):
        return self.crop_images[index], self.crop_labels[index]
    
    def __len__(self):
        return len(self.crop_images)

if __name__ == '__main__':
    # for i in range(3, 4):
    i = 1
    gen_pseudo_masks(dataset='CoNSeP', split='train', ver=i, itr=0, contain_both=True)
    gen_pseudo_masks(dataset='CoNSeP', split='test', ver=i, itr=0, contain_both=True)
