import os
import numpy as np
from skimage.io import imread, imsave
from torch.utils.data import Dataset
from dataset_reader import *
from distance_maps import get_distancemaps
from pseudo_label import gen_pseudo_label
from utils import get_point_from_instance


def flip_image(img, flip):
    """
    Generate three other images through flip.
    """
    if flip == 1:
        return np.flip(img, 0)
    elif flip == 2:
        return np.flip(img, 1)
    elif flip == 3:
        return np.flip(img, (0, 1))
    return img.copy()

def gen_pseudo_masks(root='./CoNSeP/', split='train', contain_both=False):
    """
    Generate pseudo labels, including clusters, vertical and horizontal maps.

    Args:
        root (str): path to the top of the dataset
        split (str): data types, 'train' or 'test'
        contain_both (bool): if contain exhausted masks
    """
    
    data_reader = CoNSeP(root=root, download=False) if root is not None else CoNSeP(download=False)
    IDX_LIMITS = data_reader.IDX_LIMITS
    
    for i in range(1, IDX_LIMITS[split] + 1):
        print('Generating {} dataset... {:02d}/{:02d}'.format(split, i, IDX_LIMITS[split]), end='\r')
        
        ori = data_reader.read_image(i, split)
        lab, type_ = data_reader.read_labels(i, split)
        point_mask = data_reader.read_points(i, split)
        
        # get cluster masks
        seg_mask, edges = gen_pseudo_label(ori, point_mask, return_edge=True)
        seg_mask_w_edges = seg_mask & (edges == 0)

        # generate distance maps
        for flip in range(4):
            # generate flipped images
            lab_, seg_mask_w_edges_, point_mask_ = [flip_image(tar, flip) for tar in [lab, seg_mask_w_edges, point_mask]]
            
            # morror padding (1000x1000 to 1230x1230, 95 at left and top, 135 at right and bottom)
            lab_, seg_mask_w_edges_, point_mask_ = [np.pad(tar, ((95, 135), (95, 135)), mode='reflect') for tar in [lab_, seg_mask_w_edges_, point_mask_]]

            # get distance map
            h_map, v_map = get_distancemaps(point_mask_, seg_mask_w_edges_)
            pseudo_mask = np.stack((seg_mask_w_edges_.astype(np.float32), h_map, v_map), axis=-1)
            if contain_both:
                seg_gt = np.where(lab_ > 0, 1, 0)
                h_map, v_map = get_distancemaps(None, lab_, use_full_mask=True)
                full_mask = np.stack((seg_gt.astype(np.float32), h_map, v_map), axis=-1)

            # save npy file (and png file for visualization)
            path_pseudo = '{}/{}/PseudoLabels'.format(root, split.capitalize())
            os.makedirs(path_pseudo, exist_ok=True)
            imsave('{}/{}_{}_{}.png'.format(path_pseudo, split, i, flip), seg_mask_w_edges_.astype(np.uint8) * 255)
            np.save('{}/{}_{}_{}.npy'.format(path_pseudo, split, i, flip), pseudo_mask)
            if contain_both:
                path_full = '{}/{}/FullLabels'.format(root, split.capitalize())
                os.makedirs(path_full, exist_ok=True)
                imsave('{}/{}_{}_{}.png'.format(path_full, split, i, flip), seg_gt.astype(np.uint8) * 255)
                np.save('{}/{}_{}_{}.npy'.format(path_full, split, i, flip), full_mask)
    print('')

def data_reader(root=None, split='train', channel_first=True, doflip=False, contain_both=False, part=None):
    """
    Return images and labels according to the type from split

    Args:
        root (str): path to the top of the dataset
        split (str): data types, 'train' or 'test'
        channel_first (bool): if channel is first or not
        contain_both (bool): if contain exhausted masks
        part (tuple, list): selected indice of data
    
    Returns:
        (tuple):
            images (numpy.ndarray)
            labels (numpy.ndarray)
    """
    data_reader = CoNSeP(root=root, download=False) if root is not None else CoNSeP(download=False)
    IDX_LIMITS = data_reader.IDX_LIMITS
    # select indice from dataset if customization is needed
    indice = range(1, IDX_LIMITS[split] + 1) if part is None else part
    images = []
    labels = []
    # fulllabels = []
    # pseudolabels = []
    for i, idx in enumerate(indice):
        print('Loading {} dataset... {:02d}/{:02d}'.format(split, i + 1, len(indice)), end='\r')
        # original image
        image = data_reader.read_image(idx, split) / 255

        label_path = data_reader.get_path(idx, split, 'label')
        flip_idx = range(4) if doflip else (0, )
        for flip in flip_idx:
            image_ = flip_image(image, flip)
            image_ = np.pad(image_, ((95, 135), (95, 135), (0, 0)), mode='reflect')
            images.append(image_)

            # pseudo labels
            pseudolabel_path = label_path.replace('Labels', 'PseudoLabels').replace('.npy', '_{}.npy'.format(flip))
            pseudolabels = np.load(pseudolabel_path)

            # full labels (optional)
            if contain_both:
                fulllabel_path = label_path.replace('Labels', 'FullLabels').replace('.npy', '_{}.npy'.format(flip))
                fulllabels = np.load(fulllabel_path)
                labels.append(np.concatenate((pseudolabels, fulllabels), axis=-1))
            else:
                labels.append(pseudolabels)
    print('')
    return images, labels

class AugmentedDataset(Dataset):
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

class CoNSeP_cropped(Dataset):
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
                        lbls_append(lbls[h+gap:h+gap+self.validsize, w+gap:w+gap+self.validsize, :])
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
    gen_pseudo_masks(split='train', contain_both=True)
    gen_pseudo_masks(split='test', contain_both=True)
