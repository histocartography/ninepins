import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
from dataset_reader import *

def gen_pseudo_masks(root='CoNSeP/', split='train', contain_both=False):
    """
    Generate pseudo labels, including clusters, vertical and horizontal maps.

    Args:
        root (str): path to the top of the dataset
        split (str): data types, 'train' or 'test'
        contain_both (bool): if contain exhausted masks
    """
    import os
    from skimage.io import imsave
    from skimage.morphology import binary_dilation
    from pseudo_label import gen_pseudo_label
    from distance_maps import get_distancemaps
    from utils import get_point_from_instance
    
    data_reader = CoNSeP(root=root, download=False) if root is not None else CoNSeP(download=False)
    IDX_LIMITS = data_reader.IDX_LIMITS
    
    for i in range(1, IDX_LIMITS[split] + 1):
        print('Generating {} dataset... {:02d}/{:02d}'.format(split, i, IDX_LIMITS[split]), end='\r')
        ori = data_reader.read_image(i, split)
        lab, type_ = data_reader.read_labels(i, split)
        point_mask = get_point_from_instance(lab, binary=True)
        
        # get cluster masks
        seg_mask, edges = gen_pseudo_label(ori, point_mask, np.where(point_mask, type_, 0), v2=False, return_edge=True)
        seg_mask_w_edges = seg_mask & (edges == 0)

        # generate distance maps
        point_mask = data_reader.read_points(i, split)
        v_map, h_map = get_distancemaps(point_mask, seg_mask)
        pseudo_mask = np.stack((seg_mask.astype(np.float32), v_map, h_map), axis=-1)
        if contain_both:
            seg_gt = np.where(lab > 0, 1, 0)
            v_map, h_map = get_distancemaps(None, lab, use_full_mask=True)
            full_mask = np.stack((seg_gt.astype(np.float32), v_map, h_map), axis=-1)

        # save npy file (and png file for visualization)
        path_pseudo = '{}/{}/PseudoLabels'.format(root, split.capitalize())
        os.makedirs(path_pseudo, exist_ok=True)
        imsave('{}/{}_{}.png'.format(path_pseudo, split, i), seg_mask.astype(np.uint8) * 255)
        np.save('{}/{}_{}.npy'.format(path_pseudo, split, i), pseudo_mask)
        if contain_both:
            path_full = '{}/{}/FullLabels'.format(root, split.capitalize())
            os.makedirs(path_full, exist_ok=True)
            imsave('{}/{}_{}.png'.format(path_full, split, i), seg_gt.astype(np.uint8) * 255)
            np.save('{}/{}_{}.npy'.format(path_full, split, i), full_mask)
    print('')

def data_reader(root=None, split='train', channel_first=True, contain_both=False, part=None):
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
            labels (numpy.ndarray) / (dict contain two numpy.ndarray if contain_both=True)
    """
    data_reader = CoNSeP(root=root, download=False) if root is not None else CoNSeP(download=False)
    IDX_LIMITS = data_reader.IDX_LIMITS
    # select indice from dataset if customization is needed
    indice = range(1, IDX_LIMITS[split] + 1) if part is None else part
    images = []
    labels = {}
    fulllabels = []
    pseudolabels = []
    for i, idx in enumerate(indice):
        print('Loading {} dataset... {:02d}/{:02d}'.format(split, i + 1, len(indice)), end='\r')
        # original image
        images.append(data_reader.read_image(idx, split))

        # pseudo labels
        fulllabel_path = data_reader.get_path(idx, split, 'label')
        pseudolabel_path = fulllabel_path.replace('Labels', 'PseudoLabels')
        pseudolabels.append(np.load(pseudolabel_path))

        # full labels (optional)
        if contain_both:
            fulllabel_path = fulllabel_path.replace('Labels', 'FullLabels')
            fulllabels.append(np.load(fulllabel_path))
    print('')

    labels['pseudo'] = pseudolabels
    if contain_both:
        labels['full'] = fulllabels
    return images, labels

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
