import json
import os
from io import BytesIO
import numpy as np
from minio import Minio
from minio.error import ResponseError
from PIL import Image
from skimage.io import imread, imsave
from histocartography.image.VorHoVerNet.constants import DATASET_IDX_LIMITS, MONUSEG_IDX_2_UID
from histocartography.image.VorHoVerNet.utils import get_point_from_instance
from histocartography.image.VorHoVerNet.dataset_readers.BaseDataset import *

class MoNuSeg_common(BaseDataset_common):
    """
    Dataset wrapper class for reading images and labels from MoNuSeg dataset.
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["MoNuSeg"]

class MoNuSeg_local(BaseDataset_local, MoNuSeg_common):
    """
    Dataset wrapper class for reading images and labels from local MoNuSeg dataset.
    """
    def __init__(self, ver=0, itr=0, root="MoNuSeg/"):
        super().__init__(ver=ver, itr=itr, root=root)

    def post_path(self, idx, split, type_, itr):
        uid = MONUSEG_IDX_2_UID[split][idx-1]
        if type_ == "image":
            return f"{split.capitalize()}/Images/{split}_{uid}.png"
        elif type_ == "label":
            return f"{split.capitalize()}/Labels/{split}_{uid}.npy"
        else:
            return super().post_path(idx, split, type_, itr)

class MoNuSeg_S3(BaseDataset_S3, MoNuSeg_common):
    """
    Dataset wrapper class for reading images and labels from S3 MoNuSeg dataset.
    """
    def __init__(self, root="MoNuSeg/", remote="data.digital-pathology.zc2.ibm.com:9000"):
        super().__init__(root=root, remote=remote)

    def get_path(self, idx, split, type_):
        """
        Return the path for the requested data.
        Returns:
            path (str)
        """
        if split == "train":
            split_folder = "training"
        else:
            split_folder = split
        path = self.root
        uid = MONUSEG_IDX_2_UID[split][idx-1]
        if type_ == "image":
            path += f"{split_folder}_data/{split}_{uid}.png"
        else:
            path += f"{split_folder}_data/{split}_nuclei_{uid}.json"
        return path

    def read_labels(self, idx, split):
        """
        Read instance map label and type map label from dataset.
        Returns:
            (tuple)
                instance map (numpy.ndarray[int])
                type map (numpy.ndarray[int])
        """
        self.check_idx(idx, split)
        obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "label"))
        label = self.convert_labels(obj)
        instance_map = np.array(label["detected_instance_map"], dtype=int)
        type_map = 1 * (instance_map > 0)
        return instance_map, type_map

    def read_points(self, idx, split):
        """
        Read point map label from dataset.
        Returns:
            point_map (numpy.ndarray[bool])
        """
        self.check_idx(idx, split)
        obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "label"))
        label = self.convert_labels(obj)
        point_mask = np.zeros(label["image_dimension"][:2], dtype=bool)
        for center in label["inst_centroid"]:
            x, y = map(int, center)
            point_mask[y, x] = True
        return point_mask

class MoNuSeg(BaseDataset):
    LOCAL = MoNuSeg_local
    S3 = MoNuSeg_S3