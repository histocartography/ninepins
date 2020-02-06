import json
import os
from io import BytesIO
import numpy as np
from minio import Minio
from minio.error import ResponseError
from PIL import Image
from skimage.io import imread, imsave
from histocartography.image.VorHoVerNet.constants import DATASET_IDX_LIMITS
from histocartography.image.VorHoVerNet.utils import get_point_from_instance
from histocartography.image.VorHoVerNet.dataset_readers.BaseDataset import *

class CoNSeP_common(BaseDataset_common):
    """
    Dataset wrapper class for reading images and labels from CoNSeP dataset.
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["CoNSeP"]

class CoNSeP_local(BaseDataset_local, CoNSeP_common):
    """
    Dataset wrapper class for reading images and labels from local CoNSeP dataset.
    """
    def __init__(self, ver=0, itr=0, root="CoNSeP/"):
        super().__init__(ver=ver, itr=itr, root=root)

class CoNSeP_S3(BaseDataset_S3, CoNSeP_common):
    """
    Dataset wrapper class for reading images and labels from S3 CoNSeP dataset.
    """
    def __init__(self, root="CoNSeP/", remote="data.digital-pathology.zc2.ibm.com:9000"):
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
        if type_ == "image":
            path += f"{split_folder}_data/{split}_{idx}.png"
        else:
            path += f"{split_folder}_data/{split}_nuclei_{idx}.json"
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
        type_map = np.zeros_like(instance_map)
        types = label["instance_types"]
        for i, (t,) in enumerate(types):
            type_map[instance_map == (i + 1)] = t
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
        for center in label["instance_centroid_location"]:
            x, y = map(int, center)
            point_mask[y, x] = True
        return point_mask

class CoNSeP(BaseDataset):
    LOCAL = CoNSeP_local
    S3 = CoNSeP_S3