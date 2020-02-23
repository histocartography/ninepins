import json
import os
from io import BytesIO
import numpy as np
from PIL import Image
from skimage.io import imread, imsave
from histocartography.image.VorHoVerNet.constants import DATASET_IDX_LIMITS, MONUSEG_IDX_2_UID
from histocartography.image.VorHoVerNet.utils import get_point_from_instance
from histocartography.image.VorHoVerNet.dataset_readers.BaseDataset import *

class CoNuSeg_common(BaseDataset_common):
    """
    Dataset wrapper class for reading images and labels from CoNSeP and MoNuSeg dataset.
    """
    C_IDX_LIMITS = DATASET_IDX_LIMITS["CoNSeP"]
    IDX_LIMITS = {k: DATASET_IDX_LIMITS["CoNSeP"][k] + DATASET_IDX_LIMITS["MoNuSeg"][k] for k in ["train", "test"]}

class CoNuSeg_local(BaseDataset_local, CoNuSeg_common):
    """
    Dataset wrapper class for reading images and labels from local CoNSeP dataset.
    """
    ALLOWED_SPLITS = ["train", "test"]
    def __init__(self, ver=0, itr=0, root="CoNuSeg/"):
        # assume that these two datasets are put under the same parent directory.
        self.con_root = root.replace("CoNuSeg", "CoNSeP")
        self.mon_root = root.replace("CoNuSeg", "MoNuSeg")
        super().__init__(ver=ver, itr=itr, root=self.con_root)

    def pre_get_path(self, idx, split):
        if idx > self.C_IDX_LIMITS[split]:
            self.root = self.mon_root
            self.is_mon = True
        else:
            self.root = self.con_root
            self.is_mon = False

    def post_path(self, idx, split, type_, itr):
        if self.is_mon:
            idx -= self.C_IDX_LIMITS[split]
            uid = MONUSEG_IDX_2_UID[split][idx-1]
        else:
            uid = idx
        if type_ == "image":
            return f"{split.capitalize()}/Images/{split}_{uid}.png"
        elif type_ == "label":
            return f"{split.capitalize()}/Labels/{split}_{uid}.npy"
        else:
            return super().post_path(idx, split, type_, itr)

class CoNuSeg_S3(BaseDataset_S3, CoNuSeg_common):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CoNuSeg does not support S3 version")

class CoNuSeg(BaseDataset):
    LOCAL = CoNuSeg_local
    S3 = CoNuSeg_S3