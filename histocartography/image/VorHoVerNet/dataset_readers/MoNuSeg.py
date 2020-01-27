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

class MoNuSeg:
    """
    Interface for different MoNuSeg datasets.
    """
    def __new__(cls, *args, download=True, **kwargs):
        """
        If download is True, return a S3 version instance.
        Otherwise, return a local version instance.
        """
        if download:
            return MoNuSeg_S3(*args, **kwargs)
        else:
            return MoNuSeg_local(*args, **kwargs)

class MoNuSeg_common:
    """
    Dataset wrapper class for reading images and labels from MoNuSeg dataset.
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["MoNuSeg"]

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract base class.")

    def check_idx(self, idx, split):
        """
        Check whether an index is within the limit of a dataset split.
        """
        assert 1 <= idx <= self.IDX_LIMITS[split], f"Invalid index {idx} for split {split}."

    def __str__(self):
        raise NotImplementedError("This is an abstract base class.")

    def __repr__(self):
        return str(self)

class MoNuSeg_local(MoNuSeg_common):
    """
    Dataset wrapper class for reading images and labels from local MoNuSeg dataset.
    """
    def __init__(self, ver=0, itr=0, root="MoNuSeg/"):
        self.root = root
        self.ver = ver
        self.itr = itr

    def get_path(self, idx, split, type_, itr=None):
        """
        Return the path for the requested data.
        Returns:
            path (str)
        """
        if itr is None:
            itr = self.itr
        path = self.root
        uid = MONUSEG_IDX_2_UID[split][idx-1]
        if type_ == "image":
            path += f"{split.capitalize()}/Images/{split}_{uid}.png"
        elif type_ == "label":
            path += f"{split.capitalize()}/Labels/{split}_{uid}.npy"
        elif type_ == "pseudo":
            path += f"{split.capitalize()}/version_{self.ver:02d}/PseudoLabels_{itr:02d}/{split}_{idx}.npy"
        elif type_ == "full":
            path += f"{split.capitalize()}/version_{self.ver:02d}/FullLabels/{split}_{idx}.npy"
        return path

    def read_image(self, idx, split):
        """
        Read an input image from dataset.
        Returns:
            image (numpy.ndarray[uint8])
        """
        self.check_idx(idx, split)
        return imread(self.get_path(idx, split, "image"))[..., :3]

    def read_labels(self, idx, split):
        """
        Read instance map label and type map label from dataset.
        Returns:
            (tuple)
                instance map (numpy.ndarray[int])
                type map (numpy.ndarray[int])
        """
        self.check_idx(idx, split)
        label = np.load(self.get_path(idx, split, "label"))
        return label[..., 0], label[..., 1]

    def read_points(self, idx, split):
        """
        Read point map label from dataset.
        Returns:
            point_map (numpy.ndarray[bool])
        """
        self.check_idx(idx, split)
        label = np.load(self.get_path(idx, split, "label"))
        return get_point_from_instance(label[..., 0], binary=True)

    def read_pseudo_labels(self, idx, split):
        """
        Read pseudo segmentation label from dataset.
        Returns:
            pseudo_label (numpy.ndarray[float32])
        """
        self.check_idx(idx, split)
        return np.load(self.get_path(idx, split, "pseudo"))[..., 0]

    def __str__(self):
        """
        String representation of the object.
        """
        return "MoNuSeg:local;root='{}'".format(self.root)

class MoNuSeg_S3(MoNuSeg_common):
    """
    Dataset wrapper class for reading images and labels from S3 MoNuSeg dataset.
    """
    def __init__(self, root="colorectal/PATCH/MoNuSeg/", remote="data.digital-pathology.zc2.ibm.com:9000"):
        self.minioClient = Minio(remote,
                            access_key=os.environ['ACCESS_KEY'],
                            secret_key=os.environ['SECRET_KEY'],
                            secure=False)
        self.root = root
        self.remote = remote

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

    def convert_image(self, obj):
        """
        Convert S3 object into numpy array format.
        Returns:
            converted_image (numpy.ndarray[uint8])
        """
        buffer = BytesIO()
        for d in obj.stream():
            buffer.write(d)
        img = Image.open(buffer).convert("RGB")
        return np.array(img)

    def convert_labels(self, obj):
        """
        Convert S3 object into JSON-like dictionary format.
        Returns:
            converted_label (dict)
        """
        buffer = BytesIO()
        for d in obj.stream():
            buffer.write(d)
        buffer.seek(0)
        j = json.load(buffer)
        return j

    def read_image(self, idx, split):
        """
        Read an input image from dataset.
        Returns:
            image (numpy.ndarray[uint8])
        """
        self.check_idx(idx, split)
        obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "image"))
        return self.convert_image(obj)

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

    def read_pseudo_labels(self, idx, split):
        """
        Read pseudo segmentation label from dataset.
        Returns:
            pseudo_label (numpy.ndarray[uint8])
        TODO: implement this.
        """
        raise NotImplementedError("Pseudo labels have not been uploaded to S3 yet.")

    def __str__(self):
        """
        String representation of the object.
        """
        return "MoNuSeg:S3;remote='{}', root='{}'".format(self.remote, self.root)
