import json
import os
from io import BytesIO
import numpy as np
from minio import Minio
from minio.error import ResponseError
from PIL import Image
from skimage.io import imread, imsave
from constants import DATASET_IDX_LIMITS
from utils import get_point_from_instance

class CoNSeP:
    """
    Interface for different CoNSeP datasets.
    """
    def __new__(cls, *args, download=True, **kwargs):
        """
        If download is True, return a S3 version instance.
        Otherwise, return a local version instance.
        """
        if download:
            return CoNSeP_S3(*args, **kwargs)
        else:
            return CoNSeP_local(*args, **kwargs)

class CoNSeP_common:
    """
    Dataset wrapper class for reading images and labels from CoNSeP dataset.
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["CoNSeP"]

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

class CoNSeP_local(CoNSeP_common):
    """
    Dataset wrapper class for reading images and labels from local CoNSeP dataset.
    """
    def __init__(self, itr=0, root="CoNSeP/"):
        self.root = root
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
        if type_ == "image":
            path += f"{split.capitalize()}/Images/{split}_{idx}.png"
        elif type_ == "label":
            path += f"{split.capitalize()}/Labels/{split}_{idx}.npy"
        else:
            path += f"{split.capitalize()}/PseudoLabels_{itr:02d}/{split}_{idx}.png"
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
            pseudo_label (numpy.ndarray[uint8])
        """
        self.check_idx(idx, split)
        return imread(self.get_path(idx, split, "pseudo"))

    def __str__(self):
        """
        String representation of the object.
        """
        return "CoNSeP:local;root='{}'".format(self.root)

class CoNSeP_S3(CoNSeP_common):
    """
    Dataset wrapper class for reading images and labels from S3 CoNSeP dataset.
    """
    def __init__(self, root="colorectal/PATCH/CoNSeP/", remote="data.digital-pathology.zc2.ibm.com:9000"):
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
        path = self.root
        if type_ == "image":
            path += f"{split}_data/{split}_{idx}.png"
        else:
            path += f"{split}_data/{split}_nuclei_{idx}.json"
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
        return "CoNSeP:S3;remote='{}', root='{}'".format(self.remote, self.root)
