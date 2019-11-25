import json
import os
from io import BytesIO
import numpy as np
from minio import Minio
from minio.error import ResponseError
from PIL import Image
from skimage.io import imread, imsave
from constants import DATASET_IDX_LIMITS
from utils import get_point_from_instance, booleanize_point_labels

class CoNSeP:
    """
    Dataset wrapper class for reading images and labels from CoNSeP dataset
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["CoNSeP"]

    def __init__(self, download=True):
        self.download = download
        if self.download:
            self.minioClient = Minio('data.digital-pathology.zc2.ibm.com:9000',
                                access_key=os.environ['ACCESS_KEY'],
                                secret_key=os.environ['SECRET_KEY'],
                                secure=False)
            self.root = "colorectal/PATCH/CoNSeP/"
        else:
            self.root = "CoNSeP/"

    def get_path(self, idx, split, type_):
        path = self.root
        if self.download:
            if type_ == "image":
                path += f"{split}_data/{split}_{idx}.png"
            else:
                path += f"{split}_data/{split}_nuclei_{idx}.json"
        else:
            if type_ == "image":
                path += f"{split.capitalize()}/Images/{split}_{idx}.png"
            else:
                path += f"{split.capitalize()}/Labels/{split}_{idx}.npy"
        return path

    def check_idx(self, idx, split):
        assert 1 <= idx <= self.IDX_LIMITS[split], f"Invalid index {idx} for split {split}."

    def convert_image(self, obj):
        buffer = BytesIO()
        for d in obj.stream():
            buffer.write(d)

        img = Image.open(buffer).convert("RGB")
        return np.array(img)

    def convert_labels(self, obj):
        buffer = BytesIO()
        for d in obj.stream():
            buffer.write(d)

        buffer.seek(0)

        j = json.load(buffer)
        return j

    def read_image(self, idx, split):
        self.check_idx(idx, split)
        if self.download:
            obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "image"))
            return self.convert_image(obj)
        else:
            return imread(self.get_path(idx, split, "image"))

    def read_labels(self, idx, split):
        self.check_idx(idx, split)
        if self.download:
            obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "label"))
            label = self.convert_labels(obj)
            instance_map = np.array(label["detected_instance_map"], dtype=int)
            type_map = np.zeros_like(instance_map)
            types = label["instance_types"]
            for i, (t,) in enumerate(types):
                type_map[instance_map == (i + 1)] = t
            return instance_map, type_map
        else:
            label = np.load(self.get_path(idx, split, "label"))
            return label[..., 0], label[..., 1]

    def read_points(self, idx, split):
        self.check_idx(idx, split)
        if self.download:
            obj = self.minioClient.get_object("curated-datasets", self.get_path(idx, split, "label"))
            label = self.convert_labels(obj)
            point_mask = np.zeros(label["image_dimension"][:2], dtype=bool)
            for center in label["instance_centroid_location"]:
                x, y = map(int, center)
                point_mask[y, x] = True
            return point_mask
        else:
            label = np.load(self.get_path(idx, split, "label"))
            return booleanize_point_labels(get_point_from_instance(label[..., 0]))