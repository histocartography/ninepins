import numpy as np
from minio import Minio
from minio.error import ResponseError
from skimage.io import imread
from histocartography.image.VorHoVerNet.utils import get_point_from_instance

class BaseDataset_common:
    """
    Dataset wrapper class for reading images and labels from dataset.
    """
    IDX_LIMITS = {}

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

class BaseDataset_local:
    """
    Dataset wrapper class for reading images and labels from local dataset.
    """
    ALLOWED_PATH_TYPES = ('image', 'label', 'pseudo', 'full')

    def __init__(self, ver=0, itr=0, root="BaseDataset/"):
        self.root = root
        self.ver = ver
        self.itr = itr

    def get_path(self, idx, split, type_, itr=None):
        """
        Return the path for the requested data.
        Returns:
            path (str)
        """
        self.check_path_types(type_)
        if itr is None:
            itr = self.itr
        post_path = self.post_path(idx, split, type_, itr)
        return self.root + post_path

    def check_path_types(self, type_):
        assert type_ in self.ALLOWED_PATH_TYPES, f"type_ must be one in ({', '.join(self.ALLOWED_PATH_TYPES)}), but got {type_}"

    def post_path(self, idx, split, type_, itr):
        if type_ == "image":
            return f"{split.capitalize()}/Images/{split}_{idx}.png"
        elif type_ == "label":
            return f"{split.capitalize()}/Labels/{split}_{idx}.npy"
        elif type_ == "pseudo":
            return f"{split.capitalize()}/version_{self.ver:02d}/PseudoLabels_{itr:02d}/{split}_{idx}.npy"
        elif type_ == "full":
            return f"{split.capitalize()}/version_{self.ver:02d}/FullLabels/{split}_{idx}.npy"

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
        return "{}:local;root='{}'".format(self.__class__.__name__.replace("_local", ""),self.root)

class BaseDataset_S3:
    """
    Dataset wrapper class for reading images and labels from S3 dataset.
    """
    def __init__(self, root="BaseDataset/", remote="data.digital-pathology.zc2.ibm.com:9000"):
        self.minioClient = Minio(remote,
                            access_key=os.environ['ACCESS_KEY'],
                            secret_key=os.environ['SECRET_KEY'],
                            secure=False)
        self.root = "colorectal/PATCH/" + root
        self.remote = remote

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
        return "{}:S3;remote='{}', root='{}'".format(self.__class__.__name__.replace("_S3", ""), self.remote, self.root)

class BaseDataset:
    """
    Interface for different source datasets.
    """
    LOCAL = BaseDataset_local
    S3 = BaseDataset_S3
    def __new__(cls, *args, download=False, **kwargs):
        """
        If download is True, return a S3 version instance.
        Otherwise, return a local version instance.
        """
        if download:
            return cls.S3(*args, **kwargs)
        else:
            return cls.LOCAL(*args, **kwargs)


if __name__ == "__main__":
    print(BaseDataset())
