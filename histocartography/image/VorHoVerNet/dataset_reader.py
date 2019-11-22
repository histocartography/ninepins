import numpy as np
from skimage.io import imread, imsave
from constants import DATASET_IDX_LIMITS

class CoNSeP:
    """
    TODO: data should be downloaded from S3
    Dataset wrapper class for reading images and labels from CoNSeP dataset
    """
    IDX_LIMITS = DATASET_IDX_LIMITS["CoNSeP"]

    def get_path(self, idx, split, type_):
        path = "CoNSep/"
        if type_ == "image":
            path += f"{split.capitalize()}/Images/{split}_{idx}.png"
        else:
            path += f"{split.capitalize()}/Labels/{split}_{idx}.npy"
        return path

    def check_idx(self, idx, split):
        assert 1 <= idx <= self.IDX_LIMITS[split], f"Invalid index {idx} for split {split}."

    def read_image(self, idx, split):
        self.check_idx(idx, split)
        return imread(self.get_path(idx, split, "image"))

    def read_labels(self, idx, split):
        self.check_idx(idx, split)
        label = np.load(self.get_path(idx, split, "label"))
        return label[..., 0], label[..., 1]
