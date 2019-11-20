"""Unit test for complex_module.core."""
import os
import unittest
import warnings
import numpy as np
from PIL import Image
from histocartography.io.wsi import WSI
from histocartography.io.annotations import XMLAnnotation
from histocartography.io.utils import get_s3
from histocartography.io.utils import download_file_to_local


class CoreTestBase():
    """CoreTestCase class."""

    def test_image_at(self):
        """Test image_at."""
        self.wsi.image_at(1)
        self.assertAlmostEqual(1, self.wsi.current_mag)

        image2_5x = self.wsi.image_at(0.5)
        self.assertAlmostEqual(0.5, self.wsi.current_mag)
        Image.fromarray(image2_5x).save("tmp/02_biopsy_0.5x.png")

    def test_tissue_mask(self):
        """Test tissue_mask_at."""
        tissue_mask = self.wsi.tissue_mask_at(0.5)
        Image.fromarray(tissue_mask).save("tmp/03_tissue_mask_0.5x.png")

    def test_patches(self):
        """Test patches"""
        patch_generator = self.wsi.patches(
            size=(256, 256), stride=(256, 256), annotations=True, mag=1
        )
        num_patches = 0
        for patch in patch_generator:
            loc_x, loc_y, mag, image, labels = list(patch.values())
            if np.max(labels) > 0:
                labels = np.uint8(labels * 255 / np.max(labels))

            imagename = "tmp/patches/_{}x{}_image.png".format(loc_x, loc_y)
            labelname = "tmp/patches/_{}x{}_labels.png".format(loc_x, loc_y)
            Image.fromarray(image).save(imagename)
            Image.fromarray(labels).save(labelname)
            num_patches += 1
            #only generate first 10 patches
            if num_patches > 10:
                return 

        print("Total number of patches: {}".format(num_patches))

    def tearDown(self):
        """Tear down the tests."""
        self.wsi.stack.close()
        pass
