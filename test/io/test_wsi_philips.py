"""Unit test for complex_module.core."""
import os
import unittest
import warnings
import numpy as np
from PIL import Image
from histocartography.io.wsi import WSI
from histocartography.io.annotations import ASAPAnnotation
from histocartography.io.utils import get_s3
from histocartography.io.utils import download_file_to_local

from .test_wsi import CoreTestBase


class CoreTestCase(unittest.TestCase, CoreTestBase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up test."""
        warnings.simplefilter("ignore", ResourceWarning)
        os.makedirs("tmp", exist_ok=True)
        os.makedirs("tmp/patches", exist_ok=True)
        self.s3_resource = get_s3()
        self.filename = download_file_to_local(
            s3=self.s3_resource,
            bucket_name='test-data',
            s3file='tumor_105.tif',
            local_name='tmp/philips_00_biopsy.tif'
        )
        self.annotation_file = download_file_to_local(
            s3=self.s3_resource,
            bucket_name='test-data',
            s3file='tumor_105.xml',
            local_name='tmp/philips_00_biopsy_labels.xml'
        )
        annotations = ASAPAnnotation(
            self.annotation_file
        )
        self.wsi = WSI(self.filename, annotations)
