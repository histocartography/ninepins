#!/usr/bin/env python3

import ninepins.histocartography.io.utils as utils

BUCKET = "curated-datasets"
DATASET = "colorectal/PATCH/CoNSeP"
DATA_PATH = "../../../../data/"

utils.download_s3_dataset(
	utils.get_s3(), BUCKET, DATASET, DATA_PATH
)
