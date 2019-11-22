Colorectal Nuclear Segmentation and Phenotypes (CoNSeP) Dataset

----------------------------------------------------------------------------------------------------
Overview:

This dataset was first used in our paper named,

"HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"

If using any part of this dataset or the code associated, you must give appropriate citation to 
our paper, published in Medical Image Analysis.

----------------------------------------------------------------------------------------------------

Dataset Description:

Each label is stored in the form of a H x W x 2 array, where H and W = 1000.

First channel = instance map
Second channel = class map

The instance map gives a unique integer for each individual nucleus. i.e the map ranges from 0 to N,
where 0 is the background and N is the number of nuclei

The values within the class map indicate the category of each nucleus.  

Class values: 1 = other
	      2 = inflammatory
	      3 = healthy epithelial
	      4 = dysplastic/malignant epithelial
              5 = fibroblast
              6 = muscle
	      7 = endothelial

Note, in our paper we combine classes 3 & 4 into the epithelial class and 5, 6 & 7
into the spindle-shaped class.


Total number of nuclei = 24,319

----------------------------------------------------------------------------------------------------
