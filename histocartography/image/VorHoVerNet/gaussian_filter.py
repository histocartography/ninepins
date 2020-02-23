import numpy as np
from histocartography.image.VorHoVerNet.Voronoi_label import get_voronoi_edges

def get_gaussian_mask(point_mask, sigma=5, mag=5.0, constant=1.0):
    # get distance for each pixel
    maps = {}
    get_voronoi_edges(point_mask, extra_out=maps, extra_only=True, l2_norm=True)
    dist_map = maps['dist_map']

    # applt gaussian formula
    return mag * np.exp(-dist_map/(2*sigma**2)) + constant

if __name__ == "__main__":
    import histocartography.image.VorHoVerNet.dataset_reader as dataset_reader
    from histocartography.image.VorHoVerNet.dataset import data_reader, CoNSeP_cropped

    dataset = 'CoNSeP'
    root = None
    ver = 0
    dataset_class = getattr(dataset_reader, dataset)
    data_reader = dataset_class(root=root, download=False, ver=ver) if root is not None else dataset_class(download=False, ver=ver)
    # IDX_LIMITS = data_reader.IDX_LIMITS

    img = data_reader.read_image(idx=1, split='train')[:500, :500, ...]
    point_mask = data_reader.read_points(idx=1, split='train')[:500, :500]
    gaussian_mask = get_gaussian_mask(point_mask, sigma=5, mag=5.0)

    # import matplotlib.pyplot as plt
    # %matplotlib inline
    # fig, ax = plt.subplots(4, 1, figsize=(20, 60))
    # print(gaussian_mask.max(), gaussian_mask.min())
    # ax[1].imshow(img)
    # ax[0].imshow(point_mask)
    # ax[2].imshow(gaussian_mask)
    # # ax[3].imshow(dist_map)
    # plt.show()


