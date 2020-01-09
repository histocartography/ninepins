import numpy as np
from Voronoi_label import get_voronoi_edges

def get_boundary(img):
    """
    Get the bounding box of the object in an image.
    Args:
        img (numpy.ndarray[any]): the image.
    Returns:
        (list)
            rmin (int): minimum row (top).
            rmax (int): maximum row (bottom).
            cmin (int): minimum column (left).
            cmax (int): maximum column (right).
    """
    row = np.any(img, axis=1)
    col = np.any(img, axis=0)
    rmin, rmax = np.where(row)[0][[0, -1]]
    cmin, cmax = np.where(col)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def area_vhmap(img):
    """
    Generate distance maps for an image with only one object in it.
    Args:
        img (numpy.ndarray[any]): the image
    Returns:
        (tuple)
            v_map (numpy.ndarray[float]): vertical distance map.
            h_map (numpy.ndarray[float]): horizontal distance map.
    """
    rmin, rmax, cmin, cmax = get_boundary(img)
    h_gap = cmax - cmin
    v_gap = rmax - rmin
    # print(rmin, rmax, cmin, cmax, v_gap, h_gap)
    h_step = 2. / (h_gap - 1) if h_gap > 1 else 2
    v_step = 2. / (v_gap - 1) if v_gap > 1 else 2

    v_map, h_map = [np.zeros(img.shape, dtype=np.float32) for _ in range(2)]
    v_block, h_block = [np.zeros((v_gap, h_gap), dtype=np.float32) for _ in range(2)]
    for i in range(h_gap):
        h_block[:, i] = h_step * i - 1.
    for i in range(v_gap):
        v_block[i, :] = v_step * i - 1.
    h_map[rmin:rmax, cmin:cmax] = h_block
    v_map[rmin:rmax, cmin:cmax] = v_block
    
    # cell shape mask
    h_map[img == 0] = 0
    v_map[img == 0] = 0
    
    return h_map, v_map

def get_distancemaps(point_mask, seg_mask, use_full_mask=False):
    """
    Generate distance maps from point mask and segmentation mask.
    Args:
        point_mask (numpy.ndarray[bool]): point mask. (True at nuclear point, False at background)
        seg_mask (numpy.ndarray[int]): segmentation mask.
            If use_full_mask, positive integer indicates instance index and 0 indicates background.
            Otherwise, 1 indicates nuclei while 0 indicates background.
        use_full_mask (bool): whether the segmentation mask is with instance index.
    Returns:
        (tuple)
            v_map (numpy.ndarray[float]): vertical distance map.
            h_map (numpy.ndarray[float]): horizontal distance map.
    """
    seg_mask = seg_mask if len(seg_mask.shape) == 2 else seg_mask[:, :, 0]
    if use_full_mask:
        colormap = seg_mask
        seg_mask = np.where(seg_mask > 0, 1, 0)
    else:
        # get color map
        maps = {}
        edges = get_voronoi_edges(point_mask, extra_out=maps)
        colormap = maps['Voronoi_cell']
    
    # instance masks from colormap
    h_map, v_map = [np.zeros_like(seg_mask, dtype=np.float32) for _ in range(2)]
    to_idx = int(colormap.max()) + 1
    from_idx = int(colormap.min()) + 1
    for i in range(from_idx, to_idx):
        hm, vm = area_vhmap(np.where(colormap == i, seg_mask, 0))
        h_map += hm
        v_map += vm
        
    return h_map, v_map
