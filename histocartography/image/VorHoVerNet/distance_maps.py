import numpy as np
from Voronoi_label import get_voronoi_edges

def get_boundary(img):
    row = np.any(img, axis=1)
    col = np.any(img, axis=0)
    rmin, rmax = np.where(row)[0][[0, -1]]
    cmin, cmax = np.where(col)[0][[0, -1]]
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def area_vhmap(img):
    rmin, rmax, cmin, cmax = get_boundary(img)
    v_gap = rmax - rmin
    h_gap = cmax - cmin
    # print(rmin, rmax, cmin, cmax, v_gap, h_gap)
    v_step = 2. / (v_gap - 1) if v_gap > 1 else 2
    h_step = 2. / (h_gap - 1) if h_gap > 1 else 2

    v_map, h_map = [np.zeros(img.shape, dtype=np.float32) for _ in range(2)]
    v_block, h_block = [np.zeros((v_gap, h_gap), dtype=np.float32) for _ in range(2)]
    for i in range(v_gap):
        v_block[i, :] = v_step * i - 1.
    for i in range(h_gap):
        h_block[:, i] = h_step * i - 1.
    v_map[rmin:rmax, cmin:cmax] = v_block
    h_map[rmin:rmax, cmin:cmax] = h_block
    
    # cell shape mask
    v_map[img == 0] = 0
    h_map[img == 0] = 0
    
    return v_map, h_map

def get_distancemaps(point_mask, seg_mask, use_full_mask=False):
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
    v_map, h_map = [np.zeros_like(seg_mask, dtype=np.float32) for _ in range(2)]
    to_idx = int(colormap.max()) + 1
    from_idx = int(colormap.min()) + 1
    for i in range(from_idx, to_idx):
        vm, hm = area_vhmap(np.where(colormap == i, seg_mask, 0))
        v_map += vm
        h_map += hm
        
    return v_map, h_map
