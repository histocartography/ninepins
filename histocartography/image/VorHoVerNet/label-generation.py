import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def rgb2gray(img):
    """
    Transform input rgb image to single channel (grayscale) image.
    """
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]) if len(img.shape) == 3 and img.shape[2] != 1 else np.reshape(img, img.shape + (1,))

def get_mass(binary):
    M = cv2.moments(binary)
    return [int(M['m10'] / M['m00']) ,int(M['m01'] / M['m00'])]

def get_mass_map(img, pointsize=0, ignore_size=6):
    """
    Return object masses as point annotations
    Pointsize is flexible.
    """
    mass_map = np.zeros(img.shape[:2], dtype=np.uint8)
    idx_min = int(img.min())
    idx_max = int(img.max())
    for idx in range(idx_min, idx_max + 1, 1):
        # get every object as mask
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img == idx] = 255

        # skip accidental annotations
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cv2.contourArea(contours[0]) <= ignore_size:
            continue

        # get mass coordinates
        cx, cy = get_mass(mask)

        # map to an empty image
        cv2.circle(mass_map, (cx, cy), pointsize, 255, -1)

    return mass_map

def get_voronoi_edges(point_mask, view=False):
    '''
    Input: single channel mask image with 255 as point value. 
    Return: ...
    Note: 'point' means one pixel per point.
    Reference: some codes from https://gist.github.com/bert/1188638
    '''
    def dist_func(row, col):
        nonlocal r, c
        return (row - r)**2 + (col - c)**2
    
    # create voronoi diagram (color_map)
    shape = point_mask.shape[:2]
    color_map = np.zeros(shape, np.int32)
    depth_map = np.ones(shape, np.float32) * 1e308
    points = np.argwhere(point_mask == 255)
    points = np.random.permutation(points) if view else points
    for i, (r, c) in enumerate(points):
        dist_map = np.fromfunction(dist_func, shape)
        color_map = np.where(dist_map < depth_map, i + 1, color_map)
        depth_map = np.where(dist_map < depth_map, dist_map, depth_map)

    # get voronoi edges
    edges = np.zeros(shape, dtype=np.int32)
    edges[:-1, :][color_map[1:, :] != color_map[:-1, :]] = 255
    edges[:, :-1][color_map[:, 1:] != color_map[:, :-1]] = 255

    if view:
        # global ori
        fig, ax = plt.subplots(1, 2)
        for r, c in points:
            cv2.circle(color_map, (c, r), 2, -255, -1)
        # ax[0].imshow(ori)
        ax[0].imshow(color_map)
        ax[1].imshow(edges)
        plt.show()

    return edges

if __name__ == "__main__":
    IDX = 1
    ori = cv2.cvtColor(cv2.imread(f'/Users/hun/Google Drive/IBM/Dataset/CoNSeP/Train/Images/train_{IDX}.png'), cv2.COLOR_BGR2RGB)
    lab = np.load(f'/Users/hun/Google Drive/IBM/Dataset/CoNSeP/Train/Labels/train_{IDX}.npy')

    t1 = time.time()
    mass_map = get_mass_map(lab[..., 0], pointsize=0, ignore_size=6)
    print(time.time() - t1)
    t1 = time.time()
    get_voronoi_edges(mass_map, view=True)
    print(time.time() - t1)
