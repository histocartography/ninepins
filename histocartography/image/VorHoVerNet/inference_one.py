import torch
import cv2
import numpy as np
from skimage.io import imread, imsave
from utils import shift_and_scale
from post_processing import get_output_from_model, DEFAULT_TRANSFORM
from dataset import CoNSeP_cropped
from model.vorhover_net import Net

def inference_one(ckpt, imagename, saveprefix='inference', inputsize=1230, patchsize=270, validsize=80, imagesize=1000):
    image = imread(imagename)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = np.pad(image, ((95, 135), (95, 135), (0, 0)), mode='reflect').astype(np.float32)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    checkpoint = torch.load(ckpt, map_location=device)
    print('model_name: {}'.format(ckpt))
    model = Net()
    model.load_model(checkpoint)
    model.to(device)
    model.eval()

    # compute dimensions
    rows = int((inputsize - patchsize) / validsize + 1)
    cols = int((inputsize - patchsize) / validsize + 1)
    wholesize = (validsize * rows, validsize * cols)
    total_patches = rows* cols

    # run
    pred_seg = np.zeros(wholesize, dtype=np.float32)
    pred_hor = np.zeros(wholesize, dtype=np.float32)
    pred_vet = np.zeros(wholesize, dtype=np.float32)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            posy = validsize * i
            posx = validsize * j
            img = image[posy: posy + patchsize, posx: posx + patchsize]
            img = img[None, ...]
            img = torch.from_numpy(img).permute((0, 3, 1, 2))

            seg, hor, vet = get_output_from_model(img.to(device), model, transform=DEFAULT_TRANSFORM)

            pred_seg[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = seg
            pred_hor[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = hor
            pred_vet[validsize * i: validsize * (i + 1), validsize * j: validsize * (j + 1)] = vet

            print(f'patch: {idx+1}/{total_patches}', end='\r')
            idx += 1
            
    pred_seg_ = pred_seg[:imagesize, :imagesize]
    pred_hor_ = pred_hor[:imagesize, :imagesize]
    pred_vet_ = pred_vet[:imagesize, :imagesize]

    imsave(saveprefix + "_seg.png", shift_and_scale(pred_seg_, 255, 0).astype(np.uint8))
    imsave(saveprefix + "_hor.png", shift_and_scale(pred_hor_, 255, 0).astype(np.uint8))
    imsave(saveprefix + "_vet.png", shift_and_scale(pred_vet_, 255, 0).astype(np.uint8))

if __name__ == "__main__":
    inference_one("/work/fad11204/outputs/m_eps0_01/checkpoints/m_eps0_01_ckpt_epoch_88.ckpt", "test.png")