import argparse
import glob
import math
import os
from collections import deque

import cv2
import numpy as np
from scipy import io as sio
## add for predicting and model loader

from config import Config
from misc.utils import rm_n_mkdir
from torch.autograd import Variable

import torch

import json
import operator

import time



class Inferer(Config):
    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
        """
        step_size = self.infer_mask_shape  # check opt/hover.py
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = x[row:row + win_size[0],
                      col:col + win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:  # check opt/hover.py
            mini_batch = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
            np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

        return pred_map

    ####
    def load_weights(self, model,weights_file, ):
        weights = np.load(weights_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))

    def run(self):

        #Load weights
        self.load_weights(self.weight_file)
        Model = self.get_model()
        Model.load_state_dict(torch.from_numpy(np.load(self.weight_file)))
        Model.eval()

        #Loading images
        save_dir = self.inf_output_dir
        file_list = glob.glob('%s/*%s' % (self.inf_data_dir, self.inf_imgs_ext))
        file_list.sort()


        rm_n_mkdir(save_dir)
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]
            print(self.inf_data_dir, basename, end=' ', flush=True)

            ##
            #loading images and transforming
            img = cv2.imread(self.inf_data_dir + filename) #Changed
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if torch.cuda.is_available():
                img.cuda()

            # Turn the input into a Variable
            i = Variable(img)
            #Image format should be [batch_size,channels,height,width
            #Add dimension for batch
            image_tensor = i.unsqueeze_(0)

            inp = image_tensor if not self.input_norm else image_tensor / 255.0
            # Predict class
            predictor = Model(inp)

            ##
            pred_map = self.__gen_prediction(img, predictor)
            sio.savemat('%s/%s.mat' % (save_dir, basename), {'result': [pred_map]})
            print('FINISH')


####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    inferer = Inferer()
    inferer.run()
