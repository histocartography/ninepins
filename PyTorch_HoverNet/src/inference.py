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
from model.utils import rm_n_mkdir
from torch.autograd import Variable

import torch
from model.graph import Net

import json
import operator

import time



class Inferer(Config):

    def __gen_prediction(self,x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
        """
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]
        #img = x
        #x = x.cpu().numpy()

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
                #sub_patches = torch.cat(win)
        sub_patches = torch.FloatTensor(sub_patches)
        print(sub_patches.shape)
        sub_patches = sub_patches.permute(0,3,1,2)
        if torch.cuda.is_available():
            sub_patches = sub_patches.to(self.device)
            sub_patches.cuda()

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:  # check opt/hover.py
            mini_batch = sub_patches[:self.inf_batch_size]
            #print((np.array(mini_batch)).shape)
            sub_patches = sub_patches[self.inf_batch_size:]
            print("TYPE: mini batch")
            print(type(mini_batch))
            print(mini_batch.shape)
            #if torch.cuda.is_available():
                #sub_patches = torch.from_numpy(sub_patches).float().to(self.device)
            #    mini_batch.cuda()
            with torch.no_grad():
                mini_output = predictor(mini_batch)
                print("Mini-output")#[0]
                print(mini_output.shape)
            mini_output_np = mini_output.cpu().numpy()
            mini_output_np = np.split(mini_output_np, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output_np)
        if len(sub_patches) != 0:
            with torch.no_grad():
                mini_output = predictor(sub_patches)#[0]
            mini_output_np = mini_output.cpu().numpy()
            mini_output_np = np.split(mini_output_np, len(sub_patches), axis=0)
            pred_map.extend(mini_output_np)

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
    def load_weights(self,weights_file):
        weights = np.load(weights_file)
        keys = sorted(weights.keys())
        f = open("original_model_weights.txt", "w+")
        for i, k in enumerate(keys):
            #print(i, k, np.shape(weights[k]))
            f.write("%d %s %s \n" %(i, k, np.shape(weights[k])))
        f.close()


    def run(self):

        data = np.load('/dataT/frd/Test/hover_seg_&_class_CoNSeP.npz')
        dictionary = {}
        #data = np.load('/Users/frd/Documents/Code/CoNSeP/Train/hover_seg_&_class_CoNSeP.npz')
        lst = data.files
        for item in lst:
            dictionary[item] = torch.from_numpy(data[item])  # .permute(3,2,0,1)

        for key, value in dictionary.items():
            if dictionary[key].dim() == 4:
                dictionary[key] = dictionary[key].permute(3, 2, 0, 1)

        print("shape")
        print(dictionary['hv/u3/dense/blk/3/conv1/W:0'].shape)

        #print(dictionary)

        #self.load_weights(self.weight_file)
        #Model = self.get_model()
        print("##### Loading Model #####")
        model = Net()

        model.load_state_dict(dictionary, strict=False)
        print("Model and weights LOADED successfully")
        print("Printing weights of our model")
        i=0

        #commenting out
        g = open("pytorch_model_weights.txt", "w+")

        for param_tensor in model.state_dict():
            print(i, param_tensor, "\t", model.state_dict()[param_tensor].size())
            g.write("%d  %s  %s \n" %(i, param_tensor, model.state_dict()[param_tensor].size()))
            i +=1
        g.close()

        if torch.cuda.is_available():
            model.cuda()

        model.eval()

        #Loading images
        print("Loading Images")
        save_dir = self.inf_output_dir
        file_list = glob.glob('%s/*%s' % (self.inf_data_dir, self.inf_imgs_ext))
        file_list.sort()


        rm_n_mkdir(save_dir)
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            #img = cv2.imread(data_dir + filename)
            img = cv2.imread(self.inf_data_dir+filename) #Changed
            print(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred_map = self.__gen_prediction(img, model)
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

    cuda_avail = torch.cuda.is_available()

    inferer = Inferer()
    inferer.run()

