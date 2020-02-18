import math
from torch import nn
from torch.autograd import Function
import torch
import numpy as np
from skimage import measure

import crfloss_cpp

# torch.manual_seed(42)


class CRFLossFunction(Function):

    @staticmethod
    def forward(ctx, input, image, sigma_xy=15.0, sigma_rgb=0.125):
        loss = crfloss_cpp.forward(input, image.float(), sigma_xy, sigma_rgb)
        ctx.save_for_backward(input, image)
        ctx.sigma_xy = sigma_xy
        ctx.sigma_rgb = sigma_rgb
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        input, image = ctx.saved_tensors
        grad_input = crfloss_cpp.backward(grad_output, input, image.float(), ctx.sigma_xy, ctx.sigma_rgb)
        return grad_input, None


class CRFLoss(nn.Module):
    def __init__(self, sigma_xy=15.0, sigma_rgb=0.125):
        super(CRFLoss, self).__init__()
        self.sigma_xy = sigma_xy
        self.sigma_rgb = sigma_rgb

    def forward(self, input, image):
        output = CRFLossFunction.apply(input, image, self.sigma_xy, self.sigma_rgb)
        return output

    def _get_name(self):
        return 'CRFLoss'
