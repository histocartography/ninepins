import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from histocartography.image.VorHoVerNet.crf_loss.crfloss import CRFLoss


# TODO(@hun): whether add PseudoEdgeNet concept or not
# class AttentionModule(nn.Module):
#     def __init__(self):
#         super(AttentionModule, self).__init__)()

# class PseudoEdgeNetModded(nn.Module):
#     def __init__(self, num_g=4):
#         super(PseudoEdgeNet, self).__init__()

#         self.g = self.construct_g_layer(layer_type=nn.conv2d, num_g=num_g)

#     def construct_g_layer(self, layer_type=nn.conv2d, num_g=4):
#         layers = []
#         for idx in range(num_g):
#             layers.append(layer_type())

class CustomLoss(nn.Module):

    def __init__(self, weights=[1, 1, 1, 1, 2, 1, 3], use_crf=False, dot_branch=False):
        # 'bce', 'crf', 'mbce', 'dice', 'mse', 'msge', 'ddmse'
        super(CustomLoss, self).__init__()
        self.weights = np.array(weights)
#         self.weights = self.weights / sum(self.weights)
        self.use_crf = use_crf
        self.dot_branch = dot_branch

        if self.use_crf:
            self.crfloss = CRFLoss(10.0, 10.0/255)
    
    @staticmethod
    def dice_loss(pred, gt, epsilon=1e-3):
        n = 2. * torch.sum(pred * gt)
        d = torch.sum(pred + gt)
        return 1. - (n + epsilon) / (d + epsilon)
    
    @staticmethod
    def get_gradient(maps):
        """
        Reference: some codes from https://github.com/vqdang/hover_net/blob/master/src/model/graph.py
        """
        def get_sobel_kernel(size):
            assert size % 2 == 1, 'Must be odd, get size={}'.format(size)

            h_range = np.arange(-size//2 + 1, size//2 + 1, dtype=np.float32)
            v_range = np.arange(-size//2 + 1, size//2 + 1, dtype=np.float32)
            h, v = np.meshgrid(h_range, v_range)
            kernel_h = h / (h * h + v * v + 1.0e-15)
            kernel_v = v / (h * h + v * v + 1.0e-15)
            return kernel_h, kernel_v 
        
        batchsize_ = maps.shape[0]
        hk, vk = get_sobel_kernel(5)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        hk = torch.tensor(hk, requires_grad=False).view(1, 1, 5, 5).to(device)
        vk = torch.tensor(vk, requires_grad=False).view(1, 1, 5, 5).to(device)

        h = maps[..., 0].unsqueeze(1)
        v = maps[..., 1].unsqueeze(1)

        dh = F.conv2d(h, hk, padding=2).permute(0, 2, 3, 1)
        dv = F.conv2d(v, vk, padding=2).permute(0, 2, 3, 1)
        return torch.cat((dh, dv), axis=-1)

    @staticmethod
    def dot_distance_loss(pred_dot, pred_hv, gt_dot, gt_hv):
        pred_dot = pred_dot >= 0.5
        pred_focus = torch.cat((pred_dot, pred_dot), axis=-1)
        gt_focus = torch.cat((gt_dot, gt_dot), axis=-1)
        pred_area = pred_focus * pred_hv
        gt_area = gt_focus * gt_hv
        return F.mse_loss(pred_area, gt_area)

    def msge_loss(self, pred, gt, focus):
        focus = torch.cat((focus, focus), axis=-1)
        pred_grad = self.get_gradient(pred)
        gt_grad = self.get_gradient(gt)
        # loss = pred_grad - gt_grad
        # loss = focus * (loss * loss)
        # loss = torch.sum(loss) / (torch.sum(loss) + 1.0e-8)
        return F.mse_loss(pred_grad, gt_grad)
        # return loss

    @staticmethod
    def weighted_bce(pred, gt, weight):
        wmax = weight.max()
        weight = torch.unsqueeze(weight, -1) / wmax
        pred = pred * weight
        gt = gt * weight
        return F.binary_cross_entropy(pred, gt) * wmax

    def forward(self, preds, gts, image, contain='single'):
        # transpose gts to channel last
        gts = gts.permute(0, 2, 3, 1)
        # gt_seg, gt_hv, gt_dot = torch.split(gts[..., :4], [1, 2, 1], dim=-1)
        gt_seg, gt_hv = torch.split(gts[..., :3], (1, 2), dim=-1)
        pred_seg, pred_hv = torch.split(preds, (1, 2), dim=-1)
        if self.use_crf:
            gt_gau = gts[..., 4]

        # binary cross entropy loss with gaussian mask
        if self.use_crf:
            bce = self.weighted_bce(pred_seg, gt_seg, gt_gau)
        else:
            bce = F.binary_cross_entropy(pred_seg, gt_seg)
        # crfloss
        if self.use_crf:
            crf = self.crfloss(pred_seg.permute(0, 3, 1, 2).cpu(), Net.crop_op(image.detach().clone().data.cpu(), (190, 190))).cuda()
        # masked binary cross entropy loss
        if self.dot_branch:
            mbce = F.binary_cross_entropy(pred_dot * gt_dot, gt_dot) * 3 + F.binary_cross_entropy(pred_dot, gt_dot)
            # mbce = F.binary_cross_entropy(pred_dot, gt_dot)
        # dice loss
        dice = self.dice_loss(pred_seg, gt_seg)
        # mean square error of distance maps and their gradients
        mse = F.mse_loss(pred_hv, gt_hv)
        msge = self.msge_loss(pred_hv, gt_hv, gt_seg)
        # mean square error for dot and distance maps
        # ddmse = self.dot_distance_loss(pred_dot, pred_hv, gt_dot, gt_hv)
        
        if self.use_crf and self.dot_branch:
            loss = bce * self.weights[0] + crf * self.weights[1] + mbce * self.weights[2] + dice * self.weights[3] + mse * self.weights[4] + msge * self.weights[5] + ddmse * self.weights[6]
            names = ('loss', 'crf', 'mbce', 'bce', 'dice', 'mse', 'msge', 'ddmse')
            losses = [loss, crf, mbce, bce, dice, mse, msge, ddmse]
        elif self.use_crf:
            loss = bce * self.weights[0] + crf * self.weights[1] + dice * self.weights[3] + mse * self.weights[4] + msge * self.weights[5]
            names = ('loss', 'crf', 'bce', 'dice', 'mse', 'msge')
            losses = [loss, crf, bce, dice, mse, msge]
        elif self.dot_branch:
            loss = bce * self.weights[0] + mbce * self.weights[2] + dice * self.weights[3] + mse * self.weights[4] + msge * self.weights[5] + ddmse * self.weights[6]
            names = ('loss', 'mbce', 'bce', 'dice', 'mse', 'msge', 'ddmse')
            losses = [loss, mbce, bce, dice, mse, msge, ddmse]
        else:
            loss = bce * self.weights[0] + dice * self.weights[3] + mse * self.weights[4] + msge * self.weights[5]
            names = ('loss', 'bce', 'dice', 'mse', 'msge')
            losses = [loss, bce, dice, mse, msge]

        if contain == 'single':
            return loss
        
        # if prefix is not None:
        #     names = ['{}_{}'.format(prefix, n) for n in names]
        return {name: loss for name, loss in zip(names, losses)}

class PreActResBlock(nn.Module):

    def __init__(self, ch_in, m, ksize, count, no_blocks, strides=1):
        super(PreActResBlock, self).__init__()

        #initialising some parameters
        #ch_in = l#.get_shape().as_list()
        self.multiplication_factor = 4
        self.count = count
        self.block_number = no_blocks

        self.conv1 = nn.Conv2d(ch_in, m, kernel_size=ksize[0], bias=False)  # padding=1 for same: output should be same size as input(calculated p=1: o=(i-k) +2p+1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(m)
        self.conv2 = nn.Conv2d(m, m, kernel_size = ksize[1], padding=1, stride=strides, bias=False)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(m)
        self.conv3 = nn.Conv2d(m, m * self.multiplication_factor, kernel_size=ksize[2], bias=False)
        #self.bnlast = nn.BatchNorm2d(ch_in * self.multiplication_factor)

#         self.convshortcut = nn.Sequential()
        if strides != 1 or ch_in != m * self.multiplication_factor:
            self.convshortcut = nn.Conv2d(ch_in, m * self.multiplication_factor, kernel_size=1, stride=strides, bias=False)

        if self.count != 0:
            self.preact = nn.BatchNorm2d(m * self.multiplication_factor)

        if self.count == (self.block_number - 1):
            self.bnlast = nn.BatchNorm2d(m * self.multiplication_factor)

    def forward(self,x):
        #out = F.relu(self.bn0(x))
        #shortcut = self.convshortcut(out) if hasattr(self, 'shortcut') else x
        out = self.preact(x) if hasattr(self,'preact') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn1(out)))
        out = self.conv3(F.relu(self.bn2(out)))
        shortcut = self.convshortcut(x) if hasattr(self, 'convshortcut') else x
        out += shortcut
        out = self.bnlast(out) if hasattr(self,'bnlast') else out
        return out

class Dense_block(nn.Module):

    def __init__(self, ch_in, ksize, split=1, padding=1): ##TODO: check padding= valid for pytorch
        super(Dense_block,self).__init__()
        #Here m=32

        self.preact_bna = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in, 128, kernel_size=ksize[0], bias=False)
        ##ToDO: need to do b=variance scaling for all
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=ksize[1], groups=split, padding=padding, bias=False)

    def forward(self,x):
        out = F.relu(self.preact_bna(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn1(out)))
        crop_t = (x.shape[2] - out.shape[2]) //2
        crop_b = (x.shape[2] - out.shape[2]) - crop_t
        x = x[:, :, crop_t:-crop_b, crop_t:-crop_b]
        out = torch.cat((x, out), 1)
        return out

class Encoder(nn.Module):

    def __init__(self, block_unit, num_blocks):
        super(Encoder, self).__init__()

        self.in_ch = 64
        self.multiplication_factor = 4
        out_ch = [64, 128, 256, 512]

        self.group0 = self.construct_layer(block_unit, out_ch[0], num_blocks[0], stride=1)
        self.group1 = self.construct_layer(block_unit, out_ch[1], num_blocks[1], stride=2)
        self.group2 = self.construct_layer(block_unit, out_ch[2], num_blocks[2], stride=2)
        self.group3 = self.construct_layer(block_unit, out_ch[3], num_blocks[3], stride=2)
        # TODO(@frd, cpd:@hun): check for freeze (from tensorflow)

    def construct_layer(self, block_unit, out_ch, num_blocks, stride):
        layers = []
        ksizes = [1, 3, 1]
        for idx in range(num_blocks):
            if idx == 0:
                layers.append(block_unit(self.in_ch, out_ch, ksizes, idx, num_blocks, stride))
            else:
                layers.append(block_unit(self.in_ch, out_ch, ksizes, idx, num_blocks, 1))
            self.in_ch = out_ch * self.multiplication_factor # extend channel to fit following blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded1 = self.group0(x)
        encoded2 = self.group1(encoded1)
        encoded3 = self.group2(encoded2)
        encoded4 = self.group3(encoded3)
        return [encoded1, encoded2, encoded3, encoded4]

class Decoder(nn.Module):

    def __init__(self, block_unit, num_blocks):
        super(Decoder,self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=5, stride=1, padding=0) # TODO(@frd, cpd:@hun): valid padding; (@hun) add 'padding=0'
        self.u3_dense_blk = self.construct_layer_dense(block_unit, 256, num_blocks[0]) # TODO(@frd, cpd:@hun): check stride
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=5, stride=1, padding=0)
        self.u2_dense_blk = self.construct_layer_dense(block_unit, 128, num_blocks[1])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(256, 64, kernel_size=5, padding=2)

    def construct_layer_dense(self, block_unit, in_ch, num_blocks):
        layers = []
        for idx in range(num_blocks):
            layers.append(block_unit(in_ch, [1, 5], split=4, padding=0))
            in_ch += 32
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        @x: FOUR channels (encoder1~4) from Encoder 
        """
        out = self.upsample(x[-1])
        out = torch.add(out, x[-2])
        out = self.conv1(out)
        out =  self.u3_dense_blk(out)
        decoded1 = self.conv2(out)

        out = self.upsample(decoded1)
        out = torch.add(out, x[-3])
        decoded2 = self.conv3(out)
        
        out = self.u2_dense_blk(decoded2)
        out = self.conv4(out)
        out = self.upsample(out)
        out = torch.add(out, x[-4])
        decoded3 = self.conv5(out)
        return [decoded1, decoded2, decoded3]

class Net(nn.Module):
    MAPPING = {
        'beta': 'bias',
        'gamma': 'weight',
        'mean': 'running_mean',
        'variance': 'running_var'
    }

    BNLAST_BLOCK = {
        'group0': '2',
        'group1': '3',
        'group2': '5',
        'group3': '2'
    }

    def __init__(self, batch_size=16, dot_branch=False):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.dot_branch = dot_branch

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=False)
        self.conv_params = list(self.conv0.parameters())
        # self.conv0.weight = self.conv0.weight.permute(2, 3, 1, 0)
        self.bn = nn.BatchNorm2d(64)  # TODO(@frd, cpd:@hun): remove num_batches_tracked in weights
        self.encoder = Encoder(PreActResBlock, [3, 4, 6, 3])
        self.conv_bot = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder_seg = Decoder(Dense_block, [8, 4])
        if self.dot_branch:
            self.decoder_dot = Decoder(Dense_block, [8, 4])
        self.decoder_hov = Decoder(Dense_block, [8, 4])
        self.BatchNorm = nn.BatchNorm2d(64)
        self.conv_seg = nn.Conv2d(64, 2, kernel_size = 1, stride=1, padding=0, bias=True)
        if self.dot_branch:
            self.conv_dot = nn.Conv2d(64, 2, kernel_size = 1, stride=1, padding=0, bias=True)
        self.conv_hov = nn.Conv2d(64, 2, kernel_size = 1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv0(x)))
        encodeds = self.encoder(out)

        # get and crop outputs from encoder
        encoded4 = self.conv_bot(encodeds[3]) # TODO(@hun): try to simplify
        encodeds[0] = self.crop_op(encodeds[0], (184, 184))
        encodeds[1] = self.crop_op(encodeds[1], (72, 72))
        encodeds[3] = encoded4

        # placeholder for premap last dimension = 3 channels
        # truemap_coded = Variable(torch.randn(self.batch_size, 80, 80, 3).type(torch.FloatTensor), requires_grad=False)

        # segmentation branch
        seg_decodeds = self.decoder_seg(encodeds)
        seg_pred = F.relu(self.BatchNorm(seg_decodeds[-1]))

        logi_seg = self.conv_seg(seg_pred).permute(0, 2, 3, 1)
        soft_seg = self.softmax(logi_seg)
        pred_seg = soft_seg[..., 1]
        pred_seg = torch.unsqueeze(pred_seg, -1)

        # detection branch (dot)
        if self.dot_branch:
            dot_decodeds = self.decoder_dot(encodeds)
            dot_pred = F.relu(self.BatchNorm(dot_decodeds[-1]))

            logi_dot = self.conv_dot(dot_pred).permute(0, 2, 3, 1)
            soft_dot = self.softmax(logi_dot)
            pred_dot = soft_dot[..., 1]
            pred_dot = torch.unsqueeze(pred_dot, -1)

        # HoVer Branch
        hov_decodeds = self.decoder_hov(encodeds)
        hov_pred = F.relu(self.BatchNorm(hov_decodeds[-1]))

        logi_hov = self.conv_hov(hov_pred).permute(0, 2, 3, 1)
        pred_hov = logi_hov # legacy of transfered from tensorflow, can be removed 
        
        if self.dot_branch:
            return torch.cat((pred_seg, pred_hov, pred_dot), -1)
        return torch.cat((pred_seg, pred_hov), -1)

    def load_model(self, ckpt, prefix='model.'):
        """
        Load checkpoint and remove key difference caused by CusBrontes class.
        
        Args:
            state_dict (dict): state dict in checkpoint
            prefix (str): extra prefix to be removed from keys
        """
        from collections import OrderedDict
        new_dict = OrderedDict()
        len_ = len(prefix)
        for key, value in ckpt['state_dict'].items():
            new_key = key[len_:]
            new_dict[new_key] = value
        self.load_state_dict(new_dict)

    def load_pretrained(self, npz, print_name=True):
        cur_dict = self.state_dict()
        for key, value in npz.items():
            try:
                new_key = self.tf2torch(key)
            except KeyError:
                continue
            try:
                if len(value.shape) == 4:
                    value = value.transpose(3, 2, 0, 1)
                cur_dict[new_key].copy_(torch.from_numpy(value))
            except RuntimeError as e:
                if print_name:
                    print(cur_dict[new_key].shape, value.shape)
                raise e
            else:
                if print_name:
                    print(new_key)
        self.load_state_dict(cur_dict)

    def tf2torch(self, key):
        tokens = key[:-2].split('/')
        group = tokens[0]
        if group == 'linear': raise KeyError
        if group.startswith('group'):
            block = tokens[1]
            if block.startswith('block'):
                block = block[5:]
                layer = tokens[2]
                op = tokens[3]
                if op == 'bn':
                    layer = 'bn' + layer[4:] if layer != 'preact' else layer
                    return '.'.join(['encoder', group, block, layer, self.MAPPING[tokens[4]]])
                elif op == 'W':
                    return '.'.join(['encoder', group, block, layer, 'weight'])
                else:
                    raise KeyError
            elif block == 'bnlast':
                return '.'.join(['encoder', group, self.BNLAST_BLOCK[group], block, self.MAPPING[tokens[3]]])
            else:
                raise KeyError
        elif group == 'conv0':
            op = tokens[1]
            if op == 'bn':
                return '.'.join([op, self.MAPPING[tokens[2]]])
            elif op == 'W':
                return 'conv0.weight'
            else:
                raise KeyError
        else:
            raise KeyError
    
    # def load_save_pretrained(self, npz, required_dict, output_path, prefix='model.'):
    #     pre_ckpt = f'{output_path}/model_pre_ckpt_epoch_0.ckpt'

    #     # check if any ckpt exists
    #     import os
    #     if os.path.isdir(output_path):
    #         filename = [name for name in os.listdir(output_path) if name.endswith('.ckpt')]
    #         assert len(filename) == 0 or pre_ckpt.split('/')[-1] in filename, "extra ckpt in output dir rather than pretrained."
    #     else:
    #         os.makedirs(output_path, exist_ok=True)
        
    #     # load, modify keys and save to path
    #     from collections import OrderedDict
    #     new_dict = OrderedDict()
    #     self.load_pretrained(npz, print_name=False)
    #     for key, value in self.state_dict().items():
    #         new_key = prefix + key
    #         new_dict[new_key] = value

    #     new_ckpt = {'state_dict': new_dict}
    #     for k, v in required_dict.items():
    #         new_ckpt[k] = v

    #     torch.save(new_ckpt, pre_ckpt)
    #     print(f"Pretrained weights saved as {pre_ckpt}")

    # def one_hot(self, indices, depth):
    #     """
    #     Returns a one-hot tensor.
    #     PyTorch equivalent of Tensorflow's tf.one_hot.
    #     https://github.com/kjunelee/MetaOptNet/blob/master/train.py
    #     """

    #     enco_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    #     index = indices.view(indices.size() + torch.Size([1]))
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     index = index.float().to(device)
    #     encoded_indicies = enco_indicies.scatter_(-1, index, 1) #axis=-1

    #     return encoded_indicies

    @staticmethod
    def crop_op(x, cropping, data_format='channels_first'):
        """
        Center crop image
        Args:
            cropping is the substracted portion
        """
        crop_t = cropping[0] // 2
        crop_b = cropping[0] - crop_t
        crop_l = cropping[1] // 2
        crop_r = cropping[1] - crop_l
        if data_format == 'channels_first':
            x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
        else:
            x = x[:, crop_t:-crop_b, crop_l:-crop_r]
        return x
