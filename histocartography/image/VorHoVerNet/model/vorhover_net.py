import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

    def __init__(self, weights=[1, 1, 2, 1]):
        super(CustomLoss, self).__init__()
        self.weights = torch.FloatTensor(weights)
    
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
        hk = torch.tensor(hk, requires_grad=False).view(1, 1, 5, 5).repeat(1, batchsize_, 1, 1).to(device)
        vk = torch.tensor(vk, requires_grad=False).view(1, 1, 5, 5).repeat(1, batchsize_, 1, 1).to(device)

        h = maps[..., 0].unsqueeze(0)
        v = maps[..., 1].unsqueeze(0)

        dh = F.conv2d(h, hk, padding=2).permute(0, 2, 3, 1)
        dv = F.conv2d(v, vk, padding=2).permute(0, 2, 3, 1)
        return torch.cat((dh, dv), axis=-1)

    def msge_loss(self, pred, gt, focus):
        focus = torch.cat((focus, focus), axis=-1)
        pred_grad = self.get_gradient(pred)
        gt_grad = self.get_gradient(gt)
        return F.mse_loss(pred_grad, gt_grad)

    def forward(self, preds, gts, prefix=None, mode='single'):
        gts = gts.permute(0, 2, 3, 1)
        gt_seg = gts[..., 0]
        gt_hv = gts[..., 1:3]
        pred_seg = preds[..., 0]
        pred_hv = preds[..., 1:3]
        # binary cross entropy loss
        bce = F.binary_cross_entropy(pred_seg, gt_seg)
        # dice loss
        dice = self.dice_loss(pred_seg, gt_seg)
        # mean square error of distance maps
        mse = F.mse_loss(pred_hv, gt_hv)
        mse_g = self.msge_loss(pred_hv, gt_hv, gt_seg)
        
        loss = bce * self.weights[0] + dice * self.weights[1] + mse * self.weights[2] + mse_g * self.weights[3] 

        if mode == 'single':
            return loss
        
        names = ['loss', 'bce', 'dice', 'mse', 'mse_g']
        losses = [loss, bce, dice, mse, mse_g]
        if prefix is not None:
            names = ['{}_{}'.format(prefix, n) for n in ['loss', 'bce', 'dice', 'mse', 'mse_g']]
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

    def __init__(self, batch_size=16):
        super(Net, self).__init__()

        self.batch_size = batch_size
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=False)
        self.conv_params = list(self.conv0.parameters())
        # self.conv0.weight = self.conv0.weight.permute(2, 3, 1, 0)
        self.bn = nn.BatchNorm2d(64)  # TODO(@frd, cpd:@hun): remove num_batches_tracked in weights
        self.encoder = Encoder(PreActResBlock, [3, 4, 6, 3])
        self.conv_bot = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder_seg = Decoder(Dense_block, [8, 4])
        # self.decoder_cls = Decoder(Dense_block, [8, 4])
        self.decoder_hov = Decoder(Dense_block, [8, 4])
        self.BatchNorm = nn.BatchNorm2d(64)
        # self.conv_cls = nn.Conv2d(64, 5, kernel_size = 1, stride=1, padding=0, bias=True)
        self.conv_seg = nn.Conv2d(64, 2, kernel_size = 1, stride=1, padding=0, bias=True)
        self.conv_hov = nn.Conv2d(64, 2, kernel_size = 1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
#         print('Input shape:', x.shape)
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

        # HoVer Branch
        hov_decodeds = self.decoder_hov(encodeds)
        hov_pred = F.relu(self.BatchNorm(hov_decodeds[-1]))

        logi_hov = self.conv_hov(hov_pred).permute(0, 2, 3, 1)
        pred_hov = logi_hov # legacy of transfered from tensorflow, can be removed 

        predmap_coded = torch.cat((pred_seg, pred_hov), -1)
        return predmap_coded

    def one_hot(self, indices, depth):
        """
        Returns a one-hot tensor.
        PyTorch equivalent of Tensorflow's tf.one_hot.
        https://github.com/kjunelee/MetaOptNet/blob/master/train.py
        """

        enco_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
        index = indices.view(indices.size() + torch.Size([1]))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        index = index.float().to(device)
        encoded_indicies = enco_indicies.scatter_(-1, index, 1) #axis=-1

        return encoded_indicies

    def crop_op(self, x, cropping, data_format='channels_first'):
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
