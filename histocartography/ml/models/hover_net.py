import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from torch.autograd import Variable

class PreActResBlock(nn.Module):

    def __init__(self, ch_in, m, ksize, count,no_blocks, strides=1):
        super(PreActResBlock,self).__init__()

        #initialising some parameters
        #ch_in = l#.get_shape().as_list()
        self.multiplication_factor = 4
        self.count = count
        self.block_number = no_blocks


        self.conv1 = nn.Conv2d(ch_in, m, kernel_size= ksize[0], bias=False)  # padding=1 for same: output should be same size as input(calculated p=1: o=(i-k) +2p+1)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(m)
        self.conv2 = nn.Conv2d(m, m, kernel_size = ksize[1], padding=1, stride=strides,bias=False)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(m)
        self.conv3 = nn.Conv2d(m, m * self.multiplication_factor, kernel_size= ksize[2], bias=False)
        #self.bnlast = nn.BatchNorm2d(ch_in * self.multiplication_factor)


        self.convshortcut = nn.Sequential()
        if strides != 1 or ch_in != m * self.multiplication_factor:
            self.convshortcut = nn.Sequential(nn.Conv2d(ch_in, m * self.multiplication_factor, kernel_size=1, stride=strides, bias=False))

        if self.count != 0:
            self.preact = nn.BatchNorm2d(m * self.multiplication_factor)

        if self.count == (self.block_number-1):
            self.bnlast = nn.BatchNorm2d(m * self.multiplication_factor)



    def forward(self,x):
        #out = F.relu(self.bn0(x))
        #shortcut = self.convshortcut(out) if hasattr(self, 'shortcut') else x
        print("SELF COUNT")
        print(self.count)
        print(x.shape)
        out = self.preact(x) if hasattr(self,'preact') else x
        if hasattr(self,'preact'):
            print("PREACT")
        else:
            print("No preact")
        print("After preact")
        print(out.shape)
        out = self.conv1(out)
        print("out after conv")
        print(out.shape)
        out = self.conv2(F.relu(self.bn1(out)))
        print("After conv2")
        print(out.shape)
        out = self.conv3(F.relu(self.bn2(out)))
        print("After conv3")
        print(out.shape)
        shortcut = self.convshortcut(x) if hasattr(self, 'convshortcut') else x
        if hasattr(self, 'convshortcut'):
            print("Shortcut")
            print(shortcut.shape)
        else:
            print("No scut")
            #print(shortcut.shape)
            print(out.shape)
        #print("Shorcut")
        #print(shortcut.shape)
        #out = self.bn3(self.conv3(out))
        out += shortcut
        out = self.bnlast(out) if hasattr(self,'bnlast') else out

        #out = F.relu(out)
        #self.count +=1
        return out

class Dense_block(nn.Module):
    def __init__(self,ch_in,ksize,split=1, padding =1): ##TODO: check padding= valid for pytorch
        super(Dense_block,self).__init__()
        #Here m=32

        self.preact_bna = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in,128,kernel_size=ksize[0], bias=False)
        ##ToDO: need to do b=variance scaling for all
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,32, kernel_size = ksize[1], groups=split,padding=padding, bias=False)



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
    def __init__(self,block, num_blocks):
        super(Encoder,self).__init__()

        self.ch_in = 64
        out_ch = [64,128,256,512]

        self.group0 = self.construct_layer(block, out_ch[0], num_blocks[0], stride=1)
        self.group1 = self.construct_layer(block, out_ch[1], num_blocks[1],stride=2)
        self.group2 = self.construct_layer(block, out_ch[2], num_blocks[2],stride=2)
        self.group3 = self.construct_layer(block, out_ch[3], num_blocks[3], stride=2)
        #self.freezelayer = self.freezing_funct(freeze,op) ##TODO: check for freeze


    '''def freezing_funct(self,freeze,op):
        if freeze:
            op.detach()
        return op'''

    def construct_layer(self,building_block,ch_out, num_blocks, stride):
        #strides = [stride] + [1]*(num_blocks-1)
        layers =[]

        for block in range(0,num_blocks):
            if block == 0:
                layers.append(building_block(self.ch_in, ch_out, [1, 3, 1],block,num_blocks,stride))
            else:
                #layers.append(self.preact)
                layers.append(building_block(self.ch_in, ch_out, [1, 3, 1], block,num_blocks,1))

            self.ch_in = ch_out * 4
        #layers.append(self.bnlast)

        return nn.Sequential(*layers)

    def forward(self,x):
        #out = F.relu(self.BN1(self.preact(x))) if block
        out1 = self.group0(x)
        print("x")
        print(x.shape)
        print(out1.shape)
        out2 = self.group1(out1)
        out3 = self.group2(out2)
        out4 = self.group3(out3)
        return [out1, out2, out3, out4]

class Decoder(nn.Module):
    def __init__(self,block,num_blocks):
        super(Decoder,self).__init__()
        #self.out_ch = 32
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conva = nn.Conv2d(1024, 256, kernel_size=5, stride=1) #TODO: valid padding
        self.u3_dense_blk = self.construct_layer_dense(block, 256, num_blocks[0]) ##TODO: check stride
        self.convb = nn.Conv2d(512,512,kernel_size=1,stride=1)
        self.convc = nn.Conv2d(512,128,kernel_size=5,stride=1)
        self.u2_dense_blk = self.construct_layer_dense(block,128,num_blocks[1])
        self.convd = nn.Conv2d(256,256,kernel_size=1)
        self.conve = nn.Conv2d(256,64,kernel_size=5) #TODO: check padding


    def construct_layer_dense(self,building_block,in_ch,num_blocks):
        layers_d=[]
        for block in range(0,num_blocks):
            layers_d.append(building_block(in_ch,[1,5],split=4,padding=0)) ##TODO: check padding
            in_ch += 32
        return nn.Sequential(*layers_d)

    def forward(self,x):
        out = self.upsample(x[-1])
        print("Upsampling output of enc")
        print(out.shape)
        out = torch.add(out,x[-2])
        print("After adding")
        print(out.shape)##ERROR!!!!! The size of tensor a (250) must match the size of tensor b (249) at non-singleton dimension 3##TODO: need to check upsampling and adding
        out = self.conva(out)
        print(out.shape)
        out = self.u3_dense_blk(out)
        print(out.shape)
        out1 = self.convb(out)
        print(out1.shape)
        out = self.upsample(out1)
        print(out.shape)
        out = torch.add(out, x[-3])
        print(out.shape)
        out2 = self.convc(out)
        print(out2.shape)
        out = self.u2_dense_blk(out2)
        print(out.shape)
        out = self.convd(out)
        print(out.shape)
        out = self.upsample(out)
        print(out.shape)
        out = torch.add(out, x[-4])
        print(out.shape)
        out = self.conve(out)

        return [out1, out2, out]


#04.11

class Net(nn.Module):
    def __init__(self, constant_weight):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1,bias=False)
        self.conv_params = list(self.conv0.parameters())
        #self.conv0.weight = self.conv0.weight.permute(2, 3, 1, 0)
        self.BN1 = nn.BatchNorm2d(64)  # TODO: remove num_batches_tracked in weights
        self.encoder = Encoder(PreActResBlock, [3, 4, 6, 3])
        self.conv_bot = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
        self.decoder_np = Decoder(Dense_block, [8, 4])
        self.decoder_nc = Decoder(Dense_block, [8, 4])
        self.decoder_hv = Decoder(Dense_block, [8, 4])
        self.BatchNorm = nn.BatchNorm2d(64)
        self.conv_tp = nn.Conv2d(64, 5, kernel_size = 1, stride=1,bias=True)
        self.conv_np = nn.Conv2d(64, 2, kernel_size = 1, stride=1, bias=True)
        self.conv_hv = nn.Conv2d(64, 2, kernel_size = 1, stride=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        #added for testing(constant weights)
        if(constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self,input):
        #added for testing: weights_init
        #self.weights_init()
        out = F.relu(self.BN1(self.conv0(input)))
        print(out.shape)

        output_enc = self.encoder(out) #self.encoder.forward(out)
        out4 = self.conv_bot(output_enc[3])
        print("output of encoder")
        print(out4.shape)
        output_enc[0] = self.crop_op(output_enc[0], (184,184))
        output_enc[1] = self.crop_op(output_enc[1], (72, 72))
        output_enc[3] = out4


        #placeholders for predmap last dimension = 3 channels; batch_size for inference is 16
        #TODO: batch size need to be added
        truemap_coded = Variable(torch.randn(16, 80, 80, 3).type(torch.FloatTensor), requires_grad=False)

        #For classification branch (NC)
        true_type = truemap_coded[...,1]
        true_type = true_type.int() #casts to int32
        #Not done identity
        #one-hot encoding
        #one_type = self.one_hot(true_type,5)
        true_type = torch.unsqueeze(true_type,-1)

        #For NP Branch
        true_np = true_type > 0
        true_np = true_np.int()
        #one_np = self.one_hot(torch.squeeze(true_np),2)

        #For HoVer Branch
        true_hv = truemap_coded[...,-2:]

        ##### NP Branch #####
        output_dec_np = self.decoder_np(output_enc)#self.decoder_np.forward(output_enc)
        output_dec_npx = F.relu(self.BatchNorm(output_dec_np[-1]))

        logi_np = self.conv_np(output_dec_npx)
        logi_np = logi_np.permute(0,2,3,1)
        soft_np = self.softmax(logi_np)
        prob_np = soft_np[...,1]
        prob_np = torch.unsqueeze(prob_np,-1) #expand_dims

        ##### NC Branch #####
        output_dec_nc = self.decoder_nc(output_enc)#self.decoder_nc.forward(output_enc)
        output_dec_ncx = F.relu(self.BatchNorm(output_dec_nc[-1]))

        logi_class = self.conv_tp(output_dec_ncx)
        logi_class = logi_class.permute(0,2,3,1) #~ to tf.transpose
        soft_class = self.softmax(logi_class)

        ##### HoVer Branch #####
        output_dec_hv = self.decoder_hv(output_enc)#self.decoder_hv.forward(output_enc)
        output_dec_hvx = F.relu(self.BatchNorm(output_dec_hv[-1]))

        logi_hv = self.conv_hv(output_dec_hvx)
        logi_hv = logi_hv.permute(0,2,3,1)
        prob_hv = logi_hv
        pred_hv = logi_hv

        predmap_coded = torch.cat((soft_class, prob_np, pred_hv), -1)

        return predmap_coded



    def one_hot(self,indices, depth):
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

    def crop_op(self,x, cropping, data_format='channels_first'):
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









'''def PreActResNet():
    return Encoder(PreActResBlock, [3,4,6,3]) ##TODO: add self.freeze value: true or false

def NP():
    return Decoder(Dense_block, [8,4])


def test():
    encoder_net= PreActResNet()
    decoder_input = encoder_net #TODO: Add input : remove torch.randn
    decoder_net = NP()
    return decoder_net
    y = net() "ADD input" 

class EncoderDecoder(nn.module):
    def __init__(self, ):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(PreActResBlock, [3,4,6,3])
        self.decoder = Decoder(Dense_block, [8,4])

    def forward(self,x):
        z = self.encoder(x)
        output= self.decoder(z)
        
        return output'''
