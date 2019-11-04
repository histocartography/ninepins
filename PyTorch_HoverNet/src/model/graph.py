import torch
import torch.nn as nn
import torch.nn.functional as F



class PreActResBlock(nn.Module):

    def __init__(self, ch_in, m, ksize, strides=1):
        super(PreActResBlock,self).__init__()

        #initialising some parameters
        #ch_in = l#.get_shape().as_list()
        self.multiplication_factor = 4

        self.bn0 = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in, m, kernel_size= ksize[0], bias=False) #padding=1 for same: output should be same size as input(calculated p=1: o=(i-k) +2p+1)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(m)
        self.conv2 = nn.Conv2d(m, m, kernel_size = ksize[1], padding=1, stride=strides,bias=False)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(m)
        self.conv3 = nn.Conv2d(m, m * self.multiplication_factor, kernel_size= ksize[2], bias=False)


        self.shortcut = nn.Sequential()
        if strides != 1 or ch_in != m * self.multiplication_factor:
            self.shortcut = nn.Sequential(nn.Conv2d(ch_in, m * self.multiplication_factor, kernel_size=1, stride=strides, bias=False),
                                          nn.BatchNorm2d(m * self.multiplication_factor))


    def forward(self,x):
        out = F.relu(self.bn0(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn1(out)))
        out = self.conv3(F.relu(self.bn2(out)))
        #out = self.bn3(self.conv3(out))
        out += shortcut
        #out = F.relu(out)
        return out

class Dense_block(nn.Module):
    def __init__(self,ch_in,ksize,split=1, padding =1): ##TODO: check padding= valid for pytorch
        super(Dense_block,self).__init__()
        #Here m=32

        self.bn0 = nn.BatchNorm2d(ch_in)
        self.conv1 = nn.Conv2d(ch_in,128,kernel_size=ksize[0])
        ##ToDO: need to do b=variance scaling for all
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128,32, kernel_size = ksize[1], groups=split,padding=padding)



    def forward(self,x):
        out = self.conv1(F.relu(self.bn0(x)))
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

        self.d1 = nn.Conv2d(3, 64, kernel_size=7, stride=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.layer1 = self.construct_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.construct_layer(block, 128,num_blocks[1],stride=2)
        self.layer3 = self.construct_layer(block, 256,num_blocks[2],stride=2)
        self.layer4 = self.construct_layer(block, 512, num_blocks[3], stride=2)
        #self.freezelayer = self.freezing_funct(freeze,op) ##TODO: check for freeze
        self.conv_bot = nn.Conv2d(2048, 1024, kernel_size=1)

    '''def freezing_funct(self,freeze,op):
        if freeze:
            op.detach()
        return op'''

    def construct_layer(self,building_block,ch_out, num_blocks, stride):
        #strides = [stride] + [1]*(num_blocks-1)
        layers =[]
        for block in range(0,num_blocks):
            if block == 0:
                layers.append(building_block(self.ch_in, ch_out, [1, 3, 1], stride))
            else:
                layers.append(building_block(self.ch_in, ch_out, [1, 3, 1], 1))
            self.ch_in = ch_out * 4
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.BN1(self.d1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out4 = self.conv_bot(out4)
        return [out1, out2, out3, out4]

class Decoder(nn.Module):
    def __init__(self,block,num_blocks):
        super(Decoder,self).__init__()
        #self.out_ch = 32
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conva = nn.Conv2d(1024, 256, kernel_size=5, stride=1) #TODO: valid padding
        self.layera = self.construct_layer_dense(block, 256, num_blocks[0]) ##TODO: check stride
        self.convb = nn.Conv2d(512,512,kernel_size=1,stride=1)
        self.convc = nn.Conv2d(512,128,kernel_size=5,stride=1,padding=1)
        self.layerb = self.construct_layer_dense(block,128,num_blocks[1])
        self.convd = nn.Conv2d(256,256,kernel_size=1,padding=1)
        self.conve = nn.Conv2d(256,64,kernel_size=5) #TODO: check padding


    def construct_layer_dense(self,building_block,in_ch,num_blocks):
        layers_d=[]
        for block in range(0,num_blocks):
            layers_d.append(building_block(in_ch,[1,5],split=4,padding=1)) ##TODO: check padding
            in_ch += 32
        return nn.Sequential(*layers_d)

    def forward(self,x):
        out = self.upsample(x[-1])
        out = torch.add(out,x[-2]) ##TODO: need to check upsampling and adding
        out = self.conva(out)
        out = self.layera(out)
        out1 = self.convb(out)
        out = self.upsample(out1)
        out = torch.add(out, x[-3])
        out2 = self.convc(out)
        out = self.layerb(out2)
        out = self.convd(out)
        out = self.upsample(out)
        out = torch.add(out, x[-4])
        out = self.conve(out)

        return [out1, out2, out]




def PreActResNet():
    return Encoder(PreActResBlock, [3,4,6,3]) ##TODO: add self.freeze value: true or false

def NP():
    return Decoder(Dense_block, [8,4])


'''def test():
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
