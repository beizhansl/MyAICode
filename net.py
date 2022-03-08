import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ops.spectral_norm import spectral_norm as SpectralNorm

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),  # 0.2.0_4会报错，需要在最新的分支上AvgPool3d才有padding参数
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)  # 这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
#1.门控卷积的模块
class Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=0,rate=1,activation=nn.LeakyReLU(0.2,inplace=True),use_lrn=True):
        super(Gated_Conv, self).__init__()
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=nn.Conv2d(in_ch, out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=rate)
        self.convmask = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, dilation=rate)
        self.activation=activation
        self.use_lrn = use_lrn
        self.lrn = LRN()
    def forward(self,input):
        x=self.conv(input)
        gate = self.convmask(input)
        gate=torch.sigmoid(gate)#将值限制在0-1之间
        if self.activation == None:
            out=x * gate
        else:
            out = self.activation(x) * gate
        if self.use_lrn:
            out = self.lrn(out)
        return out
#2.门控反卷积的模块
class Gated_DeConv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=0,rate=1,activation=nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_DeConv, self).__init__()
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, bias=False,dilation=rate)
        self.convmask = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding, bias=False,
                                       dilation=rate)
        self.activation=activation
        self.lrn = LRN()
    def forward(self,input):
        x = self.conv(input)
        gate = self.convmask(input)
        gate=torch.sigmoid(gate)#将值限制在0-1之间
        out=self.activation(x) * gate
        #每一层都有outlrn
        out = self.lrn(out)
        return out

class Generator_scfe(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    # input 2 images and output 2 images as well
    def __init__(self,conv_dim=64,  input_nc= 9,repeat_num = 10):
        super(Generator_scfe, self).__init__()
        #down-sampling
        self.conv1 = Gated_Conv(input_nc, conv_dim, 7, 2, 3,use_lrn=False)  # 256
        self.conv2 =Gated_Conv(conv_dim, 2 * conv_dim, 5, 2, 2)  # 128
        self.conv3 =Gated_Conv(2 * conv_dim, 4 * conv_dim, 5, 2, 2)  # 64
        self.conv4 =Gated_Conv(4 * conv_dim, 8 * conv_dim, 5, 2, 2)  # 32
        self.conv5 =Gated_Conv(8 * conv_dim, 8 * conv_dim, 5, 2, 2)  # 16 8 4

        #dilate
        self.dconv1 = Gated_Conv(8 * conv_dim, 8 * conv_dim, 3, 1, 1,rate=1)
        self.dconv2 = Gated_Conv(8 * conv_dim, 8 * conv_dim, 3, 1, 1, rate=1)
        self.dconv3 = Gated_Conv(8 * conv_dim, 8 * conv_dim, 3, 1, 1, rate=1)
        self.dconv4 = Gated_Conv(8 * conv_dim, 8 * conv_dim, 3, 1, 1, rate=1)  #4   512

        #up-sampling
        self.dwconv1 = Gated_DeConv(8 * conv_dim, 8 * conv_dim, 4, 2, 1) #8 16 32
        self.dwconv2 = Gated_DeConv(8 * conv_dim, 4 * conv_dim, 4, 2, 1) #64
        self.dwconv3 = Gated_DeConv(4 * conv_dim, 2 * conv_dim, 4, 2, 1)  # 128
        self.dwconv4 = Gated_DeConv(2 * conv_dim, conv_dim, 4, 2, 1)  # 256
        self.dwconv5 = Gated_DeConv(conv_dim, 3, 4, 2, 1)  # 512

        self.fconv1 = Gated_Conv(16 * conv_dim, 8 * conv_dim, 3, 1, 1)
        self.fconv2 = Gated_Conv(8 * conv_dim, 4 * conv_dim, 3, 1, 1)
        self.fconv3 = Gated_Conv(4 * conv_dim, 2 * conv_dim, 3, 1, 1)
        self.fconv4 = Gated_Conv(2 * conv_dim, conv_dim, 3, 1, 1)
        self.fconv5 = Gated_Conv(12, 3, 3, 1, 1,activation=None,use_lrn=False)  #结束

    def forward(self, x):
        #x就是9通道组合——mat3+sketch1+color3+mask1+nosie1，注意都是乘完mask的
        #直接在这里进行模型的建立即可（比较复杂）
        #encoder            #channel    size
        x1 = self.conv1(x)  #256
        x2 = self.conv2(x1) #128
        x3 = self.conv3(x2) #64
        x4 = self.conv4(x3) #32
        x5 = self.conv5(x4) #16
        x6 = self.conv5(x5) #8
        x7 = self.conv5(x6) #4
        #dilated conv
        x7 = self.dconv1(x7) #4
        x7 = self.dconv2(x7) #4
        x7 = self.dconv3(x7) #4
        x7 = self.dconv4(x7) #4  512
        #decoder
        x8 = self.dwconv1(x7)
        x8 = torch.concat((x6,x8),1)
        x8 = self.fconv1(x8)

        x9 = self.dwconv1(x8)
        x9 = torch.concat((x5, x9), 1)
        x9 = self.fconv1(x9)

        x10 = self.dwconv1(x9)
        x10 = torch.concat((x4, x10), 1)
        x10 = self.fconv1(x10)

        x11 = self.dwconv2(x10)
        x11 = torch.concat((x3, x11), 1)
        x11 = self.fconv2(x11)

        x12 = self.dwconv3(x11)
        x12 = torch.concat((x2, x12), 1)
        x12 = self.fconv3(x12)

        x13 = self.dwconv4(x12)
        x13 = torch.concat((x1, x13), 1)
        x13 = self.fconv4(x13)

        x14 = self.dwconv5(x13)
        x14 = torch.concat((x, x14), 1)
        x14 = self.fconv5(x14)

        output = torch.tanh(x14)

        return output


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(8, conv_dim, kernel_size=3, stride=2, padding=1))) #64     256
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(SpectralNorm(nn.Conv2d(conv_dim, 2 * conv_dim, kernel_size=3, stride=2, padding=1)))  #128    128
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        layers.append(SpectralNorm(nn.Conv2d(2 * conv_dim, 4 * conv_dim, kernel_size=3, stride=2, padding=1)))  #256     64
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        curr_dim = conv_dim * 4
        for i in range(1, repeat_num):                                                          #32 16 8
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.main = nn.Sequential(*layers)

        # conv1 remain the last square size, 256*256-->30*30
        #self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        #conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        h = h.view((x.size(0), -1))
        #out_real = self.conv1(h)
        #return out_real.squeeze(), out_makeup.squeeze()
        return h