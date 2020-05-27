import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm
from utils import image_loader

'''
    Basic DCGAN
'''


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorBasic(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64, ngpu=1):
        super(GeneratorBasic, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # Input is Z, going into convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (nfg*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (nfg*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorBasic(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(DiscriminatorBasic, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


'''
    Self Attention GAN (SAGAN)
'''


class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.channel = in_dim

        self.q_conv = nn.Conv2d(in_channels=self.channel,
                                out_channels=self.channel // 8,
                                kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=self.channel,
                                out_channels=self.channel // 8,
                                kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=self.channel,
                                out_channels=self.channel,
                                kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        m_batchsize, C, width, height = input.size()
        proj_q = self.q_conv(input).view(m_batchsize, -1,
                                         width * height).permute(0, 2, 1)
        proj_k = self.k_conv(input).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_q, proj_k)
        attention = self.softmax(energy)
        proj_v = self.v_conv(input).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_v, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + input
        return out


class GeneratorSAGAN(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64, ngpu=1):
        super(GeneratorSAGAN, self).__init__()
        self.nc = nc
        self.nz = nz
        self.ngf = ngf
        self.ngpu = ngpu

        repeat_num = int(np.log2(self.ngf)) - 3
        mult = 2 ** repeat_num
        print("Generator->OFM Multiplier (First): {}".format(self.ngf // mult))
        self.layers = []
        # Initial layer from random z-vector
        self.layers.append(spectral_norm(
            nn.ConvTranspose2d(self.nz, self.ngf * mult, 4)))
        self.layers.append(nn.BatchNorm2d(self.ngf * mult))
        self.layers.append(nn.ReLU(True))
        # Subsequent layers till reaches desired spatial dimensions
        while mult > 1:
            print("Generator->OFM Multiplier (Loop): {}".format(self.ngf // mult))
            self.layers.append(spectral_norm(
                nn.ConvTranspose2d(self.ngf * mult,
                                   self.ngf * (mult // 2), 4, 2, 1)))
            self.layers.append(nn.BatchNorm2d((self.ngf * mult) // 2))
            self.layers.append(nn.ReLU(True))
            if mult <= 4:  # Put Self-Attention in last two levels before end
                self.layers.append(SelfAttn((self.ngf * mult) // 2))
            mult = mult // 2

        print("Generator->OFM Multiplier (Last): {}".format(self.ngf // mult))
        self.layers.append(spectral_norm(
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1)))
        self.layers.append(nn.Tanh())
        self.main = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.main(input)


class DiscriminatorSAGAN(nn.Module):
    def __init__(self, nc=3, ndf=64, ngpu=1):
        super(DiscriminatorSAGAN, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.ngpu = ngpu

        repeat_num = int(np.log2(self.ndf)) - 3
        mult = 1
        print("Discriminator->OFM Multiplier (First): {}".format(self.ndf * mult))
        self.layers = []
        self.layers.append(spectral_norm(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1)))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        while mult < 2 ** repeat_num:
            print("Discriminator->OFM Multiplier (Loop): {}".format(self.ndf * mult))
            self.layers.append(spectral_norm(
                nn.Conv2d(self.ndf * mult, self.ndf * (mult * 2), 4, 2, 1, bias=False)))
            self.layers.append(nn.BatchNorm2d((self.ndf * mult) * 2))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            if mult > (2 ** (repeat_num - 1)) // 4:
                self.layers.append(SelfAttn((self.ndf * mult) * 2))
            mult = mult * 2
        print("Discriminator->OFM Multiplier (Last): {}".format(self.ndf * mult))
        self.layers.append(nn.Conv2d(
            self.ndf * mult, 1, 4, 1, 0, bias=False))
        self.layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    # Test model construction

    model_type = 'SAGAN'  # Supported : DCGAN, SAGAN
    image_size = 64
    nc = 3  # Number of channels in the training images
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Number of training epochs
    ngpu = 1

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create generator and discriminator models
    if model_type == 'DCGAN':
        netG = GeneratorBasic(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
        netD = DiscriminatorBasic(nc=nc, ndf=ndf, ngpu=ngpu).to(device)
    elif model_type == 'SAGAN':
        netG = GeneratorSAGAN(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
        netD = DiscriminatorSAGAN(nc=nc, ndf=ndf, ngpu=ngpu).to(device)
    else:
        print("Unsupported Model type!")
        exit(1)
    # Print Models
    print("GENERATOR MODEL: {}\n{}".format(model_type, netG))
    print("DISCRIMINATOR MODEL: {}\n{}".format(model_type, netD))

    '''
        GENERATOR/DISCRIMINATOR TESTS
    '''
    # Generate batch of latent vectors
    noise = torch.randn(1, nz, 1, 1, device=device)
    # Generate fake image batch with Gen
    fake = netG(noise)
    label = torch.full((1,), 0, device=device)
    # Classify all fake batch with Discrim
    output = netD(fake.detach()).view(-1)
    print("Output: {}".format(output))
