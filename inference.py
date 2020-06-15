import os
import random
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from gans import GeneratorBasic, GeneratorSAGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_type', '-t', type=str, default='DCGAN',
                        help='Which model type to load.')
    parser.add_argument('-model_file', '-m', type=str, default=None,
                        help='Directory path for dataset.')
    parser.add_argument('-seed', '-s', type=int, default=999,
                        help='Set -dev to True so progress images are not displayed.')
    parser.add_argument('-num_channel', '-nc', type=int, default=3,
                        help='Number of channels in the training images.')
    parser.add_argument('-z_size', '-nz', type=int, default=100,
                        help='Size of z latent vector (i.e. size of generator input).')
    parser.add_argument('-gen_features', '-ngf', type=int, default=64,
                        help='Size of feature maps in generator, relates to depth.')
    parser.add_argument('-num_gpu', '-ngpu', type=int, default=1,
                        help='Number of GPUs to run on system. Set to 0 for CPU only.')
    parser.add_argument('-num_images', '-ni', type=int, default=64,
                        help='Number of images to generator.')
    parser.add_argument('-show_real', '-r', type=bool, default=False,
                        help='Show real images along side generated.')
    parser.add_argument('-data_path', '-p', type=str, default='./data/CelebA',
                        help='Directory path for dataset')

    args = parser.parse_args()

    print("Random Seed, {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    nc = args.num_channel  # Number of channels in the training images
    nz = args.z_size  # Size of z latent vector (i.e. size of generator input)
    ngf = args.gen_features  # Size of feature maps in generator, relates to depth
    ngpu = args.num_gpu  # Number of GPUs available.

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create generator and discriminator models
    if args.model_type == 'DCGAN':
        netG = GeneratorBasic(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
    elif args.model_type == 'SAGAN':
        netG = GeneratorSAGAN(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
    else:
        print("Unsupported Model type!")
        sys.exit()
        # Do multi-gpu, if possible

    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.load_state_dict(torch.load(args.model_file))
    netG.eval()
    print(netG)

    # Create batch of latent vectors used to visualize
    # the progression of generator
    noise = torch.randn(args.num_images, nz, 1, 1, device=device)

    # Do model inference using random noise vector
    fake = netG(noise).detach().cpu()

    result = vutils.make_grid(fake, padding=2, normalize=True)

    plt.figure(figsize=(15, 15))
    if args.show_real:
        # Load dataset
        dataset = dset.ImageFolder(root=args.data_path,
                                   transform=transforms.Compose([
                                       transforms.Resize(ngf),
                                       transforms.CenterCrop(ngf),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.num_images,
                                                 shuffle=True, num_workers=1, pin_memory=False)
        real_batch = next(iter(dataloader))

        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                          padding=5, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(1, 2, 2)

    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(result, (1, 2, 0)))
    plt.show()
    plt.savefig("generated.png", bbox_inches='tight')
