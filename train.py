import os
import random
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from gans import init_weights, GeneratorBasic, DiscriminatorBasic, \
    DiscriminatorSAGAN, GeneratorSAGAN
from losses import loss_discriminator, loss_generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', '-p', type=str, default='./data/CelebA',
                        help='Directory path for dataset')
    parser.add_argument('-dev', type=bool, default=False,
                        help='Set -dev to True so progress images are not displayed')
    parser.add_argument('-num_channel', '-nc', type=int, default=3,
                        help='Number of channels in the training images.')
    parser.add_argument('-z_size', '-nz', type=int, default=100,
                        help='Size of z latent vector (i.e. size of generator input).')
    parser.add_argument('-gen_features', '-ngf', type=int, default=64,
                        help='Size of feature maps in generator, relates to depth.')
    parser.add_argument('-dis_features', '-ndf', type=int, default=64,
                        help='Size of feature maps in discriminator, relates to depth.')
    parser.add_argument('-num_gpu', '-ngpu', type=int, default=1,
                        help='Number of GPUs to run on system. Set to 0 for CPU only.')
    parser.add_argument('-num_images', '-ni', type=int, default=64,
                        help='Number of images to generator.')
    args = parser.parse_args()
    """
        DEFINE DATASET AND TRAINING HYPERPARAMETERS
    """
    manualSeed = 999
    print("Random Seed, {}".format(manualSeed))
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = os.path.abspath(args.data_path)
    model_type = 'DCGAN'  # Supported : DCGAN, SAGAN
    batch_size = 128
    image_size = 64
    nc = args.num_channel  # Number of channels in the training images
    nz = args.z_size  # Size of z latent vector (i.e. size of generator input)
    ngf = args.gen_features  # Size of feature maps in generator, relates to depth
    ndf = args.dis_features  # Size of feature maps in discriminator, relates to depth
    num_epochs = 5  # Number of training epochs
    g_lr = 0.0002
    d_lr = 0.0002
    use_scheduler = False
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    beta2 = 0.999
    loss_type = 'BCE'  # Options: BCE, Hinge, Wass, DCGAN
    clip_value = 0.01
    ngpu = args.num_gpu  # Number of GPUs available. Use 0 for CPU mode.
    workers = 32  # number of workers for dataloader
    discrim_iters = 1  # Num of times to train discriminator before generator

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers, pin_memory=True)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    if args.dev is False:
        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                          padding=2, normalize=True).cpu(), (1, 2, 0)))
        # plt.show()

    '''
        CREATE MODELS and INITIALIZE TRAINING VARIABLES
    '''
    # Create generator and discriminator models
    if model_type == 'DCGAN':
        netG = GeneratorBasic(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
        netD = DiscriminatorBasic(nc=nc, ndf=ndf, ngpu=ngpu).to(device)
    elif model_type == 'SAGAN':
        netG = GeneratorSAGAN(nc=nc, nz=nz, ngf=ngf, ngpu=ngpu).to(device)
        netD = DiscriminatorSAGAN(nc=nc, ndf=ndf, ngpu=ngpu).to(device)
    else:
        print("Unsupported Model type!")
        sys.exit()
    # Do multi-gpu, if possible
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(init_weights)
    print(netG)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(init_weights)
    print(netD)

    # Create batch of latent vectors used to visualize
    # the progression of generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Conventions for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup optimizers for Generator and Discriminator
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(beta1, beta2))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
    """
        TRAINING LOOP
    """
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # for each batch in dataloader
        for i, data in enumerate(dataloader, 0):
            netG.train()
            netD.train()
            ############################
            # (1) Update D network:
            ###########################
            ## Train with all-real batch
            optimizerD.zero_grad()
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Forward pass REAL batch through Discrim D(x)
            d = netD(real_cpu).view(-1, 1)
            # Classify all FAKE batch with Discrim: D(G(z))
            fake = netG(noise)
            d_g = netD(fake).view(-1, 1)
            dis_loss = loss_discriminator(d, d_g, loss_type=loss_type,
                                          batch_size=b_size)
            dis_loss.backward()
            # Update D
            optimizerD.step()
            # Clip weights if using Wasserstein Loss
            if loss_type == 'Wass':
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            D_x = d.mean().item()
            D_G_z1 = d_g.mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            D_G_z2 = 0.0
            if i % discrim_iters == 0:
                optimizerG.zero_grad()
                netG.zero_grad()
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                d_g = netD(fake).view(-1, 1)
                # Calculate Gen's loss based on output
                gen_loss = loss_generator(d_g, loss_type=loss_type, batch_size=batch_size)
                # Calcualte gradient for G
                gen_loss.backward()
                # Update G
                optimizerG.step()
                D_G_z2 = d_g.mean().item()

            # Output training stats
            if i % 50 == 0:
                print('[{}/{}][{}/{}]\tLoss_D: {}\tLoss_G: {}\tD(x): {}\tD(G(z)): {}/{}'
                      .format(epoch + 1, num_epochs, i, len(dataloader),
                              dis_loss.item(), gen_loss.item(),
                              D_x, D_G_z1, D_G_z2))
                # Save Losses for plotting later
                G_losses.append(dis_loss.item())
                D_losses.append(gen_loss.item())
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    netG.eval()
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

            if use_scheduler:
                scheduler_d.step()
                scheduler_g.step()

        # Ssve models after training
        torch.save(netD.state_dict(), 'saved_models/discriminator_' + model_type + '_' + loss_type + '.pt')
        torch.save(netG.state_dict(), 'saved_models/generator' + model_type + '_' + loss_type + '.pt')
        if args.dev is False:
            plt.figure(figsize=(10, 5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(G_losses, label="G")
            plt.plot(D_losses, label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            # plt.show()
            plt.savefig('GenDiscrimLoss.png', bbox_inches='tight')

            # %%capture
            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

            HTML(ani.to_jshtml())

            # Grab a batch of real images from the dataloader
            real_batch = next(iter(dataloader))

            # Plot the real images
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(
                np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                              padding=5, normalize=True).cpu(), (1, 2, 0)))
            plt.savefig("real_images.png", bbox_inches='tight')
            # Plot the fake images from the last epoch
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
            plt.show()
            plt.savefig("fake_images.png", bbox_inches='tight')
