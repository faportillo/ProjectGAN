import argparse
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
    parser.add_argument('-dev', type=bool, default=False,
                        help='Set -dev to True so progress images are not displayed')
    args = parser.parse_args()
    """
        DEFINE DATASET AND TRAINING HYPERPARAMETERS
    """
    '''manualSeed = 999
    print("Random Seed, {}".format(manualSeed))
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)'''

    dataroot = os.path.abspath("D:\CelebA")
    model_type = 'DCGAN'  # Supported : DCGAN, SAGAN
    batch_size = 128
    image_size = 64
    nc = 3  # Number of channels in the training images
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    ngf = 128  # Size of feature maps in generator, relates to depth
    ndf = 128  # Size of feature maps in discriminator, relates to depth
    num_epochs = 10  # Number of training epochs
    g_lr = 0.0001
    d_lr = 0.0004
    beta1 = 0.0  # Beta1 hyperparam for Adam optimizers
    beta2 = 0.9
    loss_type = 'BCE'  # Options: BCE, Hinge, Wass
    ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
    workers = 32  # number of workers for dataloader
    discrim_iters = 3  # Num of times to train discriminator before generator

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

    # netG.apply(init_weights)
    print(netG)

    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # netD.apply(init_weights)
    print(netD)

    # Create batch of latent vectors used to visualize
    # the progression of generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Conventions for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup optimizers for Generator and Discriminator
    optimizerG = optim.Adam(filter(lambda p: p.requires_grad,
                                   netG.parameters()), lr=g_lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad,
                                   netD.parameters()), lr=d_lr, betas=(beta1, beta2))

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
            ############################
            # (1) Update D network:
            ###########################
            for _ in range(discrim_iters):
                ## Train with all-real batch
                netD.zero_grad()
                netG.zero_grad()
                # Format batch
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                # Forward pass REAL batch through Discrim D(x)
                d = netD(real_cpu).view(-1).unsqueeze(1)
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate FAKE image batch with Gen
                fake = netG(noise)
                # Classify all fake batch with Discrim: D(G(z))
                d_g = netD(fake.detach()).view(-1).unsqueeze(1)
                ###test_d = d_g.cpu().detach().numpy() # Test print
                ###print(test_d) # Test print
                # Calculate losses for real and fake batches
                dis_loss = loss_discriminator(d, d_g, loss_type=loss_type, batch_size=b_size)
                # Calculate gradients for Discrim
                dis_loss.backward()
                # Update D
                optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            d_g = netD(fake).view(-1).unsqueeze(1)
            # Calculate Gen's loss based on output
            gen_loss = loss_generator(d_g, loss_type=loss_type, batch_size=batch_size)
            # Calcualte gradient for G
            gen_loss.backward()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[{}/{}][{}/{}]\tLoss_D: {}\tLoss_G: {}'
                      .format(epoch + 1, num_epochs, i, len(dataloader),
                              dis_loss.item(), gen_loss.item()))
            # Save Losses for plotting later
            G_losses.append(dis_loss.item())
            D_losses.append(gen_loss.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Ssve models after training
    torch.save(netD.state_dict(), 'saved_models/discriminator_' + model_type + '.pt')
    torch.save(netG.state_dict(), 'saved_models/generator' + model_type + '.pt')
    if args.dev:
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
