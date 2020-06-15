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
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from gans import init_weights, GeneratorBasic, DiscriminatorBasic, \
    DiscriminatorSAGAN, GeneratorSAGAN
from losses import loss_discriminator, loss_generator

if __name__ == '__main__':
    """
        DEFINE DATASET AND TRAINING HYPERPARAMETERS
    """
    manualSeed = 999
    print("Random Seed, {}".format(manualSeed))
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = os.path.abspath("D:\CelebA")
    model_type = 'SAGAN'  # Supported : DCGAN, SAGAN
    batch_size = 128
    image_size = 64
    nc = 3  # Number of channels in the training images
    nz = 100  # Size of z latent vector (i.e. size of generator input)
    ngf = 64  # Size of feature maps in generator, relates to depth
    ndf = 64  # Size of feature maps in discriminator, relates to depth
    num_epochs = 50  # Number of training epochs
    lr = 0.0002
    beta1 = 0.5  # Beta1 hyperparam for Adam optimizers
    ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
    workers = 32  # number of workers for dataloader
    loss_type = 'BCE'
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

    if loss_type == 'BCE':
        criterion = nn.BCELoss()
    elif loss_type == 'Hinge':
        criterion = nn.HingeEmbeddingLoss()
    elif loss_type == 'Wass':
        fake_label = -1
        # ToDO: Implement Wasserstein Loss as class
    elif loss_type == 'DCGAN':
        pass  # ToDO: Implement DCGAN Loss as class
    else:
        raise ValueError('''Unsupported Loss Type!
                            Supported losses for Discriminator are:
                            \'BCE\', \'Wass\', \'Hinge\'''')

    # Create batch of latent vectors used to visualize
    # the progression of generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Conventions for real and fake labels during training
    real_label = 1
    fake_label = 0

    print("Loss type: {}".format(criterion))

    # Setup optimizers for Generator and Discriminator
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    """
        TRAINING LOOP
    """
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # errD_real = -torch.mean(output)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            # print(netD.main[11].weight.grad)
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # errD_fake = torch.mean(output)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            if i % discrim_iters == 0:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # errG = -torch.mean(output)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # Ssve models after training
    torch.save(netD.state_dict(), 'saved_models/discriminator_' + model_type + '.pt')
    torch.save(netG.state_dict(), 'saved_models/generator' + model_type + '_' + '.pt')

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
