import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def wasserstein_loss(output, target):
    return -torch.mean(output * target)


def loss_discriminator(d, d_g, loss_type='BCE', batch_size=None, device=torch.device("cuda:0")):
    if loss_type == 'BCE':  # Binary Cross-Entropy Error
        try:
            if batch_size is not None:
                return (nn.BCELoss()(d, torch.full((batch_size, 1), 1.0, device=device)) \
                        + nn.BCELoss()(d_g, torch.full((batch_size, 1), 0.0, device=device)))
            else:
                raise TypeError
        except TypeError:
            print('TypeError: Batch size cannot be \'None\' for loss type \'BCE\'')
    elif loss_type == 'Wass':  # Wasserstein Distance Loss
        return -torch.mean(d) + torch.mean(d_g)
    elif loss_type == 'Hinge':  # Hinge Loss
        loss = torch.mean(torch.min(Variable(torch.zeros(batch_size, 1).cuda(),
                                             -1.0 + d)))
        loss += torch.mean(torch.min(Variable(torch.zeros(batch_size, 1).cuda(), -1.0 - d_g)))
        return loss
    elif loss_type == 'DCGAN':
        loss = torch.mean(F.softplus(-d))
        loss += torch.mean(F.softplus(d_g))
        return loss
    else:
        raise ValueError('''Unsupported Loss Type!
                            Supported losses for Discriminator are:
                            \'BCE\', \'Wass\', \'Hinge\'''')


def loss_generator(d_g, loss_type='BCE', batch_size=None, device=torch.device("cuda:0")):
    if loss_type == 'BCE':  # Binary Cross-Entropy Error
        try:
            if batch_size is not None:
                return nn.BCEWithLogitsLoss()(d_g, torch.full((batch_size, 1), 1, device=device))
            else:
                raise TypeError
        except TypeError:
            print('TypeError: Batch size cannot be \'None\' for loss type \'BCE\'')
    elif loss_type == 'Wass':  # Wasserstein Distance Loss
        return -d_g.mean()
    elif loss_type == 'Hinge':  # Hinge Loss
        loss = -torch.mean(d_g)
        return loss
    elif loss_type == 'DCGAN':
        loss = torch.mean(F.softplus(-d_g))
        return loss
    else:
        raise ValueError('''Unsupported Loss Type!
                            Supported losses for Generator are:
                            \'BCE\', \'Wass\', \'Hinge\'''')
