import torch
import torch.nn as nn
from torch.autograd import Variable


def loss_discriminator(d, d_g, loss_type='BCE', batch_size=None):
    if loss_type == 'BCE':  # Binary Cross-Entropy Error
        try:
            if batch_size is not None:
                return nn.BCEWithLogitsLoss()(d, Variable(torch.ones(batch_size).cuda())).mean() \
                       + nn.BCEWithLogitsLoss()(d_g, Variable(torch.zeros(batch_size).cuda()))
            else:
                raise TypeError
        except TypeError:
            print('TypeError: Batch size cannot be \'None\' for loss type \'BCE\'')
    elif loss_type == 'Wass':  # Wasserstein Distance Loss
        return -d.mean() + d_g.mean()
    elif loss_type == 'Hinge':  # Hinge Loss
        return nn.ReLU()(1.0 - d).mean() + nn.ReLU()(1.0 + d_g).mean()
    else:
        raise ValueError('''Unsupported Loss Type!
                            Supported losses for Discriminator are:
                            \'BCE\', \'Wass\', \'Hinge\'''')


def loss_generator(d_g, loss_type='BCE', batch_size=None):
    if loss_type == 'BCE':  # Binary Cross-Entropy Error
        try:
            if batch_size is not None:
                return nn.BCEWithLogitsLoss()(d_g, Variable(torch.ones(batch_size).cuda()))
            else:
                raise TypeError
        except TypeError:
            print('TypeError: Batch size cannot be \'None\' for loss type \'BCE\'')
    elif loss_type == 'Wass':  # Wasserstein Distance Loss
        return -d_g.mean()
    elif loss_type == 'Hinge':  # Hinge Loss
        return -d_g.mean()
    else:
        raise ValueError('''Unsupported Loss Type!
                            Supported losses for Generator are:
                            \'BCE\', \'Wass\', \'Hinge\'''')
