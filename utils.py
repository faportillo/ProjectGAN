import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2


def image_loader(filename, imsize=64):
    loader = transforms.Compose([transforms.Scale(imsize),
                                 transforms.ToTensor()])
    img = cv2.imread(filename, -1)
    img = loader(img).float()
    img = Variable(img)
    img = img.unsqueeze(0)
    return img
