
import torch    
import torchvision
from torchvision.models import resnet34
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class Autoencoder(nn.Module):

    def __init__(self, embedding_dim, img_dim):
        super(Autoencoder, self).__init__()
        dim = img_dim
        print(dim)
        self.conv1 = nn.Conv2d(3, dim, kernel_size=5)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5)
        self.conv3 = nn.Conv2d(dim, 2*dim, kernel_size=5)
        self.fc1 = nn.Linear(1024, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=True)
        x = self.fc2(x)
        return x, e1


class ResNetAutoencoder:

    def __init__(self, n_classes, n_channels, embedding_dim, device):
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.model = resnet34(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.embedding_dim)


if __name__ == "__main__":
    encoder = ResNetAutoencoder(10,3,512,'cpu')
    print(encoder.model)
