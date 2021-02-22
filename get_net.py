
from torchvision.models import resnet18, resnet34
import torch.nn as nn

def get_net(name, args):

    if name=='resnet18':
        net = ResNet18(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    elif name=='resnet34':
        net = ResNet18(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    elif name=='CIFAR_NET':
        net = ResNet18(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    else:
        return "Invalid name"
    return net

class ResNet18:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))
    

    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.n_classes)

    def get_embedding_dim(self) -> int:
        return 512


class ResNet34:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.n_classes)

    def get_embedding_dim(self) -> int:
        return 512

class CIFAR_NET(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

