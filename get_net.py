
from torchvision.models import resnet18, resnet34
import torch.nn as nn
import torch.nn.functional as F
import resnet

def get_net(name, args):

    if name=='resnet18':
        net = resnet.ResNet18()
        #net = ResNet18(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    elif name=='resnet34':
        net = ResNet34(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    elif name=='net3':
        net = Net3()
    elif name == 'net5':
        net = Net5()
    else:
        return "Invalid name"
    return net

class ResNet18:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=False, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))
    
    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.n_classes)

    def get_embedding_dim(self) -> int:
        return 50


class ResNet34:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet34(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.n_classes)

    def get_embedding_dim(self) -> int:
        return 50

class Net3(nn.Module):
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

class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,  64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3,padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3,padding=1)
        self.fc1 = nn.Linear(256*10*10, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 400)
        self.fc4 = nn.Linear(400, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1, 256 * 10 * 10)
        e1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(e1))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, e1
    
    def get_embedding_dim(self) -> int:
        return 50
