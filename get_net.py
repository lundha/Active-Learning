
from torchvision.models import resnet18, resnet34
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet




def get_net(name, args, strategy):

    if name=='resnet18':
        net = resnet.ResNet18(strategy, args)
    elif name=='resnet34':
        net = ResNet34(n_classes=args['num_classes'], n_channels=args['num_channels'], device=args['device'])
    elif name=='net3':
        net = Net3(args)
    elif name == 'net5':
        net = COAPModNet(args)
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
    def __init__(self, args):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(args['num_channels'], args['img_dim'], kernel_size=5)
        self.conv2 = nn.Conv2d(args['img_dim'], args['img_dim'], kernel_size=5)
        self.conv3 = nn.Conv2d(args['img_dim'], 2*args['img_dim'], kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, args['num_classes'])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = F.relu(x1)
        x3 = self.conv2(x2)
        x4 = F.relu(F.max_pool2d(x3, 2))
        x5 = self.conv3(x4)
        x6 = F.relu(F.max_pool2d(x5, 2))
        x = x6.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1, [x1,x3,x5,x6]

    def get_embedding_dim(self):
        return 50


class test_Net3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(args['num_channels'], args['img_dim'], kernel_size=5)
        self.conv2 = nn.Conv2d(args['img_dim'], args['img_dim'], kernel_size=5)
        self.conv3 = nn.Conv2d(args['img_dim'], 2*args['img_dim'], kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, args['num_classes'])

    def forward(self, x):
        x0 = x
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(F.max_pool2d(x2,2))
        x4 = F.relu(self.conv3(x3))
        x5 = F.relu(F.max_pool2d(x4,2))
        x6 = x5.view(-1, 1024)
        e1 = F.relu(self.fc1(x6))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1, [x0,x1,x2,x3,x4,x5,x6]

    def get_embedding_dim(self):
        return 50



def test(data, args):
    net = test_Net3(args)
    for name, params in net.named_parameters():
        if 'conv' in name:
            print(name, params.size())
   
    y, e1,(x0,x1,x2,x3,x4,x5,x6) = net(data)
    print(x0.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
    print(x6.shape)
    print(e1.shape)
    print(y.shape)

if __name__ == "__main__":
    DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    args = {'img_dim': 32, 'num_channels': 3, 'num_classes': 10}
    data = torch.randn(64,3,32,32)
    from torch.autograd import Variable
    data = Variable(data)

    print(type(data))
    test(data, args)