
from torchvision.models import resnet18, resnet34

def get_net(name):

    if name=='resnet18':
        return ResNet18
    elif name=='resnet34':
        return ResNet34
    else:
        return "Invalid name"


class ResNet18:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))
    

    def __change_last_layer(self) -> None:
        self.model.fc = torch.nn.Linear(512, self.n_classes)

    def get_embedding_dim() -> int:
        return 512


class ResNet34:

    def __init__(self, n_classes, n_channels, device):
        self.n_classes = n_classes
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = torch.nn.Linear(512, self.n_classes)

    def get_embedding_dim(self) -> int:
        return 512


