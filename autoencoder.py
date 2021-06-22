
import torch    
import torchvision
from torchvision.models import resnet34, resnet18
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from config import DATASET, STRATEGY
from get_dataset import get_dataset
from activelearningdataset import ActiveLearningDataset
import torch.optim as optim
from config import args
from PIL import Image

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
        self.model = resnet18(pretrained=True, progress=True)
        self.__change_last_layer()
        self.device = device
        print("The code is running on {}".format(self.device))

    def __change_last_layer(self) -> None:
        self.model.fc = nn.Linear(512, self.embedding_dim)

class EncoderDataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.astype(np.uint8))
            #x = Image.fromarray((x * 255).astype(np.uint8)) # Added to fix some bug, consider removing

            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        print(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            print(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True

def train_autoencoder(ALD):

    epochs = 200
    DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    learning_args = args[DATASET]['learning_args']
    tr_args = learning_args['tr_args']
    te_args = learning_args['te_args']
    valid_args = learning_args['valid_args']
    
    transform = learning_args['transform']
    idx_lb = ALD.index['labeled']
    print(f'Len idx labeled: {len(idx_lb)}')

    handler = EncoderDataHandler(ALD.X[idx_lb], ALD.Y[idx_lb], transform)
    loader = DataLoader(handler, shuffle=False, drop_last=False, batch_size=tr_args['batch_size'], num_workers=tr_args['num_workers'])
    encoder = Autoencoder(50,data_args['img_dim'])
    encoder.train()
    encoder = encoder.to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(encoder.parameters(), lr=tr_args['lr'], weight_decay=tr_args['weight_decay'])

    for epoch in range(epochs):
        val_loss = 0
        print(epoch)
        for _, (x, y, _) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out, _ = encoder(x)
            loss = criterion(out, y)
            val_loss += torch.sum(loss.data) / loss.size(0)
            loss.sum().backward()
            optimizer.step()
        P = predict(ALD, transform, valid_args, encoder, DEVICE)
        valid_acc = round(1.0 * (ALD.Y_valid==P).sum().item() / len(ALD.Y_valid),3)
        print(f"Val loss: {round(val_loss.item(), 3)}")
        print(f"Val acc: {valid_acc}")

    torch.save(encoder.state_dict(), f'{DATASET}_weights.pt')

    new_encoder = Autoencoder(50,data_args['img_dim'])
    new_encoder.load_state_dict(torch.load(f'{DATASET}_weights.pt'))
    new_encoder = encoder.to(DEVICE)
    new_encoder.train()

    dict_a, dict_b = encoder.state_dict(), new_encoder.state_dict()

    print(validate_state_dicts(dict_a, dict_b))

def predict(ALD, transform, valid_args, autoencoder, DEVICE):
    #X, Y = self.ALD.X_test, self.ALD.Y_test

    handler = EncoderDataHandler(ALD.X_valid, ALD.Y_valid, transform)
    loader = DataLoader(handler, shuffle=False, drop_last=False, batch_size=valid_args['batch_size'], num_workers=valid_args['num_workers'])

    autoencoder.eval()
    Y = ALD.Y_valid
    Y = Y.type(torch.LongTensor)    
    P = torch.zeros(len(Y), dtype=Y.dtype)
    with torch.no_grad():
        try:
            for x,y,idx in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out, _ = autoencoder(x)
                pred = out.max(1)[1]
                P[idx] = pred.cpu()
        except Exception as e:
            print(f'Exception: {str(e)}')
    return P


if __name__ == "__main__":
    data_args = args[DATASET]['data_args']
    X_tr, Y_tr, X_te, Y_te, X_val, Y_val = get_dataset(DATASET, data_args)
    ALD = ActiveLearningDataset(X_tr, Y_tr, X_te, Y_te, X_val, Y_val, len(X_tr))
    train_autoencoder(ALD)