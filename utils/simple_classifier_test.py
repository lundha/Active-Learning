
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder, EncoderDataHandler
from torch.utils.data import DataLoader, Dataset
from get_dataset import get_dataset
from config import DATASET, args
import torch
import torchvision.transforms as transforms

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

# Load and visualize images
def load_images(args):
    data_args = args[DATASET]['data_args']
    X_tr, Y_tr, _, _, _, _ = get_dataset(DATASET, data_args)
    return X_tr[:10], Y_tr[:10]

def visualize_images(X, Y, Y_pred, args):
    _, axes = plt.subplots(2,5, figsize=(25,10))
    class_names = args[DATASET]['data_args']['class_names']
    axes = axes.flatten()
    for idx, (image, label, prediction) in enumerate(zip(X, Y, Y_pred)):
        ax = axes[idx]
        ax.imshow(image)
        ax.set_title(f"Label: {class_names[label]}\n Pred: {class_names[prediction]}")
        ax.axis('off')
    plt.savefig('test-test.png')

# Load and print model
def load_model(args, device, pretrained=True):
    encoder = Autoencoder(50, args[DATASET]['data_args']['img_dim'])
    if pretrained:
        encoder.load_state_dict(torch.load(f'{DATASET}_weights.pt'))
    encoder = encoder.to(device)
    return encoder

# Classify 10 images with and without a pre-trained model
def classify_images(X, Y, model, device):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.95,), std=(0.2,))])
    handler = EncoderDataHandler(X, Y, transform)
    dataloader = DataLoader(handler, drop_last=False, batch_size=10)
    P = torch.zeros(len(Y), dtype=Y.dtype)
    with torch.no_grad():
        for x,y,idx in dataloader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            pred = out.max(1)[1]
            P[idx] = pred.cpu()
    return P
    


if __name__ == "__main__":
    X, Y = load_images(args)
    model = load_model(args, device, pretrained=True)
    Y_pred = classify_images(X, Y, model, device)
    print(f"Y_pred: {Y_pred}\n Y:{Y}\n Acc: {round(1.0 * (Y==Y_pred).sum().item() / len(Y_pred),3)}")
    visualize_images(X, Y, Y_pred, args)
    
