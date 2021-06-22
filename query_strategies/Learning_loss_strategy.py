# Python
import os
import random
import sys

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
import visdom
from tqdm import tqdm

# Active Learning
from .Strategy import Strategy

# Learning Loss Custom
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from Lossnet import LossNet

# K-means
from sklearn.metrics import pairwise_distances
from scipy import stats



class Learning_Loss_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, **kwargs):
        super().__init__(ALD, net, args, logger)
        self.blending_constant = 1
        self.loss_net = LossNet().to(device=self.device)
        self.ll_features = None
        
    def query(self, num_query):

        idx_ulb = self.ALD.index['unlabeled']
        unlabeled_loader = self.prepare_loader(self.ALD.X[idx_ulb], self.ALD.Y[idx_ulb], self.args['transform'], self.args['tr_args'], \
                                                drop_last=False)
    
        uncertainty_samples, embedding = self.get_uncertainty(unlabeled_loader)
        embedding = np.asarray(embedding.to(device='cpu'))
        

        split = (int(len(uncertainty_samples) * self.blending_constant))

        top_uncertain_samples = np.argsort(uncertainty_samples)[:split]
        self.blending_constant = self.blending_constant * 0.9

        embedding = embedding[top_uncertain_samples]

        query = self.init_centers(embedding, num_query)

        return query

    def _train(self, epoch, loader_tr, optimizer, loss_optimizer):


        self.classifier.train()
        self.loss_net.train()

        for _, (x, y, _) in enumerate(loader_tr):
           
            x, y = x.to(self.device), y.to(self.device)
            
            # Zero out all gradients due to PyTorch' gradient accumulation
            # on subsequent backward passes
            optimizer.zero_grad()
            loss_optimizer.zero_grad()


            out, _, self.ll_features = self.classifier(x)
            print(self.ll_features[0].shape)
            target_loss = self.criterion(out, y)
            
            # Predict loss on data based on feature extraction from 
            # training model
            pred_loss = self.loss_net(self.ll_features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            # Calculate loss
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss = self.LossPredLoss(pred_loss, target_loss, margin=1.0)
            loss = m_backbone_loss + 1 * m_module_loss

            # Compute gradients
            loss.backward()

            # Update all parameters
            optimizer.step()
            loss_optimizer.step()

    
    def train(self):

        n_epoch = self.args['n_epoch']

        X_valid, Y_valid = self.ALD.X_valid, self.ALD.Y_valid

        optimizer = optim.Adam(self.net.parameters(), lr=self.args['tr_args']['lr'], weight_decay=self.args['tr_args']['weight_decay'])
        loss_optimizer = optim.Adam(self.loss_net.parameters(), lr=self.args['tr_args']['lr'], weight_decay=self.args['tr_args']['weight_decay'])

        
        idx_lb = self.ALD.index['labeled']
        loader_tr = self.prepare_loader(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], self.args['tr_args'])        

        for epoch in range(n_epoch):
            self._train(epoch, loader_tr, optimizer, loss_optimizer)
            val_loss, val_acc, _ = self.val_loss_accuracy(X_valid, Y_valid)
            self.logger.debug(f"Epoch: {epoch}, Val acc: {val_acc}, Val loss: {val_loss}")

    def LossPredLoss(self, input, target, margin=1.0, reduction='mean'):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape
        
        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
        
        if reduction == 'mean':
            loss = torch.sum(torch.clamp(margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif reduction == 'none':
            loss = torch.clamp(margin - one * input, min=0)
        else:
            NotImplementedError()
        
        return loss

    def get_uncertainty(self, unlabeled_loader):
        
        self.classifier.eval()
        self.loss_net.eval()

        uncertainty = torch.tensor([]).to(device=torch.device(self.device))
        embedding_dim = 512
        self.logger.debug(f"Length unlabeled loader: {len(unlabeled_loader.dataset)}, Embedding dim: {embedding_dim}")
        #embedding = torch.zeros([len(unlabeled_loader.dataset), embedding_dim])
        embedding = torch.tensor([]).to(device=self.device)
    
        with torch.no_grad():
            for _, (x, y, _) in enumerate(unlabeled_loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                x = x.float()

                _, e1, features = self.classifier(x)
                pred_loss = self.loss_net(features)
                pred_loss = pred_loss.view(pred_loss.size(0))


                uncertainty = torch.cat((uncertainty, pred_loss), 0)
     
                embedding = torch.cat((embedding, e1), 0)
        '''
        self.logger.debug(embedding)
        self.logger.debug(embedding.shape)
        self.logger.debug(type(embedding))
        '''
        return uncertainty.cpu(), embedding  

    
