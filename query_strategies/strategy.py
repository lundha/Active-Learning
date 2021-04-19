import torch
from torch import nn
from dataloader import Resize, Normalize, ToTensor, Convert2RGB, DataHandler
from autoencoder import Autoencoder
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Strategy():
    def __init__(self, ALD, net, args):
        self.ALD = ALD
        self.net = net #net.model
        self.args = args
        self.n_pool = len(ALD.index['unlabeled'])
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.classifier = self.net.to(self.device)

    def query(self):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    '''
    def _train(self, epoch, loader_tr, optimizer):
        accFinal = 0

        for batch_idx, (x,y,idx) in enumerate(loader_tr):
            #y = (torch.max(y,1)[0]).type(torch.LongTensor)

            ### AD HOC
            #y = y.type(torch.LongTensor)

            x,y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out = self.classifier(x)
            loss = nn.CrossEntropyLoss()
            output = loss(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()

            output.backward()
            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)


    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        n_epoch = self.args['n_epoch']
        self.classifier.train()
        optimizer = optim.Adam(self.net.parameters(), lr=0.05, weight_decay=0.000)

        idx_lb = self.ALD.index['labeled']
        loader_tr = self.prepare_loader(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], self.args['loader_tr_args'])
        epoch = 0
        accCurrent = 0.
        while accCurrent < 0.99: 
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            if (epoch % 10 == 0) and (accCurrent < 0.8): # reset if not converging
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = 0.05, weight_decay=0)
    '''
    def _train(self, epoch, loader_tr, optimizer):
        self.classifier.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.classifier(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            optimizer.step()
        return accFinal / len(loader_tr.dataset.X)


    
    def train(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        n_epoch = self.args['n_epoch']
        self.classifier.train()
        optimizer = optim.Adam(self.net.parameters(), lr=args['tr_args']['lr'], weight_decay=args['tr_args']['weight_decay'])

        idx_lb = self.ALD.index['labeled']
        loader_tr = self.prepare_loader(self.ALD.X[idx_lb], self.ALD.Y[idx_lb], self.args['transform'], self.args['tr_args'])
   
        epoch = 1
        accCurrent = 0.
        accPrev = 0.
        accCounter = 0
        while accCurrent < 0.99: 
            accPrev = accCurrent
            accCurrent = self._train(epoch, loader_tr, optimizer)
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)

            if accPrev == accCurrent:
                counter += 1
                if counter > 10:
                    print('Training has converged')
                    break
                elif epoch > 200:
                    print('Training will not converge')
                    break
            elif accPrev != accCurrent:
                counter = 0

            if (epoch % 50 == 0) and (accCurrent < 0.2): # reset if not converging
                self.classifier = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.classifier.parameters(), lr = 0.05, weight_decay=0)

    def predict(self, Xte, Yte):
        loader_te = self.prepare_loader(Xte, Yte, self.args['transform'], self.args['te_args'])
        self.classifier.eval()
        Yte = Yte.type(torch.LongTensor)

        try:
            P = torch.zeros(len(Yte), dtype=Yte.dtype)
        except Exception as e:
            print(f"Exception: {str(e)}")
            P = np.zeros(len(Yte), dtype=Yte.dtype)
        with torch.no_grad():
            for x,y,idx in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, _ = self.classifier(x)
                pred = out.max(1)[1]
                P[idx] = pred.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = self.prepare_loader(X, Y, self.args['transform'], self.args['te_args'])

        self.classifier.eval()
        try:
            probs = torch.zeros([len(Y), len(np.unique(Y))])
        except Exception as e:
            print(f"Exception: {str(e)}")
            P = np.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.classifier(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def prepare_handler(self, X_unlabeled, Y_unlabeled, transform):
        return DataHandler(X_unlabeled, Y_unlabeled, transform)


    def prepare_loader(self, X_unlabeled, Y_unlabeled, transform, args):
        '''
        Creates a dataloader from unlabeled X and Y data points
        :param: X_unlabeled: Unlabeled datapoints Y_unlabeled: Unlabeled labels
        '''
        handler = DataHandler(X_unlabeled, Y_unlabeled, transform)
        loader = DataLoader(handler, shuffle=False, drop_last=True, batch_size=args['batch_size'], num_workers=args['num_workers'])
        return loader

    def get_embedding(self, dataloader, embedding_dim):
        '''
        Create and save embedding 
        :param: dataloader:dataloader, embedding_din:embedding dimension 
        '''
        encoder = Autoencoder(embedding_dim, self.args['img_dim'])
        data_set = self.args['data_set']
        embedding = torch.zeros([len(dataloader.dataset), embedding_dim])
        with torch.no_grad():
            for x, y, idxs in dataloader:
                out, e1 = encoder(x)
                embedding[idxs] = e1.cpu()
        np.save(f'./{data_set}_EMBEDDING.npy', embedding)
        embedding = embedding.numpy()
        return embedding


    def calculate_distance_matrix(self, embedding) -> np.array:
        '''
        Calculate and save distance matrix, input is embedding
        '''
        data_set = self.args['data_set']
        t_start = datetime.now()
        dist_mat = np.matmul(embedding, embedding.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(embedding), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        np.save(f'./{data_set}_DIST_MATRIX.npy', dist_mat)
        print(f"Time taken to generate distance matrix: {datetime.now() - t_start}")
        return dist_mat