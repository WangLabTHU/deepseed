import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
import pandas as pd

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='../../deepinfomax/data/ecoli_expr_wy.xlsx', isTrain=True, isGpu=True):
        self.path = path
        files = pd.read_csv(self.path)
        seqs = list(files['realB'])
        exprs = list(files['expr'])
        random.seed(0)
        index = list(np.arange(len(seqs)))
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isTrain = isTrain
        self.split_r = 0.9
        self.isGpu = isGpu
        maxE = 1
        minE = 0
        if self.isTrain:
            start, end = 0, int(len(index)*self.split_r)
        else:
            start, end = int(len(index)*self.split_r), len(index)
        for i in range(start, end):
            self.pSeq.append(self.oneHot(seqs[i]))
            self.expr.append((exprs[i] - minE)/(maxE - minE))

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        Z = self.expr[item]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Z = transforms.ToTensor()(np.asarray([[Z]]))
        Z = torch.squeeze(Z)
        Z = Z.float()
        if self.isGpu:
            X, Z = X.cuda(), Z.cuda()
        return {'x': X, 'z':Z}

    def __len__(self):
        return len(self.expr)


