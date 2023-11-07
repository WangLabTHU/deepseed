import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
from torch.utils.data import DataLoader
import collections
import pandas as pd


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class predictDataset(Dataset):

    def __init__(self, path='../results/case2_results.csv', isGpu=True):
        self.path = path
        df = pd.read_csv(path)
        seqs = list(df['realB'])
        nrows = len(seqs)
        index = list(np.arange(nrows))
        self.pSeq = []
        self.isGpu = isGpu
        start, end = 0, len(index)
        for i in range(start, end):
            self.pSeq.append(self.oneHot(seqs[i]))

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        oh = np.zeros([4, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        X_cut = self.pSeq[item][:, 0 : 75]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        X_cut = transforms.ToTensor()(X_cut)
        X_cut = torch.squeeze(X_cut)
        X_cut = X_cut.float()
        if self.isGpu:
            X = X.cuda()
            X_cut = X_cut.cuda()
        return {'X': X, 'X_cut': X_cut}

    def __len__(self):
        return len(self.pSeq)


def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M', 'N', 'H', 'Z']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq


def main():
    model_expr_path = 'results/model/165_mpra_expr_denselstm.pth'
    dataset_input = DataLoader(dataset=predictDataset(path='../data/ecoli_mpra_expr_test.csv', isGpu=True), batch_size=256,
                              shuffle=False)
    model_expr = torch.load(model_expr_path)
    pSeqList = []
    exprList = []
    for k, inputLoader in enumerate(dataset_input):
        expr = model_expr(inputLoader['X'])
        seqs = inputLoader['X'].detach()
        seqs = seqs.cpu().float().numpy()
        expr = expr.detach()
        expr = expr.cpu().float().numpy()
        for i in range(np.size(expr)):
            tempSeq = seqs[i, :, :]
            pSeqList.append(decode_oneHot(tempSeq))
            exprList.append(2**(expr[i]))
    predictResults = collections.OrderedDict()
    predictResults['seq'] = pSeqList
    predictResults['expr'] = exprList
    predictResults = pd.DataFrame(predictResults)
    predictResults.to_csv('../data/ecoli_mpra_expr_test_predict_densenet.csv', index=False)


if __name__ == '__main__':
    main()
