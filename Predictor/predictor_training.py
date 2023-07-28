import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from expression_dataset import SeqDataset
from SeqRegressionModel import Seq2Scalar
from matplotlib import pyplot as plt
import numpy as np
import collections
import pandas as pd
from utils import EarlyStopping
from tqdm import tqdm


class Seq2ScalarTraining:
    def __init__(self):
        self.batch_size = 128
        self.lr = 0.001
        self.lr_expr = 0.005
        self.gpu = True
        self.patience = 10
        self.epoch = 100
        self.seqL = 165
        self.mode = 'denselstm'
        self.name = 'expr_' + self.mode
        self.symb = '165_mpra'
        self.dataset_train = DataLoader(dataset=SeqDataset(path='../data/ecoli_mpra_expr.csv', isTrain=True, isGpu=self.gpu), batch_size=self.batch_size, shuffle=True)
        self.dataset_valid = DataLoader(dataset=SeqDataset(path='../data/ecoli_mpra_expr.csv', isTrain=False, isGpu=self.gpu), batch_size=self.batch_size, shuffle=False)
        self.dataset_test = DataLoader(dataset=SeqDataset(path='../data/ecoli_mpra_expr_test.csv', isTrain=True, isGpu=self.gpu, split_r=1.0), batch_size=self.batch_size, shuffle=False)
        self.model_ratio = Seq2Scalar(input_nc=4, seqL=self.seqL, mode=self.mode)
        self.save_path = 'results/model/'
        if self.gpu:
            self.model_ratio=self.model_ratio.cuda()
        self.loss_y = torch.nn.MSELoss()
        if self.mode == 'deepinfomax':
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.fc.parameters(), lr=self.lr_expr)
        else:
            self.optimizer_ratio = torch.optim.Adam(self.model_ratio.parameters(), lr=self.lr_expr)

    def training(self):
        trainingLog = collections.OrderedDict()
        trainingLog['train_loss'] = []
        trainingLog['test_coefs'] = []
        trainingLog['test_loss'] = []
        early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.save_path + self.symb + '_' + self.name + '.pth', stop_order='max')
        for ei in range(self.epoch):
            train_loss_y = 0
            train_num_y = 0
            test_loss_y = 0
            test_num = 0
            self.model_ratio.train()
            print('Training iters')
            for trainLoader in tqdm(self.dataset_train):
                train_data, train_y = trainLoader['x'], trainLoader['z']
                predict = self.model_ratio(train_data)
                predict_y = torch.squeeze(predict)
                loss_y = self.loss_y(predict_y, train_y)
                self.optimizer_ratio.zero_grad()
                loss_y.backward()
                self.optimizer_ratio.step()
                train_loss_y += loss_y
                train_num_y = train_num_y + 1
            test_predict_expr = []
            test_real_expr = []
            self.model_ratio.eval()
            print('Test iters')
            for testLoader in tqdm(self.dataset_valid):
                test_data, test_y = testLoader['x'], testLoader['z']
                predict_y = self.model_ratio(test_data)
                predict_y = predict_y.detach()
                predict_y2 = predict_y
                predict_y = predict_y.cpu().float().numpy()
                predict_y = predict_y[:]
                real_y = test_y.cpu().float().numpy()
                for i in range(np.size(real_y)):
                    test_real_expr.append(real_y[i])
                    test_predict_expr.append(predict_y[i])
                test_loss_y += self.loss_y(predict_y2, test_y)
                test_num = test_num + 1
            coefs = np.corrcoef(test_real_expr, test_predict_expr)
            coefs = coefs[0, 1]
            test_coefs = coefs
            trainingLog['test_coefs'].append(coefs)
            trainingLog['train_loss'].append(float(train_loss_y)/train_num_y)
            trainingLog['test_loss'].append(float(test_loss_y)/test_num)
            print('epoch:{}train_loss y:{} test_loss y:{} test_coefs:{}'.format(ei, train_loss_y/train_num_y, test_loss_y/test_num, coefs))
            early_stopping(val_loss=test_coefs, model=self.model_ratio)
            if early_stopping.early_stop:
                print('Early Stopping......')
                break
        predict_ratio = []
        real_ratio = []
        self.model_ratio = torch.load(self.save_path + self.symb + '_' + self.name + '.pth')
        self.model_ratio.eval()
        for testLoader in self.dataset_test:
            test_data, test_y = testLoader['x'], testLoader['z']
            predict_y = self.model_ratio(test_data)
            predict_y = predict_y.detach()
            predict_y = predict_y.cpu().float().numpy()
            real_y = test_y.cpu().float().numpy()
            for i in range(np.size(real_y)):
                real_ratio.append(real_y[i])
                predict_ratio.append(predict_y[i])
        ## scatter
        real_expr = np.asarray(real_ratio)
        predict_expr = np.asarray(predict_ratio)
        plt.scatter(real_expr, predict_expr, alpha=0.5, c='brown')
        coefs = np.corrcoef(real_expr, predict_expr)
        coefs = coefs[0, 1]
        plt.title('pearson coefficient:{:.2f}'.format(coefs))
        plt.xlabel('real expression (Log2)')
        plt.ylabel('predict expression (Log2)')
        plt.savefig('results/scatter_fig/scatter_' + self.symb + '_' + self.name + '.png')
        trainingLog = pd.DataFrame(trainingLog)
        trainingLog.to_csv('results/training_log' + self.symb + '_' + self.name + '.csv', index=False)


def main():
    analysis = Seq2ScalarTraining()
    analysis.training()

if __name__ == '__main__':
    main()
