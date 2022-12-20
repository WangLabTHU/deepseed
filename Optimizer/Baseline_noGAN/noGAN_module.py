#from SeqRegressionModelTest import *
from SeqRegressionModel import *
from wgan_attn import *
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
from torch.utils.data import DataLoader
import collections
import pandas as pd
from sko.GA import GA
from sko.tools import set_run_mode


def edit_distance(word1, word2):
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if word1[i - 1] == word2[j - 1]:
                temp = 0
            else:
                temp = 1
            dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


# 190801
# 根据编辑距离计算相似度
def similarity_func(word1, word2):
    res = edit_distance(word1, word2)
    maxLen = max(len(word1), len(word2))
    return 1-res*1.0/maxLen


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class LoadData(Dataset):

    def __init__(self, data, is_train=True, gpu_ids='0'):
        self.storage = []
        self.gpu_ids = gpu_ids
        for i in range(np.size(data, 0)):
            self.storage.append(data[i])

    def __getitem__(self, item):
        in_seq = transforms.ToTensor()(self.storage[item])
        if len(self.gpu_ids) > 0:
            return in_seq[0, :].float().cuda()
        else:
            return in_seq[0, :].float()

    def __len__(self):
        return len(self.storage)


def one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        encoded[charmap[seq[i]], i] = 1
    return encoded


def backbone_one_hot(seq):
    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded


def decode_oneHot(seq):
    keys = ['A', 'T', 'C', 'G', 'M', 'N', 'H', 'Z']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        pos = np.argmax(seq[:, i])
        dSeq += keys[pos]
    return dSeq


class Optimizer:

    def __init__(self,
                 predictor_path="/home/hwxu/promoterDesigner/Predictor/results/model/165_mpra_expr_denselstm_0.76.pth",
                 epochs=5000,
                 lr=0.01,
                 is_gpu=True,
                 seqL=165,
                 gen_num=3,
                 similarity_penalty=0.9,
                 save_path='results/optimize_results_denselstm_0.76.txt'):
        self.predictor = torch.load(predictor_path)
        self.epochs = epochs
        self.lr = lr
        self.is_gpu = is_gpu
        self.seqL = seqL
        self.gen_num = gen_num
        self.save_path = save_path
        self.similarity_penalty = similarity_penalty
        for p in self.predictor.parameters():
            p.requires_grad = False
        self.seqs, self.masks, self.randns = [], [], []
        self.best_expr, self.best_seq = 0, ''
        self.seq_results, self.expr_results, self.control_results = collections.OrderedDict(), collections.OrderedDict(), collections.OrderedDict()

    def set_input(self, polish_seqs, control_seqs):
        self.seqs_string = polish_seqs
        self.control_seqs_string = control_seqs
        for i in range(len(polish_seqs)):
            seq_i = polish_seqs[i]
            self.seq_results[seq_i], self.expr_results[seq_i] = [], []
            for j in range(self.gen_num):
                self.seq_results[seq_i].append(control_seqs[i])
                self.expr_results[seq_i].append(0.0)
            self.seq_results[seq_i], self.expr_results[seq_i] = np.array(self.seq_results[seq_i]), np.array(self.expr_results[seq_i])
            mask_i = np.zeros([4, len(seq_i)])
            for j in range(len(seq_i)):
                if seq_i[j] == 'M':
                    mask_i[:, j] = np.ones([4, 1])[:, 0]
            self.seqs.append(backbone_one_hot(polish_seqs[i]))
            self.masks.append(mask_i)
            self.i = 0

    def opt_func(self, p):
        lb_output = -1000
        p_reshape = np.zeros([np.size(p, 0), 4, self.seqL])
        mask_i = self.masks[self.i]
        for i in range(np.size(p, 0)):
            p_onehot = np.zeros([4, self.seqL])
            for j in range(self.seqL):
                p_onehot[int(p[i, j]), j] = 1
            p_reshape[i, :, :] = np.multiply(self.seqs[self.i], 1 - self.masks[self.i]) + np.multiply(mask_i, p_onehot.reshape([4, -1]))
        generateData = DataLoader(LoadData(data=p_reshape), batch_size=1024, shuffle=False)
        predictions = []
        seq_generate = []
        for j, eval_data in enumerate(generateData):
            seq_generate.append(eval_data)
            predictions.append(self.predictor(eval_data).detach())
        seq_generate = torch.cat(seq_generate, dim=0).cpu().float().numpy()
        predictions = torch.cat(predictions, dim=0).cpu().float().numpy()
        for k in range(np.size(predictions, 0)):
            seq_decode_k = decode_oneHot(np.squeeze(seq_generate[k, :, :]).reshape([4, -1]))
            for m_j in range(self.seqL):
                if self.seqs_string[self.i][m_j] != 'M' and self.seqs_string[self.i][m_j] != seq_decode_k[m_j]:
                    predictions[k] = lb_output
                    break
        preList = np.argsort(-predictions)
        seq_max = seq_generate[preList[0]]
        expression_eval = predictions[preList[0]]
        seq_opt = decode_oneHot(np.squeeze(seq_max))

        similarity = 0
        for seq_k_j in self.seq_results[self.seqs_string[self.i]]:
            similarity = max(similarity_func(seq_opt, seq_k_j), similarity)
        if similarity > self.similarity_penalty: expression_eval = lb_output

        if 'GGGGG' in seq_opt or 'CCCCC' in seq_opt: expression_eval = lb_output
        if expression_eval > min(self.expr_results[self.seqs_string[self.i]]):
            if seq_opt not in list(self.seq_results[self.seqs_string[self.i]]):
                self.seq_results[self.seqs_string[self.i]][-1] = seq_opt
                self.expr_results[self.seqs_string[self.i]][-1] = expression_eval
                sort_idx = np.argsort(-self.expr_results[self.seqs_string[self.i]])
                self.seq_results[self.seqs_string[self.i]] = self.seq_results[self.seqs_string[self.i]][sort_idx]
                self.expr_results[self.seqs_string[self.i]] = self.expr_results[self.seqs_string[self.i]][sort_idx]
        if expression_eval > self.best_expr:
            self.best_expr = expression_eval
            self.best_seq = seq_opt
            percentage = collections.OrderedDict()
            for char_i in seq_opt:
                if char_i in percentage.keys():
                    percentage[char_i] += 1
                else:
                    percentage[char_i] = 0
            for char_i in percentage.keys():
                print('{} percentage: {}'.format(char_i, percentage[char_i]/len(seq_opt)))
            print('best seq:{} values:{}'.format(self.best_seq, 2**expression_eval))
        return -predictions

    def optimization(self):
        keys = ['A', 'T', 'C', 'G']
        mode = 'vectorization'
        set_run_mode(self.opt_func, mode)
        for i in range(len(self.seqs)):
            self.best_expr, self.best_seq = 0, ''
            print('Optimize seq {}'.format(i))
            seq_i = self.seqs_string[i]
            self.control_results[seq_i] = self.control_seqs_string[i]
            self.i = i
            lb, ub, precision = [], [], []
            for j in range(self.seqL):
                lb.append(0)
                ub.append(3)
                precision.append(1)
            ga = GA(func=self.opt_func, n_dim=self.seqL, size_pop=3*1024, max_iter=100, prob_mut=0.005, lb=lb, ub=ub,
                    precision=precision)
            ga.run()