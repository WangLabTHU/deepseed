import noGAN_module
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


def main():
    f = open('../../experiment_9_10_2021/polisher_input.txt')
    lines = f.readlines()
    polish_seq, control_seq, original_seq, original_expression = [], [], [], []
    for line in lines:
        #if len(control_seq) == 1: break
        if '>' not in line:
            if 'M' in line:
                polish_seq.append(line.strip())
            else:
                control_seq.append(line.strip())
        else:
            if 'original' in line:
                original_seq.append(line.strip())
            else:
                original_expression.append(line.strip())
    predictor_path = "/disk1/hwxu/promoterPolisher/ecoli_mpra_165_inducible/experiment_9_10_2021/165_mpra_expr_denselstm_0.76.pth"
    generator_path = "/disk1/hwxu/promoterPolisher/ecoli_mpra_165_inducible/experiment_9_10_2021/net_G_7899_9_10_21.pth"
    save_path = 'results/noGAN_inducible_mpra.txt'
    op = noGAN_module.Optimizer(predictor_path=predictor_path,
                                save_path=save_path)
    op.set_input(polish_seq, control_seq)
    op.optimization()

    with open(op.save_path, "w") as f:
        i = 0
        for seq in op.seqs_string:
            f.write('{} optimize results:\n'.format(seq))
            control_seq = op.control_results[seq]
            seq_control_eval = transforms.ToTensor()(noGAN_module.one_hot(control_seq)).float()
            if op.is_gpu:
                seq_control_eval = seq_control_eval.cuda()
            expression_eval = op.predictor(seq_control_eval)
            f.write(original_expression[i] + '\n')
            f.write(original_seq[i] + '\n')
            f.write('control case:{} predict_expression:{}\n'.format(control_seq, 2 ** expression_eval.item()))
            for j in range(op.gen_num):
                f.write('{} optimize expression: {}\n'.format(op.seq_results[seq][j], 2 ** op.expr_results[seq][j]))
            i += 1
    f.close()



if __name__ == '__main__':
    main()