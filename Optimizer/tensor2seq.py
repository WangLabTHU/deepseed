"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import collections
import pandas as pd
import time


def tensor2seq(input_sequence, label):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_sequence, np.ndarray):
        if isinstance(input_sequence, torch.Tensor):  # get the data from a variable
            sequence_tensor = input_sequence.data
        else:
            return input_sequence
        sequence_numpy = sequence_tensor.cpu().float().numpy()  # convert it into a numpy array
    else:  # if it is a numpy array, do nothing
        sequence_numpy = input_sequence
    return decode_oneHot(sequence_numpy, label)


def reserve_percentage(tensorInput, tensorSeq):
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
    c, n = 0.0, 0.0
    for i in range(len(results['fakeB'])):
        seqA = results['realA'][i]
        seqB = results['fakeB'][i]
        for j in range(len(seqA)):
            if seqA[j] != 'M':
                n += 1
                if seqA[j] == seqB[j]:
                    c += 1
    return 100*c/n


def save_sequence(tensorSeq, tensorInput, tensorRealB, save_path='results/', name='', cut_r=0.1):
    i = 0
    results =collections.OrderedDict()
    results['fakeB'] = []
    results['realA'] = []
    results['realB'] = []
    for seqT in tensorSeq:
        label = 'fakeB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorInput:
        label = 'realA'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for seqT in tensorRealB:
        label = 'realB'
        for j in range(seqT.size(0)):
            seq = tensor2seq(torch.squeeze(seqT[j, :, :]), label)
            results[label].append(seq)
        i = i + 1
    for label in ['realA', 'fakeB', 'realB']:
        results[label] = results[label][0 : int(cut_r * len(results[label]))]
    results = pd.DataFrame(results)
    save_name = save_path + name + time.strftime('%Y-%m-%d-%H-%M-%S_', time.localtime(time.time())) + 'results.csv'
    results.to_csv(save_name, index=False)
    return save_name


def decode_oneHot(seq, label):
    keys = ['A', 'T', 'C', 'G']
    dSeq = ''
    for i in range(np.size(seq, 1)):
        if label == 'realA':
            if np.max(seq[:, i]) != 1:
                dSeq += 'M'
            else:
                pos = np.argmax(seq[:, i])
                dSeq += keys[pos]
        else:
            pos = np.argmax(seq[:, i])
            dSeq += keys[pos]
    return dSeq