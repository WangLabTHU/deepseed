# -*- coding: utf-8 -*-
"""
Created on March 30 9:32 2021

@author: Hanwen Xu

E-mail: xuhw20@mails.tsinghua.edu.cn

Integration of all the SeqRegression Model used by Wang Lab.
"""
from torch import nn
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader


def Seq2Scalar(input_nc,
               norm_layer=nn.BatchNorm1d,
               use_dropout=True,
               n_blocks=3,
               padding_type='reflect',
               mode='Hanwen',
               seqL=100):
    """Construct a Resnet-based generator
    Parameters:
        input_nc (int)      -- the number of channels in input seq
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        mode (str)          -- the model you wanna use: Hanwen | WangYe | GPRO
    """
    print('The model type is : ')
    print(mode)
    if mode == 'Hanwen':
        return HanwenModel(input_nc, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, padding_type=padding_type, seqL=seqL)
    elif mode == 'deepinfomax':
        return deepinfomax()
    elif mode == 'deepgoplus':
        return deepGoPlusModel(input_nc, norm_layer=norm_layer, use_dropout=use_dropout,
                           n_blocks=n_blocks, padding_type=padding_type, seqL=seqL)
    elif mode == 'WangYe':
        return WangYeModel(input_nc, norm_layer=norm_layer, use_dropout=use_dropout,
                           n_blocks=n_blocks, padding_type=padding_type, seqL=seqL)
    elif mode == 'GPRO':
        return GPROModel(input_nc, norm_layer=norm_layer, use_dropout=use_dropout,
                           n_blocks=n_blocks, padding_type=padding_type, seqL=seqL)
    elif mode == 'densenet':
        return DenseNet(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0.2, input_length=seqL)
    elif mode == 'denselstm':
        return DenseLSTM(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0.2, input_length=seqL)
    elif mode == 'denseconnectedlstm':
        return DenseConnectedLSTM(input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0.2, input_length=seqL)


class HanwenModel(nn.Module):

    def __init__(self,
                 input_nc,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True, n_blocks=3, padding_type='reflect', seqL = 100):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input seq
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(HanwenModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        ngf = 32
        self.conv1 = nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
        model = [nn.ReflectionPad1d(3),
                 self.conv1,
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        seqT = seqL
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            seqT = np.floor((seqT +2 - 3)/2 + 1)
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            seqT = np.floor((seqT - 1) * 2 - 2 + 2 + 2)
        model += [nn.ReflectionPad1d(3)]
        model += [nn.Conv1d(ngf, input_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model1 = nn.Sequential(*model)
        self.layer3 = nn.Sequential(
            nn.Conv1d(4, 100, kernel_size=7, padding=3),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(100, 100, kernel_size=3, padding=1),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(100 * int(np.floor(np.floor(seqT/2)/2)), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, inputSeq):
        inputSeq = self.model1(inputSeq)
        x = self.layer3(inputSeq)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        output = output.squeeze(-1)
        return output


class deepGoPlusModel(nn.Module):

    def __init__(self,
                 input_nc,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True, n_blocks=3, padding_type='reflect', seqL = 100, max_kernels=59, in_nc=512, hidden_dense=0):
        """

        :param input_nc:
        :param in_nc:
        :param max_kernels:
        :param dense_num:
        :param seqL:
        """
        super(deepGoPlusModel, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(5, max_kernels, 2)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec("self.conv1d_{} = nn.Conv1d(in_channels=input_nc, out_channels=in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(i))
            exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seqL - kernels[i] + 1, stride=1)".format(i))
        self.fc = []
        for i in range(hidden_dense):
            self.fc.append(nn.Linear(len(kernels)*in_nc, len(kernels)*in_nc))
        self.fc.append(nn.Linear(len(kernels)*in_nc, 1))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x_list = []
        for i in range(self.kernel_num):
            exec("x_i = self.conv1d_{}(x)".format(i))
            exec("x_i = self.pool1d_{}(x_i)".format(i))
            if x.size(0) > 1:
                exec("x_list.append(torch.squeeze(x_i))")
            else:
                exec("x_list.append(torch.squeeze(x_i).reshape([1, -1]))")
        x1 = torch.cat(tuple(x_list), dim=1)
        x2 = self.fc(x1)
        output = x2.squeeze(-1)
        return output


class WangYeModel(nn.Module):
    def __init__(self,
                 input_nc,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True,
                 n_blocks=3,
                 padding_type='reflect',
                 seqL=100):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input seq
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            seqL (int)          -- the length of your input
        """
        assert (n_blocks >= 0)
        super(WangYeModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        self.nLayers = 8
        model = [nn.ReflectionPad1d((3, 2)),
                 nn.Conv1d(input_nc, 128, kernel_size=6, padding=0),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        seqT = np.floor(seqL/2)
        model += [nn.Conv1d(128, 256, kernel_size=3, padding=1),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        seqT = np.floor(seqT / 2)
        for i in range(self.nLayers):
            model += [ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias, kernel_size=3),
                      nn.ReLU(inplace=True),]
        model +=[nn.Conv1d(256, 512, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv1d(512, 512, kernel_size=3, padding=1),
                 nn.ReLU(inplace=True),
                 nn.Conv1d(512, 512, kernel_size=3, padding=1)]
        self.model = nn.Sequential(*model)
        self.Linear = nn.Sequential(nn.Linear(int(seqT)*512, 512),
                                     nn.Linear(512, 1))

    def forward(self, inputSeq):
        x = self.model(inputSeq)
        x = x.view(x.size(0), -1)
        output = self.Linear(x)
        output = nn.ReLU(inplace=True)(output.squeeze(-1))
        return output


class Encoder(nn.Module):
    def __init__(self, seqL=75):
        super().__init__()
        self.c0 = nn.Conv1d(4, 64, kernel_size=5, stride=1, padding=2)
        self.c1 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.c2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.c3 = nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=2)
        self.l1 = nn.Linear(512*seqL, 64)

        self.b1 = nn.BatchNorm1d(128)
        self.b2 = nn.BatchNorm1d(256)
        self.b3 = nn.BatchNorm1d(512)

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features


class GlobalDiscriminator(nn.Module):
    def __init__(self, seqL=75, y_size=64):
        super().__init__()
        self.c0 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.c1 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.l0 = nn.Linear(32 * seqL + y_size, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv1d(192, 512, kernel_size=1)
        self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        self.c2 = nn.Conv1d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1, seqL=100):
        super().__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.prior_d = PriorDiscriminator()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seqL = 100

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1)
        y_exp = y_exp.expand(-1, y.size(1), M.size(2))

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma

        return LOCAL + GLOBAL + PRIOR


class deepinfomax(nn.Module):
    def __init__(self,
                 path=r"/disk1/hwxu/deepinfomax/models\run_wy/encoder1160.wgt",
                 in_features=64):
        super(deepinfomax, self).__init__()
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(Path(path)))
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = [nn.Linear(in_features, 8),
                   nn.ReLU(inplace=True),
                   nn.Linear(8, 1)]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.fc(x1[0])
        return nn.ReLU(inplace=True)(x2.squeeze(-1))


class GPROModel(nn.Module):
    def __init__(self,
                 input_nc,
                 norm_layer=nn.BatchNorm1d,
                 use_dropout=True,
                 n_blocks=3,
                 padding_type='reflect',
                 seqL=100):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input seq
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            seqL (int)          -- the length of your input
        """
        assert (n_blocks >= 0)
        super(GPROModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d
        model = [nn.ReflectionPad1d((2, 2)),
                 nn.Conv1d(input_nc, 128, kernel_size=5, padding=0),
                 nn.BatchNorm1d(128),
                 nn.ReLU(inplace=True),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        model += [nn.ReflectionPad1d((2, 2)),
                 nn.Conv1d(128, 256, kernel_size=5, padding=0),
                 nn.BatchNorm1d(256),
                 nn.ReLU(inplace=True),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        model += [nn.ReflectionPad1d((2, 2)),
                 nn.Conv1d(256, 512, kernel_size=5, padding=0),
                  nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.MaxPool1d(kernel_size=2, stride=2)]
        seqT = np.floor(seqL/8)
        self.model = nn.Sequential(*model)
        self.Linear = nn.Sequential(nn.Linear(int(seqT)*512, 1))

    def forward(self, inputSeq):
        x = self.model(inputSeq)
        x = x.view(x.size(0), -1)
        output = self.Linear(x)
        output = output.squeeze(-1)
        return output


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=3):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=kernel_size)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size=3):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))


class DenseLayerLSTM(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayerLSTM, self).__init__()
        #self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        #self.add_module('relu1', nn.ReLU(inplace=True)),
        self.model1 = nn.Sequential(OrderedDict([]))
        self.model1.add_module('lstm1', nn.LSTM(input_size=num_input_features, hidden_size=bn_size*growth_rate,
                                                       num_layers=3, bias=True, batch_first=True, bidirectional=True))
        #self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        #self.add_module('relu2', nn.ReLU(inplace=True)),
        self.model2 = nn.Sequential(OrderedDict([]))
        self.model2.add_module('lstm2', nn.LSTM(input_size=2 * bn_size * growth_rate, hidden_size=growth_rate,
                                         num_layers=3, bias=True, batch_first=True, bidirectional=True))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features, (h_n, c_n) = self.model1(x)
        new_features, (h_n, c_n) = self.model2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 2)


class DenseBlockLSTM(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlockLSTM, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([]))
        self.num_layers = num_layers
        for i in range(num_layers):
            self.model1.add_module('denselayer%d' % (i + 1), DenseLayerLSTM(num_input_features + 2 * i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        x = self.model1(x)
        return x


class TransitionLSTM(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLSTM, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([]))
        self.model1.add_module('lstm1', nn.LSTM(input_size=num_input_features, hidden_size=num_output_features,
                                         num_layers=1, bias=True, batch_first=True, bidirectional=True))

    def forward(self, x):
        x, (h_n, c_n) = self.model1(x)
        return x


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                length = np.floor((length - 1 - 1) / 2 + 1)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        # Linear layer
        self.ratio = nn.Linear(int(length) * num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool1d(out, kernel_size=7, stride=1, padding=3).view(out.size(0), -1)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out


class DenseLSTM(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=32, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseLSTM, self).__init__()

        # First convolution
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features = nn.Sequential(OrderedDict([]))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        self.lstm = nn.Sequential(OrderedDict([]))
        self.lstm.add_module('lstm_layer', torch.nn.LSTM(input_size=num_init_features, hidden_size=num_init_features,
                                                       num_layers=3, bias=True, batch_first=True, bidirectional=True))
        num_features = 2*num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                length = np.floor((length - 1 - 1) / 2 + 1)

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))

        # Linear layer
        self.ratio = nn.Linear(int(length) * num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features0 = self.features0(x)
        features0 = features0.permute(0, 2, 1)
        features1, (h_n, c_n) = self.lstm(features0)
        features1 = features1.permute(0, 2, 1)
        features1 = self.features(features1)
        out = F.relu(features1, inplace=True)
        out = F.avg_pool1d(out, kernel_size=7, stride=1, padding=3).view(out.size(0), -1)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out


class DenseConnectedLSTM(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_nc=4, growth_rate=64, block_config=(2, 2, 4, 2),
                 num_init_features=64, bn_size=4, drop_rate=0, input_length=100):

        super(DenseConnectedLSTM, self).__init__()

        # First convolution
        self.features0 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(input_nc, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.features = nn.Sequential(OrderedDict([]))
        length = np.floor((input_length + 2 * 1 - 1 - 2)/2 + 1)
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlockLSTM(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + 2 * num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLSTM(num_input_features=num_features, num_output_features=num_features // 4)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = 2 * (num_features // 4)

        # Final batch norm
        self.fb_norm = nn.Sequential(OrderedDict([]))
        self.fb_norm.add_module('norm5', nn.BatchNorm1d(num_features))
        # Linear layer
        self.ratio = nn.Linear(int(length) * num_features, 1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features0 = self.features0(x)
        features0 = features0.permute(0, 2, 1)
        features1 = self.features(features0)
        features1 = features1.permute(0, 2, 1)
        features1 = self.fb_norm(features1)
        out = F.relu(features1, inplace=True)
        out = F.avg_pool1d(out, kernel_size=7, stride=1, padding=3).view(out.size(0), -1)
        out = self.ratio(out)
        out = out.squeeze(-1)
        return out