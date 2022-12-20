import torch
import torch.nn as nn
from torch.nn import init
from pro_data import LoadData
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tensor2seq import save_sequence, tensor2seq, reserve_percentage
from transfer_fasta import csv2fasta
import utils


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def get_infinite_batches(data_loader):
    while True:
        for i, data in enumerate(data_loader):
            yield data


class ResBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel_size=13, padding=6, bias=True):
        super(ResBlock, self).__init__()
        model = [nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 nn.ReLU(inplace=False),
                 nn.Conv1d(input_nc, output_nc, kernel_size=kernel_size, padding=padding, bias=bias),
                 ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + 0.3*self.model(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, emb_num, num_heads):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(emb_num, num_heads)

    def forward(self, x):
        x1, self.weights = self.attn(x, x, x)
        return x1


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=512, seqL=50, bias=True, layer_num=1, num_heads=16):
        super(Generator, self).__init__()
        self.ngf, self.seqL = ngf, seqL
        self.first_linear = nn.Linear(seqL*input_nc, seqL*ngf, bias=bias)
        self.layer_num = layer_num
        for i in range(self.layer_num):
            exec("self.layer_{} = EncoderLayer(ngf, {})".format(i, num_heads))
        model = [ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 ResBlock(ngf, ngf),
                 nn.Conv1d(ngf, output_nc, 1),
                 nn.Softmax(dim=1), ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.first_linear(x.view(x.size(0), -1)).reshape([-1, self.ngf, self.seqL])
        x_t = x1.permute(2, 0, 1)
        for i in range(self.layer_num):
            exec("x_t = self.layer_{}(x_t)".format(i))
        features0 = x_t.permute(1, 2, 0)
        return self.model(features0)


class Discriminator(nn.Module):

    def __init__(self, output_nc, ndf=512, seqL=50, bias=True, layer_num=1, num_heads=16):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv1d(output_nc, ndf, 1)
        self.layer_num = layer_num
        for i in range(self.layer_num):
            exec("self.layer_{} = EncoderLayer(ndf, {})".format(i, num_heads))
        model = [ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf),
                 ResBlock(ndf, ndf), ]
        self.model = nn.Sequential(*model)
        self.last_linear = nn.Linear(seqL*ndf, 1, bias=bias)

    def forward(self, x):
        x1 = self.Conv1(x)
        x_t = x1.permute(2, 0, 1)
        for i in range(self.layer_num):
            exec("x_t = self.layer_{}(x_t)".format(i))
        x_t = x_t.permute(1, 2, 0)
        x_t = self.model(x_t)
        return self.last_linear(x_t.contiguous().view(x_t.size(0), -1))


class WGAN():

    def __init__(self, input_nc, output_nc, seqL=100, lr=1e-4, gpu_ids='0', l1_w=10):
        super(WGAN, self).__init__()
        self.gpu_ids = gpu_ids
        self.l1_w = l1_w
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.generator = Generator(input_nc, output_nc, seqL=seqL)
        self.discriminator = Discriminator(input_nc + output_nc, seqL=seqL)
        if len(gpu_ids) > 0:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        self.l1_loss = torch.nn.L1Loss()
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    def backward_g(self, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = False
        self.generator.zero_grad()
        self.fake_inputs = self.generator(fake_x)
        pred_fake = self.discriminator(torch.cat((fake_x, self.fake_inputs), 1))
        self.g_loss = -pred_fake.mean()
        self.g_l1 = self.l1_w*self.l1_loss(fake_x, self.fake_inputs)
        self.g_total_loss = self.g_loss + self.g_l1
        self.g_total_loss.backward()
        self.optim_g.step()

    def backward_d(self, real_x, fake_x):
        for p in self.discriminator.parameters():
            p.requires_grad = True
        self.fake_inputs = self.generator(fake_x)
        fakeAB = torch.cat((fake_x, self.fake_inputs), 1)
        realAB = torch.cat((fake_x, real_x), 1)
        self.discriminator.zero_grad()
        pred_fake, pred_real = self.discriminator(fakeAB), self.discriminator(realAB)
        self.d_loss = pred_fake.mean() - pred_real.mean()
        self.gp, gradients = cal_gradient_penalty(self.discriminator, realAB, fakeAB, device=self.device)
        self.d_total_loss = self.d_loss + self.gp
        self.d_total_loss.backward()
        self.optim_d.step()


def main():
    data_name = 'ecoli_100_-10_-35'
    train_data, test_data = DataLoader(LoadData(is_train=True, path='data/{}.csv'.format(data_name), split_r=1),
                                       batch_size=32, shuffle=True), DataLoader(LoadData(is_train=True, path='data/{}.csv'.format(data_name), split_r=1), batch_size=32)
    train_data = get_infinite_batches(train_data)
    logger = utils.get_logger(log_path='cache/attn_-10_-35/training_log/')
    n_critics = 5
    n_iters = 10000
    model = WGAN(input_nc=4, output_nc=4, seqL=100, l1_w=30)
    d_loss, g_loss, g_l1 = 0, 0, 0
    tensorSeq, tensorInput, tensorRealB = [], [], []
    for i in range(n_iters):
        #train discriminators
        for j in range(n_critics):
            _data = train_data.__next__()
            model.backward_d(_data['out'], _data['in'])
            d_loss += float(model.d_loss)
        model.backward_g(_data['in'])
        g_loss += float(model.g_loss)
        g_l1 += float(model.g_l1)
        if i % 100 == 99:
            tensorSeq, tensorInput, tensorRealB = [], [], []
            for j, eval_data in enumerate(test_data):
                with torch.no_grad():
                    tensorSeq.append(model.generator(eval_data['in']))
                    tensorInput.append(eval_data['in'])
                    tensorRealB.append(eval_data['out'])
            logger.info('Training: iters: {}, dloss: {}, gloss:{}, gl1:{}'.format(i, d_loss / 100 / n_critics, g_loss / 100, g_l1 / 100))
            logger.info('Testing: reserve percentage: {}%'.format(reserve_percentage(tensorInput, tensorSeq)))
            d_loss, g_loss, g_l1 = 0, 0, 0
            csv_name = save_sequence(tensorSeq, tensorInput, tensorRealB, save_path='cache/attn_-10_-35/', name='-10_-35_', cut_r=0.1)
            A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref = utils.polyAT_freq(csv_name, 'data/{}.csv'.format(data_name))
            logger.info('polyA valid AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_valid['AAAAA'],
                                                                                 A_dict_valid['AAAAAA'],
                                                                                 A_dict_valid['AAAAAAA'],
                                                                                 A_dict_valid['AAAAAAAA']))
            logger.info('polyA ref AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_ref['AAAAA'],
                                                                                 A_dict_ref['AAAAAA'],
                                                                                 A_dict_ref['AAAAAAA'],
                                                                                 A_dict_ref['AAAAAAAA']))
            logger.info('polyT valid TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_valid['TTTTT'],
                                                                                 T_dict_valid['TTTTTT'],
                                                                                 T_dict_valid['TTTTTTT'],
                                                                                 T_dict_valid['TTTTTTTT']))
            logger.info('polyT ref TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_ref['TTTTT'],
                                                                               T_dict_ref['TTTTTT'],
                                                                               T_dict_ref['TTTTTTT'],
                                                                               T_dict_ref['TTTTTTTT']))
            utils.kmer_frequency(csv_name, 'data/{}.csv'.format(data_name), k=4, save_path='cache/attn_-10_-35/figure/', save_name=str(i))
            csv2fasta(csv_name, 'cache/attn_-10_-35/gen_', 'iter_{}'.format(i))
            torch.save(model.generator, 'cache/attn_-10_-35/' + 'net_G_' + str(i) + '.pth')
            torch.save(model.discriminator, 'cache/attn_-10_-35/' + 'net_D_' + str(i) + '.pth')


if __name__ == '__main__':
    main()
