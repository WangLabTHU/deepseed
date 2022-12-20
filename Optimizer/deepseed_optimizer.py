import optimizer_module
from SeqRegressionModel import *
from wgan_attn import *
import torchvision.transforms as transforms
import collections
import pandas as pd


def main():
    f = open("/home/hwxu/promoterPolisher/ecoli_mpra_3_laco/experiment_2022_03_22/input_promoters.txt")
    lines = f.readlines()
    polish_seq, original_seq, seq_name = [], [], collections.OrderedDict()
    size_prop = 5*1024
    max_iter = 100
    gen_num = 5
    similarity_penalty = 0.85
    prob_mut = 0.005
    for line in lines:
        #if len(control_seq) == 1: break
        if '>' not in line:
            if 'M' in line:
                polish_seq.append(line.strip())
                seq_name[line.strip()] = names
            else:
                original_seq.append(line.strip())
        else:
            names = (line.split('>')[1].strip())
    predictor_path = "../Predictor/results/model/165_mpra_expr_denselstm.pth"
    generator_path = "../Generator/check_points/ecoli_mpra_3_laco_net_G_8899.pth"
    op = optimizer_module.optimizer_fix_flank(predictor_path=predictor_path,
                                             generator_path=generator_path,
                                             size_pop=size_prop,
                                             max_iter=max_iter,
                                             gen_num=gen_num,
                                             similarity_penalty=similarity_penalty,
                                             prob_mut=prob_mut)
    print('{} {}'.format(len(polish_seq), len(original_seq)))
    op.set_input(polish_seq, original_seq)
    print('|***************************|')
    print(' Polishing Process Start!')
    print('|***************************|')
    op.optimization()
    opt_save_path = 'results/ecoli_3_laco.txt'
    history_save_path = 'results/ecoli_3_laco.csv'

    with open(opt_save_path, "w") as f:
        f.write('predictor path: {} \n'.format(predictor_path))
        f.write('generator path: {} \n'.format(generator_path))
        f.write('size population: {}, max iter: {}, generation number: {}, similarity penalty: {}, probability of mutation: {}\n'.format(size_prop, max_iter, gen_num, similarity_penalty, prob_mut))
        i = 0
        for seq in op.seqs_string:
            f.write('Input sequences: {}\n'.format(seq))
            control_seq = op.control_results[seq]
            seq_control_eval = transforms.ToTensor()(optimizer_module.one_hot(control_seq)).float()
            if op.is_gpu:
                seq_control_eval = seq_control_eval.cuda()
            expression_eval = op.predictor(seq_control_eval)
            f.write(seq_name[seq] + '\n')
            f.write('control: {} predict_expression:{}\n'.format(control_seq, 2 ** expression_eval.item()))
            for j in range(op.gen_num):
                f.write('case: {} optimize expression: {}\n'.format(op.seq_results[seq][j], 2 ** op.expr_results[seq][j]))
            i += 1
    f.close()
    history_data = pd.DataFrame(op.seq_opt_history)
    history_data.to_csv(history_save_path, index=False)
    print('|***************************|')
    print(' Polishing Process Finished!')
    print('|***************************|')



if __name__ == '__main__':
    main()