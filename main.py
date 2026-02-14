# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/9/17
@author: Ruofan Jin
@Filename: main.py
@Software: PyCharm
"""
import pandas as pd
import torch
from numpy import *
import numpy as np
import random
import time
from model import *
import timeit
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
#from data import diction,dataset
import sys
sys.path.append("../")
from data.load import split

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True

    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    csv = pd.read_csv('../data/AB1101order.csv',usecols=['PDB','Mutation','antibody_light_seq','antibody_heavy_seq','antigen_a_seq','antigen_b_seq','antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_a_seq_mut','antigen_b_seq_mut', 'ddG'])
    names = csv['PDB'].tolist()
    abls = csv['antibody_light_seq'].tolist()
    abhs = csv['antibody_heavy_seq'].tolist()
    agas = csv['antigen_a_seq'].tolist()
    agbs = csv['antigen_b_seq'].tolist()
    abls_m = csv['antibody_light_seq_mut'].tolist()
    abhs_m = csv['antibody_heavy_seq_mut'].tolist()
    agas_m = csv['antigen_a_seq_mut'].tolist()
    agbs_m = csv['antigen_b_seq_mut'].tolist()
    labels = csv['ddG'].tolist()

    interactions = np.array(labels)
    antibodies = np.load('../data/antibody.npy', allow_pickle=True)
    antigens = np.load('../data/antigen.npy', allow_pickle=True)
    antibodies_mut = np.load('../data/antibody_mut.npy', allow_pickle=True)
    antigens_mut = np.load('../data/antigen_mut.npy', allow_pickle=True)


    """Start training."""
    print('Training...')

    train_index, val_index = split('../data/AB1101order.csv', scale=0.2)

    """ create model ,trainer and tester """
    antibody_dim = 20
    # protein_dim = 100
    antigen_dim = 20
    # atom_dim = 34
    hid_dim = 256
    n_heads = 8
    pf_dim = 64
    dropout = 0.1

    # modified
    n_layers = 3  # origin:3
    batch = 16  # origin:8
    lr = 0.0001  # origin:0.0001
    iteration = 10 # origin:150

    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1
    kernel_size = 7  # 7
    minloss = 1000
    best_pearson = -1000
    best_r2 = -1000

    encoder = Encoder(antibody_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(antigen_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)

    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    with open('../1101analysis/split.txt','w') as file:
        file.write(str(train_index))
        file.write('\n')
        file.write(str(val_index))
        file.write('\n')
    file.close()

    antigens_train, antigens_val = np.array(antigens)[train_index], np.array(antigens)[val_index]  # ag
    antibodies_train, antibodies_val = np.array(antibodies)[train_index], np.array(antibodies)[val_index]  # ab
    antigens_mut_train, antigens_mut_val = np.array(antigens_mut)[train_index], np.array(antigens_mut)[val_index]  # ag_mut
    antibodies_mut_train, antibodies_mut_val = np.array(antibodies_mut)[train_index], np.array(antibodies_mut)[val_index]  # ab_mut
    interactions_train, interactions_val = np.array(interactions)[train_index], np.array(interactions)[val_index]  # Y

    dataset_train = list(zip(antigens_train, antibodies_train, antigens_mut_train, antibodies_mut_train, interactions_train))
    #dataset_train = torch.utils.data.DataLoader(dataset_train, sampler=torch.utils.data.distributed.DistributedSampler(dataset_train, num_replicas=ngpus_per_node, rank=0))
    dataset_val = list(zip(antigens_val, antibodies_val, antigens_mut_val, antibodies_mut_val, interactions_val))
    #dataset_val = torch.utils.data.DataLoader(dataset_val, sampler=torch.utils.data.distributed.DistributedSampler(dataset_val, num_replicas=ngpus_per_node, rank=0))

    """Output files."""
    file_loss_min_PCCS = '../output1101/loss_min_result/RECORD.txt'
    file_loss_min_model = '../output1101/loss_min_model/model'
    file_best_pcc_PCCS = '../output1101/best_pcc_result/RECORD.txt'
    file_best_pcc_model = '../output1101/best_pcc_model/model'
    file_best_r2_PCCS = '../output1101/best_r2_result/RECORD.txt'
    file_best_r2_model = '../output1101/best_r2_model/model'
    
    PCCS = ('Epoch\tTime(sec)\tLoss_train\tLoss_val\tpearson\tmae\tmse\trmse\tr2')
    print(PCCS)

    with open(file_loss_min_PCCS, 'w') as f:
        f.write(PCCS + '\n')
    with open(file_best_pcc_PCCS, 'w') as f:
        f.write(PCCS + '\n')
    with open(file_best_r2_PCCS, 'w') as f:
        f.write(PCCS + '\n')

    start = timeit.default_timer()
        
    early_stopping = EarlyStopping(patience=7, verbose=True)
    for epoch in range(1, iteration + 1):

        print('Epoch:',epoch)
        loss_train_fold, y_train_true, y_train_predict = trainer.train(dataset_train, device, epoch)  # numpy arrays record for an epoch loss.
        pccs_val, mae_val, mse_val, rmse_val, r2_val, loss_val_fold, y_val_true, y_val_predict = tester.test(dataset_val, epoch)  # pccs_dev && loss_val_fold are for an epoch
        print(pccs_val, mae_val, mse_val, rmse_val, r2_val)
        
        end = timeit.default_timer()
        time = end - start
            
        early_stopping(loss_val_fold.tolist(), model, file_loss_min_model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

        PCCS = [epoch, time, loss_train_fold.tolist(), loss_val_fold.tolist(), pccs_val.tolist(), mae_val, mse_val, rmse_val, r2_val]
        
        if loss_val_fold.tolist() < minloss:
            tester.save_pccs(PCCS, file_loss_min_PCCS)
            tester.save_model(model, file_loss_min_model)
            minloss = loss_val_fold.tolist()
        
        if pccs_val.tolist() > best_pearson:
            tester.save_pccs(PCCS, file_best_pcc_PCCS)
            tester.save_model(model, file_best_pcc_model)
            best_pearson = pccs_val.tolist()
        
        if r2_val > best_r2:
            tester.save_pccs(PCCS, file_best_r2_PCCS)
            tester.save_model(model, file_best_r2_model)
            best_r2 = r2_val

        print('\t'.join(map(str, PCCS)))
            
