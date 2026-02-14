#!/bin/bash
# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/9/17
@author: Ruofan Jin
@Filename: main.py
@Software: PyCharm
"""
import random
import torch
import pandas as pd
from numpy import *
import numpy as np
import os
import sys



class feature(object):
    def __init__(self,seq):
        self.seq = seq
        #self.length = max

    def seq2onehot(self):
        aas = {'X':0,'A':1,'R':2,'N':3,'D':4,'C':5,
               'Q':6,'E':7,'G':8,'H':9,'I':10,
               'L':11,'K':12,'M':13,'F':14,'P':15,
               'S':16,'T':17,'W':18,'Y':19,'V':20}
        seq_onehot = np.zeros((len(self.seq),len(aas)))
        for i, aa in enumerate(self.seq[:]):
            seq_onehot[i, (aas[aa])] = 1
        #seq_onehot = ''.join(seq_onehot)
        seq_onehot = seq_onehot[:,1:]  # except X
        return seq_onehot

    def seq2pssm(self):
        with open('1101aa.fasta', 'w') as f:
            f.write('>name\n')
            f.write(self.seq)
        f.close()
        os.system('{}/blast-2.15.0+/bin/psiblast -query 1101aa.fasta -db {}/blast-2.15.0+/bin/sp -num_iterations 3 -out_ascii_pssm 1101aa.pssm'.format(path,path))
        with open('1101aa.pssm', 'r') as inputpssm:
            count = 0
            pssm_matrix = []
            for eachline in inputpssm:
                count += 1
                if count <= 3:
                    continue
                if not len(eachline.strip()):
                    break
                col = eachline.strip()
                col = col.split(' ')
                col = [x for x in col if x != '']
                col = col[2:22]
                col = [int(x) for x in col]
                oneline = col
                pssm_matrix.append(oneline)
            seq_pssm = np.array(pssm_matrix)
        return seq_pssm

def all_feature(ls):
    all = []
    for s in ls:
        if s==s:
            f = feature(s).seq2pssm()
            all.append(f)
        else:
            all.append(s)  # if s!=s that f is nan
    #all = np.array(all)
    return all



if __name__ == "__main__":
    path = sys.argv[1]

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
    csv = pd.read_csv('{}/data/AB1101order.csv'.format(path),usecols=['PDB','Mutation','antibody_light_seq','antibody_heavy_seq','antigen_a_seq','antigen_b_seq','antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_a_seq_mut','antigen_b_seq_mut', 'ddG'])
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

    antibodies_l = all_feature(abls)
    antibodies_h = all_feature(abhs)
    antigens_a = all_feature(agas)
    antigens_b = all_feature(agbs)
    antibodies_l_mut = all_feature(abls_m)
    antibodies_h_mut = all_feature(abhs_m)
    antigens_a_mut = all_feature(agas_m)
    antigens_b_mut = all_feature(agbs_m)

    antibodies = []  #1101
    antigens = []
    antibodies_mut = []
    antigens_mut = []
    i=0
    for i in range(len(antibodies_l)):
        if isinstance(antibodies_h[i],float):
            antibodies.append(antibodies_l[i])
        else:
            antibodies.append(np.concatenate((antibodies_l[i],antibodies_h[i]),axis=0))
        if isinstance(antigens_b[i],float):
            antigens.append(antigens_a[i])
        else:
            antigens.append(np.concatenate((antigens_a[i],antigens_b[i]),axis=0))
        if isinstance(antibodies_h_mut[i],float):
            antibodies_mut.append(antibodies_l_mut[i])
        else:
            antibodies_mut.append(np.concatenate((antibodies_l_mut[i],antibodies_h_mut[i]),axis=0))
        if isinstance(antigens_b_mut[i],float):
            antigens_mut.append(antigens_a_mut[i])
        else:
            antigens_mut.append(np.concatenate((antigens_a_mut[i],antigens_b_mut[i]),axis=0))
        i+=1
    print('i:',i)
    interactions = np.array(labels)

    np.save('{}/data/antibody.npy'.format(path),antibodies)
    np.save('{}/data/antigen.npy'.format(path),antigens)
    np.save('{}/data/antibody_mut.npy'.format(path),antibodies_mut)
    np.save('{}/data/antigen_mut.npy'.format(path),antigens_mut)
