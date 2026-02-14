import pandas as pd
from gensim.models import Word2Vec
from word2vec import seq_to_kmers, get_protein_embedding
import numpy as np



model = Word2Vec.load("word2vec_30.model")
datas = pd.read_csv('../data/AB645-no-wt.csv',usecols=['PDB','antibody_light_seq','antibody_heavy_seq','antigen_seq',
                                                      'antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_seq_mut','ddG'])

name = datas['PDB'].values.tolist()

ab_l_seqs = datas['antibody_light_seq'].values.tolist()
ab_h_seqs = datas['antibody_heavy_seq'].values.tolist()
ab_seqs = []
for idx in range(len(ab_l_seqs)):
    if ab_h_seqs[idx] != ab_h_seqs[idx]:
        ab_h_seqs[idx] = ''
    ab_seq = ab_l_seqs[idx] + ab_h_seqs[idx]
    ab_seqs.append(ab_seq)
ag_seqs = datas['antigen_seq'].values.tolist()
ag = []
for ag_seq in ag_seqs:
    if ',' in ag_seq:
        ag_seq = ''.join(ag_seq.split(','))
    ag.append(ag_seq)
ag_seqs = ag

ab_l_seqs_mut = datas['antibody_light_seq_mut'].values.tolist()
ab_h_seqs_mut = datas['antibody_heavy_seq_mut'].values.tolist()
ab_seqs_mut = []
for idx in range(len(ab_l_seqs_mut)):
    if ab_h_seqs_mut[idx] != ab_h_seqs_mut[idx]:
        ab_h_seqs_mut[idx] = ''
    ab_seq_mut = ab_l_seqs_mut[idx] + ab_h_seqs_mut[idx]
    ab_seqs_mut.append(ab_seq_mut)
ag_seqs_mut = datas['antigen_seq_mut'].values.tolist()
ag_mut = []
for ag_seq_mut in ag_seqs_mut:
    if ',' in ag_seq_mut:
        ag_seq_mut = ''.join(ag_seq_mut.split(','))
    ag_mut.append(ag_seq_mut)
ag_seqs_mut = ag_mut


def seq2vec(seqset,name):
    proteins = []
    for sequence in seqset:
        protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
        proteins.append(protein_embedding)
    proteins = np.array(proteins)
    npyfile = np.save('../data/seq2vec/{}'.format(name), proteins)
    return npyfile


seq2vec(ab_seqs,'ab.npy')
seq2vec(ag_seqs,'ag.npy')
seq2vec(ab_seqs_mut,'ab_mut.npy')
seq2vec(ag_seqs_mut,'ag_mut.npy')
Y = datas['ddG'].values.tolist()
np.save('../data/Y.npy',Y)