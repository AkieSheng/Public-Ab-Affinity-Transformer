import torch
import pandas as pd

a = pd.read_csv(csv, usecols=['PDB','Mutation',
                              'antibody_light_seq','antibody_heavy_seq','antigen_seq',
                              'antibody_light_seq_mut','antibody_heavy_seq_mut','antigen_seq_mut',
                              'ddG'])
print(a)