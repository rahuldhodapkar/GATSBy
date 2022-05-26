#!/usr/bin/env python
#
# Main Graph Attention scaffolding - utilizing prebuilt layers sourced from
# pytorch-geometric
#
# @author Rahul Dhodapkar
#

################################################################################
## Imports
################################################################################

## general
import csv
import numpy as np
import pandas as pd
import os
import itertools
import random
import pickle

## torch
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from src.model.GATSBY import GATSBY
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T

## scRNAseq
import scanpy
import anndata


################################################################################
## Helper Functions
################################################################################

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#
# Adapted from PetarV-/GAT
#
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def scale(X, x_min, x_max):
    nom = (X - X.min(axis=0)) * (x_max - x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return x_min + nom / denom


################################################################################
## Load Data
################################################################################

visium_path = './data/visium/human_prostate_adenocarcinoma'
visium_raw = scanpy.read_visium(visium_path)

################################################################################
# Wire the graph
################################################################################

################################################################################
# (1) Create a node for each spot.
#

scanpy.pp.normalize_total(visium_raw)

NUM_VARIABLE_GENES = 200
raw_expression_matrix = torch.tensor(np.log1p(visium_raw.X.todense()), dtype=torch.float)
nonzero_genes = raw_expression_matrix.numpy().sum(axis=0) > 0
#scaled_expression_matrix = (raw_expression_matrix[:, nonzero_genes]
#                            / raw_expression_matrix[:, nonzero_genes].numpy().sum(axis=0))
#scaled_gene_names = visium_raw.var_names[nonzero_genes]
scaled_expression_matrix = raw_expression_matrix
scaled_gene_names = visium_raw.var_names


expression_matrix_highvar = scaled_expression_matrix[:, np.argsort(np.argsort(
    -1 * np.var(scaled_expression_matrix.numpy(), axis=0))) < NUM_VARIABLE_GENES]
highvar_gene_names = scaled_gene_names[np.argsort(np.argsort(
    -1 * np.var(scaled_expression_matrix.numpy(), axis=0))) < NUM_VARIABLE_GENES]

expression_matrix_highvar_censored = expression_matrix_highvar.clone()
n_row, n_col = expression_matrix_highvar_censored.shape
coords_to_censor = [(int(x / n_col), x % n_col)
                    for x in random.sample(range(n_row * n_col), k=int(n_row * n_col * 0.1))]
#identity_coords_to_train = [(int(x / n_col), x % n_col)
#                            for x in random.sample(range(n_row * n_col), k=int(n_row * n_col * 0.1))]

for (r, c) in coords_to_censor:
    expression_matrix_highvar_censored[r, c] = 0

#train_ixs = tuple(zip(* (coords_to_censor[1:int(len(coords_to_censor) / 2)] + identity_coords_to_train) ))
train_ixs = tuple(zip(* (coords_to_censor[1:int(len(coords_to_censor) / 2)])))
test_ixs = tuple(zip(*coords_to_censor[int(len(coords_to_censor) / 2):]))

################################################################################
# (2) Create edges between vertices corresponding to proximal spots, as
#     well as loop edges for the same spot.
#
# NOTE that this wiring must take into account the spatial arrangement of the
#      spots.  For example, Visium uses a honeycomb or "orange-packing" spot
#      geometry, which yields 6 surrounding spots for each central spot.
#      Other technologies, e.g. DBIT-seq may create a more traditional grid
#      amenable to standard von Neumann neighborhood assignment.
#

def prune_invalid_visium_coordinates(coords):
    N_ROWS = 78
    N_COLS = 128
    coords_pruned = []
    for (r, c) in coords:
        if r < 0 or r >= N_ROWS:
            continue
        if c < 0 or c >= N_COLS:
            continue
        coords_pruned += [(r, c)]
    return (coords_pruned)


# Based on published specificiations from 10x on the Visium data format:
# (https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/images)
#
# NOTE that the constant values are defined from this specification, which may
#      be subject to change as the protocol is updated. All "even" indexed rows
#      (e.g. 0, 2, ... etc) are shifted to the left, as below:
#
#                   (r0,c0)  (r0, c1)  (r0, c2)
#                       (r1, c0)  (r1, c1)  (r1, c2)
#                   (r2,c0)  (r2, c1)  (r2, c2)
#
# RETURN a list of tuples corresponding to the array coordinates of all
#        neighbors to the spot provided.
#
def get_visium_neighborhood(array_row, array_col):
    neighbors = [(array_row, array_col)]
    neighbors += [(array_row, array_col - 1), (array_row, array_col + 1)]
    neighbors += (
        [(array_row - 1, i) for i in [array_col - 1, array_col]]
        + [(array_row + 1, i) for i in [array_col - 1, array_col]]
        if array_row % 2 == 0 else
        [(array_row - 1, i) for i in [array_col, array_col + 1]]
        + [(array_row + 1, i) for i in [array_col, array_col + 1]]
    )
    return (prune_invalid_visium_coordinates(neighbors))


coords2spot = {}
for i in range(visium_raw.obs.shape[0]):
    coords2spot[(visium_raw.obs.array_row[i], visium_raw.obs.array_col[i])] = i

adjacent_spots = []
for i in range(visium_raw.obs.shape[0]):
    neighborhood_spots = [
        coords2spot[c] for c in get_visium_neighborhood(
            visium_raw.obs.array_row[i]
            , visium_raw.obs.array_col[i])
        if c in coords2spot
    ]
    adjacent_spots += [(i, j) for j in neighborhood_spots]

edge_index = torch.tensor(np.array(adjacent_spots), dtype=torch.long)

################################################################################
# (3) Train graph attention model
#


data = Data(x=expression_matrix_highvar_censored, y=1,
            edge_index=edge_index.t().contiguous())
'''
transform = T.RandomNodeSplit(
        num_train_per_class = 20,
        num_val = 500,
        num_test = 1000
    )
data = transform(data)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATSBY(data).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# train model
random.seed(42)

epoch_ix = []
train_losses = []
train_cos = []
test_losses = []
test_cos = []

for epoch in range(100):
    model.train()
    # prepare data
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data)
    loss = torch.sqrt(F.mse_loss(out[train_ixs], expression_matrix_highvar[train_ixs]))
    cos_sim_train = cosine_sim(out[train_ixs].detach().numpy(), expression_matrix_highvar[train_ixs].detach().numpy())
    loss.backward()
    optimizer.step()
    # eval
    model.eval()
    pred = model(data)
    test_loss = torch.sqrt(F.mse_loss(out[test_ixs], expression_matrix_highvar[test_ixs]))
    cos_sim_test = cosine_sim(out[test_ixs].detach().numpy(), expression_matrix_highvar[test_ixs].detach().numpy())
    print("Epoch {:05d} | Train MSE Loss {:.4f}; CosineSim {:.4f} | Test MSE Loss {:.4f}; CosineSim {:.4f} ".format(
        epoch
        , loss.item(), cos_sim_train
        , test_loss.item(), cos_sim_test))
    epoch_ix.append(epoch)
    train_losses.append(loss.item())
    train_cos.append(cos_sim_train)
    test_losses.append(test_loss.item())
    test_cos.append(cos_sim_test)

loss_curves = pd.DataFrame({
    'epoch': epoch_ix
    , 'train_loss': train_losses
    , 'train_cos_sim': train_cos
    , 'test_loss': test_losses
    , 'test_cos_sim': test_cos
})
loss_curves.to_csv('./calc/graph_attention/loss_curves.csv')

# save attention data
importance_df = pd.DataFrame({
    'i': model.attention2[0][0, :].detach().numpy()
    , 'j': model.attention2[0][1, :].detach().numpy()
    , 'v': np.transpose(model.attention2[1].detach().numpy())[0]
})

original_edge_df = pd.DataFrame({
    'i': [i for (i, j) in adjacent_spots]
    , 'j': [j for (i, j) in adjacent_spots]
    , 'v': itertools.repeat(1, len(adjacent_spots))
})

os.makedirs('./calc/graph_attention', exist_ok=True)
importance_df.to_csv('./calc/graph_attention/importance_df.csv')
original_edge_df.to_csv('./calc/graph_attention/original_edge_df.csv')

torch.save(model, './calc/graph_attention/model.pickle')

X = model.latent_embedding2.detach().numpy()
np.save('./calc/graph_attention/embeddings.npy', X)

print("All done!")
