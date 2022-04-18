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
from tqdm import tqdm
import numpy as np

## torch
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from src.model.GATSBY import GATSBY
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

## scRNAseq
import scanpy
import anndata

################################################################################
## Helper Functions
################################################################################

#
# Adapted from PetarV-/GAT
#
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

################################################################################
## Load Data
################################################################################

visium_path = './data/visium/normal_human_prostate'
visium_raw = scanpy.read_visium(visium_path)

################################################################################
# Wire the graph
################################################################################

################################################################################
# (1) Create a node for each spot x gene pair.
#
# NOTE that we will create a node only for pairs with non-zero expression
#      (e.g. sparsely) and will store these as a tensor in "spot"-major order.
#
#           V(spot_i, gene_j) = V[(i-1)*n_genes + j]
#
coo_expr_data = visium_raw.X.tocoo()
x = torch.tensor(np.transpose(coo_expr_data.data), dtype=torch.float)

################################################################################
# (2) Create edges between vertices corresponding to proximal spots, as
#     well as those corresponding to the same spot.
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
    return(coords_pruned)

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
            [(array_row - 1, i) for i in [array_col-1,array_col]]
            + [(array_row + 1, i) for i in [array_col-1,array_col]]
        if array_row % 2 == 0 else
            [(array_row - 1, i) for i in [array_col,array_col+1]]
            + [(array_row + 1, i) for i in [array_col,array_col+1]]
    )
    return(prune_invalid_visium_coordinates(neighbors))


coords2spot = {}
for i in range(visium_raw.obs.shape[0]):
    coords2spot[(visium_raw.obs.array_row[i], visium_raw.obs.array_col[i])] = i


adjacent_spots = []
for i in range(visium_raw.obs.shape[0]):
    neighborhood_spots =[
        coords2spot[c] for c in get_visium_neighborhood(
                visium_raw.obs.array_row[i]
                ,visium_raw.obs.array_col[i])
        if c in coords2spot
    ]
    adjacent_spots += [(i, j) for j in neighborhood_spots]

edge_index = torch.tensor(np.array(adjacent_spots), dtype=torch.long)

print('ERROR: ***Script not fully functional, terminating***' ,file=sys.stderr)
sys.exit(1)

spot2datarange = {}
for i in tqdm(range(len(coo_expr_data.data))):
    if coo_expr_data.row[i] in spot2datarange:
        spot2datarange[coo_expr_data.row[i]].append(i)
    else:
        spot2datarange[coo_expr_data.row[i]] = [i]


edges = []
for si, sj in tqdm(adjacent_spots):
    for node_in_si in spot2datarange[si]:
        for node_in_sj in spot2datarange[sj]:
            edges.append((node_in_si, node_in_sj))


edge_index = torch.tensor(np.array(edges), dtype=torch.long)


################################################################################
# (3) Train graph attention model
#

data_obj = Data(x=x, edge_index=edge_index.t().contiguous())
data_list = [data_obj]
loader = DataLoader(data_list, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATSBY(loader.dataset[0]).to(device)
data = loader.dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Loss {:.4f}".format(
        epoch, loss.item()))

