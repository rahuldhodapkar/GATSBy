#!/usr/bin/env python
## gene_attention_synthetic.py
#
# Implement basic scaffolding for "gene-level attention" message passing.
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import torch
import torch_geometric as pyg

from torch_geometric.data import Data

from src.model.GATSBY import GATSBY

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.model.GATSBYGene import GATSBYGene

import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.decomposition import PCA


def embed_genes(expression_matrix, gene_embed_dim, expr_embed_dim):
    pca = PCA(n_components=gene_embed_dim)
    gene_embedding = torch.from_numpy(pca.fit_transform(expression_matrix.numpy().transpose()))
    reshaped_data = torch.reshape(expression_matrix, (expression_matrix.shape[0] * expression_matrix.shape[1], 1))
    genes_with_embedding = torch.cat([
        gene_embedding.repeat(expression_matrix.shape[0], 1),
        reshaped_data.repeat(1, expr_embed_dim)], dim=1)
    embedded_per_spot = torch.reshape(genes_with_embedding,
                                      (expression_matrix.shape[0],
                                       expression_matrix.shape[1] * (gene_embed_dim + expr_embed_dim)))
    return embedded_per_spot



edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
'''
x = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                  [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
                  [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]], dtype=torch.float)
'''
expression_matrix = torch.tensor([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9]], dtype=torch.float)

preprocessed = embed_genes(expression_matrix=expression_matrix, gene_embed_dim=2, expr_embed_dim=4)

data = Data(x=preprocessed, edge_index=edge_index.t().contiguous(), y = 1)
transform = T.RandomNodeSplit(
        num_train_per_class = 20,
        num_val = 500,
        num_test = 1000
    )
data = transform(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATSBYGene(expression_matrix=expression_matrix,
                   num_heads=2,
                   embed_dim=6,
                   input_dim=6).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Loss {:.4f}".format(
        epoch, loss.item()))

print('All done!')
