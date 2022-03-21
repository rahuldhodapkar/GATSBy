#!/usr/bin/env python
## Main.py
#
# Implement basic graph attentional network (GAT)-like code
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import torch
import torch_geometric as pyg

from torch_geometric.data import Data

from src.model.GATSBY import GATSBY

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch.nn.functional as F

edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[5],
                  [3],
                  [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index.t().contiguous())
data_list = [data]
loader = DataLoader(data_list, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATSBY(loader.dataset[0]).to(device)
data = loader.dataset[0].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

model.train()
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.x)
    loss.backward()
    optimizer.step()
    print("Epoch {:05d} | Loss {:.4f}".format(
        epoch, loss.item()))


##### https://github.com/pyg-team/pytorch_geometric/issues/1227
torch.manual_seed(0)
l = pyg.nn.GATConv(6,6)
x = torch.distributions.normal.Normal(0,1).sample((4,6))
ei = torch.tensor([[1,2,2,0,3,0,1],
                   [0,0,1,2,2,3,3]],dtype=torch.long)

x, alpha = l(x,ei,return_attention_weights=True)

print(alpha)






