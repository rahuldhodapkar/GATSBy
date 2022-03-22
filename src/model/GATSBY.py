#!/usr/bin/env python
#
# Core GAT model for GATSBY
#
# @author Rahul Dhodapkar
#

import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv

class GATSBY(torch.nn.Module):
    def __init__(self, dataset):
        super(GATSBY, self).__init__()

        self.hid = 8 # ***NOTE*** need to tune this param
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(
            in_channels = dataset.num_features,
            out_channels = self.hid,
            heads=self.in_head,
            dropout=0.6)
        self.conv2 = GATConv(
            in_channels = self.hid*self.in_head,
            out_channels = dataset.num_features,
            heads=self.out_head,
            dropout=0.6,
            concat=False)
        self.attention = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x, self.attention = self.conv2(x, edge_index,
            return_attention_weights=True)

        return x
