
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv

class GATSBY(torch.nn.Module):
    def __init__(self, dataset):
        super(GATSBY, self).__init__()

        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv1.__save_att__ = True
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_features, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.conv2.__save_att__ = True

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

