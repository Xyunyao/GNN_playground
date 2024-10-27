from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TopKPooling

class molecule_GNN(torch.nn.Module):
  def __init__(self, hidden_channels):
    super(molecule_GNN, self).__init__()
    torch.manual_seed(12345)
    self.conv1=GATConv(dataset.num_node_features, hidden_channels, heads=3)
    self.batch_norm1=torch.nn.BatchNorm1d(hidden_channels*3)
    self.pool1 = TopKPooling (hidden_channels*3, ratio=0.8)
    self.conv2=GATConv(hidden_channels*3, hidden_channels, heads=3
    self.conv2=GATConv(hidden_channels*3, hidden_channels, heads=3)
    self.pool2=global_mean_pool
    self.conv3=GATConv(hidden_channels*3, hidden_channels, heads=3)
    self.pool3=global_mean_pool
    self.lin1=torch.nn.Linear(hidden_channels*3, dataset.num_classes)

                       
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x=self.conv1(x,edge_index)
    x=self.batch_norm1(x)
    x=F.relu(x)
    x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
    x1=global_mean_pool(x, batch)