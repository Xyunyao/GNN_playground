from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TopKPooling
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, ModuleList
import torch
torch.manual_seed(12345)

class Molecule_GNN(torch.nn.Module):
  def __init__(self, model_params):
    super(Molecule_GNN, self).__init__()
    self.node_embedding =model_params['node_embedding']
    self.edge_embedding=model_params['edge_embedding']
    self.num_classes=model_params['num_classes']
    self.num_layers=model_params['num_layers']
    self.hidden_channels=model_params['hidden_channels']
    self.dropout=model_params['dropout']
    self.heads=model_params['heads']

    #preprocesing layer
    self.lin1=torch.nn.Linear(self.node_embedding, 2*self.hidden_channels)
    self.lin2=torch.nn.Linear(2*self.hidden_channels, self.hidden_channels)
    
    self.conv_layers=ModuleList()
    self.batch_norms=ModuleList()
    #self.pool_layers=ModuleList()

    
    self.conv1=GATConv(self.hidden_channels, self.hidden_channels, heads=3)
    self.batch_norm1=torch.nn.BatchNorm1d(self.hidden_channels*3)

    
    for i in range (self.num_layers-1):
      self.conv_layers.append(GATConv(self.hidden_channels*3, self.hidden_channels, heads=3))
      self.batch_norms.append(torch.nn.BatchNorm1d(self.hidden_channels*3))

    self.pool1=global_mean_pool
    self.lin3=torch.nn.Linear(self.hidden_channels*3, self.hidden_channels)
    self.lin4=torch.nn.Linear(self.hidden_channels, self.num_classes)

                       
  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    x=ReLU(self.lin1(x))
    x=ReLU(self.lin2(x))
    x=F.dropout(x, p=self.dropout, training=self.training)
    x=self.conv1(x,edge_index)
    x=self.batch_norm1(x)
    for i in range(self.num_layers-1):
      x=self.conv_layers[i](x,edge_index)
      x=self.batch_norms[i](x)
    x=self.pool1(x,batch)
    x=ReLU(self.lin3(x))
    x=F.dropout(x, p=self.dropout, training=self.training)
    x=self.lin4(x)

    return F.log_softmax(x, dim=-1)