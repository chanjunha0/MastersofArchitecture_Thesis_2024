import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv, GATConv, EdgeCNN
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import _spmm, sparse


## Classes

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        '''
        Initializes a model that processes graph-structured data using 
        a simple two-layer GCN architecture.
        
        Args:
            num_node_features: Number of features each node in the input graph has
            num_classes: Number of output classes
        '''
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # First GCN layer
        self.conv2 = GCNConv(16, num_classes)        # Second GCN layer

    def forward(self, x, edge_index):
        '''
        Defines the forward pass of the model.
        
        Args:
            x: Node features of shape [num_nodes, num_node_features]
            edge_index: Graph connectivity in COO format with shape [2, num_edges]
        '''
        x = self.conv1(x, edge_index)    # Apply the first GCN convolution layer
        x = F.relu(x)                    # Apply the ReLU activation function
        x = F.dropout(x, training=self.training) # Apply dropout to reduce overfitting
        x = self.conv2(x, edge_index)    # Apply the second GCN convolution layer
        
        return x  # Model's predictions for each node

class SimpleGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads=8, dropout=0.6):
        '''
        Initializes a model that processes graph-structured data using 
        a simple two-layer GAT architecture.
        
        Args:
            num_node_features: Number of features each node in the input graph has
            num_classes: Number of output classes
            heads: Number of attention heads for the GAT layers
            dropout: Dropout rate applied to the features
        '''
        super(SimpleGAT, self).__init__()
        self.heads = heads
        self.dropout = dropout

        # Define the first GAT layer
        self.conv1 = GATConv(num_node_features, 8, heads=self.heads, dropout=self.dropout)

        # For the second GAT layer, multiply the number of output channels by the number of heads
        # from the first layer because they are concatenated together. Then, reduce to the number
        # of classes as output.
        self.conv2 = GATConv(8 * self.heads, num_classes, heads=1, concat=False, dropout=self.dropout)

    def forward(self, x, edge_index):
        '''
        Defines the forward pass of the model.
        
        Args:
            x: Node features of shape [num_nodes, num_node_features]
            edge_index: Graph connectivity in COO format with shape [2, num_edges]
        '''
        # Apply the first GAT convolution layer with ReLU activation and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the second GAT convolution layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)  # Return log-softmaxed predictions
    
class SimpleEdgeCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleEdgeCNN, self).__init__()
        # Define the neural network (nn) to be used in EdgeConv
        # This nn takes concatenated node features and outputs transformed features
        nn = Sequential(Linear(2 * in_channels, 64),
                        ReLU(),
                        Linear(64, out_channels))
        
        # Initialize the EdgeConv layer with the defined nn
        self.edge_conv = EdgeConv(nn, aggr='max')  # You can change the aggregation type if needed

    def forward(self, x, edge_index):
        # Apply the EdgeConv operation
        x = self.edge_conv(x, edge_index)
        return x
    

# class SimpleLabelPropagation(MessagePassing):
#     """
#     Implements the Label Propagation algorithm for semi-supervised learning
#     on graphs, using the message passing interface.

#     This implementation follows the Label Propagation technique introduced in
#     "Learning from Labeled and Unlabeled Data with Label Propagation" and is
#     adapted to the PyTorch Geometric framework.

#     Args:
#         num_layers (int): The number of propagations.
#         alpha (float): The alpha coefficient controlling the update rule.
#     """
#     def __init__(self, num_layers: int, alpha: float):
#         super(SimpleLabelPropagation, self).__init__(aggr='add')  # Aggregation method: add
#         self.num_layers = num_layers
#         self.alpha = alpha

#     @torch.no_grad()
#     def forward(self, x: Tensor, edge_index: Adj, mask: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
#         # x = y refers to label information rather than traditional node features. 
#         """
#         Propagates labels through the graph for semi-supervised learning.

#         Args:
#             y (Tensor): Ground-truth labels (one-hot encoded for classification).
#             edge_index (Adj): Graph connectivity in COO format with shape [2, num_edges].
#             mask (OptTensor, optional): Mask denoting which nodes are used for label propagation.
#             edge_weight (OptTensor, optional): Edge weights.

#         Returns:
#             Tensor: The propagated labels.
#         """
#         if x.dtype == torch.long:
#             x = F.one_hot(x.view(-1), num_classes=x.max() + 1).to(x.dtype)

#         out = x
#         if mask is not None:
#             out = torch.zeros_like(x)
#             out[mask] = x[mask]

#         edge_index, edge_weight = gcn_norm(edge_index, edge_weight=edge_weight, num_nodes=x.size(0), dtype=x.dtype)

#         for _ in range(self.num_layers):
#             out = self.propagate(edge_index, x=out, edge_weight=edge_weight, size=None)
#             out = self.alpha * out + (1 - self.alpha) * x  # Update rule
#             out = out.clamp(0, 1)  # Clamping as post-processing step

#         return out

#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         """
#         Computes the message to be sent to each node.

#         Args:
#             x_j (Tensor): The input features of the neighbors.
#             edge_weight (OptTensor): Edge weights.

#         Returns:
#             Tensor: The message tensor.
#         """
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(num_layers={self.num_layers}, alpha={self.alpha})'

