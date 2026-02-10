import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
import math
from torch.nn.parameter import Parameter

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

def get_configs(dataset):
    dataset_class = get_dataset_class(dataset)
    hparams_class = get_hparams_class(dataset)
    return dataset_class(), hparams_class()

# Spatial-Temporal Feature Encoder
class TemporalSpatialNN_new(nn.Module):
    def __init__(self, configs):
        super(TemporalSpatialNN_new, self).__init__()
        self.configs = configs
        self.temporal_cnn = Temporal_CNN(self.configs)
        self.graph_learner = Graph_Learner_new()
        self.spatial_gnn = Spatial_GNN(self.configs)

    def forward(self, x):
        temp_x = self.temporal_cnn(x)
        adj = self.graph_learner(temp_x)
        spat_x, spat_flat = self.spatial_gnn(temp_x, adj)

        return spat_x, spat_flat

# Temporal CNN
class Temporal_CNN(nn.Module):
    '''
    extractor each channels features.
    '''
    def __init__(self, configs):
        super(Temporal_CNN, self).__init__()
        self.configs = configs

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels * configs.input_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2), groups=configs.input_channels),
            nn.BatchNorm1d(configs.mid_channels * configs.input_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * configs.input_channels, configs.mid_channels * 2 * configs.input_channels, kernel_size=8, stride=1, bias=False, padding=4, groups=configs.input_channels),
            nn.BatchNorm1d(configs.mid_channels * 2 * configs.input_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2 * configs.input_channels, configs.final_channels * configs.input_channels, kernel_size=8, stride=1, bias=False,
                      padding=4, groups=configs.input_channels),
            nn.BatchNorm1d(configs.final_channels * configs.input_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        out = self.aap(x).view(x.shape[0], self.configs.input_channels, -1)

        return out

# Graph Learner
class Graph_Learner_new(nn.Module):
    def __init__(self):
        super(Graph_Learner_new, self).__init__()

    def forward(self, x):
        '''
        input: node_feats: [batch, num_nodes, feat_dim]
        output: graph adjs: [batch, num_nodes, num_nodes]
        '''
        x_n = F.normalize(x, p =2.0, dim=-1) # normalize features. [batch, num_nodes, feats]
        x_n_t = torch.transpose(x_n, 1, 2)  # [batch, feats, num_nodes]
        adj = torch.bmm(x_n, x_n_t) # [batch, num_nodes, num_nodes]
        adj = F.relu(adj) # non-linear act

        return adj

# Graph Convolution
class GraphConvolution(nn.Module):
    """
    GCN Convolution Module.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('mik,kj->mij', input, self.weight)
        out = torch.einsum('mki,mij->mkj', adj, support)
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# Spatial GNN
class Spatial_GNN(nn.Module):
    def __init__(self, configs):
        super(Spatial_GNN, self).__init__()
        self.conv1 = GraphConvolution(configs.gnn_input_dim, configs.gnn_output_dim)
        self.dp = nn.Dropout(configs.dropout_spatial_gnn)
        self.act = nn.PReLU()
        self.bn = nn.BatchNorm1d(configs.gnn_output_dim)
        self.adj_norm = configs.adj_norm

    def forward(self, x, adj):
        h = self.dp(x)
        if self.adj_norm:
            adj = normalize_batch_adj_both(adj)
        h = self.conv1(h, adj)
        h = torch.transpose(self.bn(h.transpose(1, 2)), dim0=1, dim1=2)
        h = self.act(h)

        h_flat = h.contiguous().view(x.shape[0], -1)

        return h, h_flat

# Classifier
class Classifier(nn.Module):
    def __init__(self, configs):
        super(Classifier, self).__init__()

        # model_output_dim = configs.final_channels  # do not use graph learning.
        model_output_dim = configs.gnn_output_dim
        self.logits = nn.Linear(model_output_dim * configs.input_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions

# Spatial Rewiring

# Regressor (for continuous output like NASA-TLX scores)
class Regressor(nn.Module):
    def __init__(self, configs):
        super(Regressor, self).__init__()
        model_output_dim = configs.gnn_output_dim
        input_dim = model_output_dim * configs.input_channels  # 64 * 32 = 2048 for SENSE_42
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # shape: (batch,) not (batch, 1)

class GraphRecover_new(nn.Module):
    def __init__(self, configs):
        super(GraphRecover_new, self).__init__()
        self.conv1 = GraphConvolution(configs.gnn_output_dim, configs.gnn_output_dim)
        self.dp = nn.Dropout(configs.dropout_graph_recover)
        self.act = nn.PReLU()
        self.bn = nn.BatchNorm1d(configs.gnn_output_dim)
        self.adj_norm = configs.adj_norm

    def forward(self, x, adj):
        # Extract node features.
        h = self.dp(x)
        if self.adj_norm:
            adj = normalize_batch_adj_both(adj)
        h = self.conv1(h, adj)
        h = torch.transpose(self.bn(h.transpose(1, 2)), dim0=1, dim1=2)
        h = self.act(h)

        h = F.normalize(h, p=2.0, dim=-1)  # should normalize features.

        h_t = torch.transpose(h, 1, 2)  # [batch, feat_dim, num_nodes]
        new_adj = torch.bmm(h, h_t)  # [batch, num_nodes, num_nodes]
        new_adj = F.relu(new_adj)

        return new_adj

# Normlization Adjs
def normalize_batch_adj_both(batch_adj_matrices):
    """
    Normalize batched adjacency matrices based on the formula A_norm = D^(-1/2) A D^(-1/2),
    where D is the degree matrix.

    Args:
    - batch_adj_matrices (torch.Tensor): Input batched adjacency matrices with shape [batch, num_nodes, num_nodes].

    Returns:
    - normalized_batch_adj_matrices (torch.Tensor): Normalized batched adjacency matrices.
    """
    # Calculate degree matrices D
    degree = batch_adj_matrices.sum(dim=2)

    # Invert square root of the degree matrices, handle division by zero by replacing with 0
    degree_sqrt_inv = torch.where(degree != 0, torch.pow(degree, -0.5), torch.tensor(0.))

    # Create diagonal matrices with inverted square root of the degree matrices
    D_inv_sqrt = torch.diag_embed(degree_sqrt_inv)

    # Normalize adjacency matrices: A_norm = D^(-1/2) A D^(-1/2)
    normalized_batch_adj_matrices = torch.matmul(torch.matmul(D_inv_sqrt, batch_adj_matrices), D_inv_sqrt)

    return normalized_batch_adj_matrices

# Temporal Restoration
class Temporal_Imputer(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.input_channels
        self.hid_dim = configs.AR_hid_dim
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)

        return out

# Spatial Masking
def mask_adj_matrices_edges(batch_adj_matrices, mask_ratio=0.2):
    """
    Masks a batch of fully connected adjacency matrices with a given mask ratio.

    Parameters:
    batch_adj_matrices (torch.Tensor): Tensor of shape (batch_size, N, N) representing a batch of adjacency matrices.
    mask_ratio (float): The ratio of non-zero elements to mask in each adjacency matrix.

    Returns:
    torch.Tensor: Tensor of the masked adjacency matrices.
    """
    batch_size, N, _ = batch_adj_matrices.shape

    # Create an index tensor for the adjacency matrices
    all_indices = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N)), dim=-1).reshape(-1, 2)
    # Exclude diagonal indices
    off_diag_indices = all_indices[all_indices[:, 0] != all_indices[:, 1]]
    total_off_diag_elements = off_diag_indices.shape[0]  # Number of off-diagonal elements
    # Calculate the number of elements to mask
    mask_count = int(mask_ratio * total_off_diag_elements)
    # Generate random indices to mask
    random_indices = torch.randperm(total_off_diag_elements)[:mask_count]
    # Create the mask
    mask = torch.ones(batch_size, N, N, dtype=torch.bool, device=batch_adj_matrices.device)
    mask_indices = off_diag_indices[random_indices]
    # Apply the mask to each adjacency matrix in the batch
    mask[:, mask_indices[:, 0], mask_indices[:, 1]] = False
    # Apply the mask to the adjacency matrices
    masked_adj_matrices = batch_adj_matrices * mask

    return masked_adj_matrices

# Temporal Masking
def masking2(x, num_splits=8, num_masked=4):
    # Reshape input tensor to create patches
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()

    # Generate random indices for masking
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1
    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    mask = rearrange(mask, 'a b p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask