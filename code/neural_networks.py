import torch
from torch import nn
from .util import MergeLayer


class MLPMixer(nn.Module):
    def __init__(self, num_tokens: int, num_channels: int, n_layers: int = 2, hidden_dim: int = 256):
        super(MLPMixer, self).__init__()
        # Token Mixing
        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(num_tokens, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_tokens)
                )
                for _ in range(n_layers)
            ]
        )
        # Channel Mixing
        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(num_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_channels)
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_tensor: torch.Tensor):
        # Mix tokens
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        hidden_tensor = self.token_mixer(hidden_tensor).permute(0, 2, 1)
        output_tensor = hidden_tensor + input_tensor
        # Mix channels
        hidden_tensor = self.channel_norm(output_tensor)
        hidden_tensor = self.channel_mixer(hidden_tensor)
        output_tensor = hidden_tensor + output_tensor
        return output_tensor


class GATLayer(torch.nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, output_dimension, n_head=2, dropout=0.1):
        super(GATLayer, self).__init__()
        self.act = torch.nn.ReLU()
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.query_dim = n_node_features
        self.key_dim = n_neighbors_features + time_dim + n_edge_features
        self.transform = nn.Linear(self.key_dim, self.feat_dim)
        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)
        self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim, kdim=self.feat_dim, vdim=self.feat_dim, num_heads=n_head, dropout=dropout)

    def forward(self, src_node_features, neighbors_features, neighbors_time_features, edge_features, neighbors_padding_mask, weights):

        query = torch.unsqueeze(src_node_features, dim=1)  
        key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)
        key = self.transform(key)
        key = key * weights[:, :, None]

        query = query.permute([1, 0, 2]) 
        key = key.permute([1, 0, 2]) 
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True) 
        neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
        attn_output, _attn_output_weights = self.multi_head_target(query=query, key=key, value=key, key_padding_mask=neighbors_padding_mask)

        attn_output = attn_output.squeeze() 
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0) 
        attn_output = self.act(attn_output)
        attn_output = self.merger(attn_output, src_node_features)

        return attn_output


class MaxPoolingAggregator(torch.nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, output_dimension, n_head=2, dropout=0.1):
        super(MaxPoolingAggregator, self).__init__()
        n_neighbors_features = n_neighbors_features + time_dim + n_edge_features
        self.act = torch.nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.transform = nn.Linear(n_neighbors_features, n_node_features)
        self.merger = MergeLayer(n_node_features, n_node_features, n_node_features, output_dimension)

    def forward(self, src_node_features, neighbors_features, neighbors_time_features, edge_features, neighbors_padding_mask, weights):

        neighbor_embeddings = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)
        neighbor_embeddings = self.transform(neighbor_embeddings)
        neighbor_embeddings = neighbor_embeddings * weights[:, :, None]    

        neighbor_embeddings = neighbor_embeddings * (~neighbors_padding_mask).unsqueeze(-1).float()
        neighbor_transformed = self.act(neighbor_embeddings)
        neighbor_transformed = self.drop(neighbor_transformed)

        neighbor_MaxPooling = torch.max(neighbor_transformed, dim=1)[0]
        out = self.merger(neighbor_MaxPooling, src_node_features)
        return out


class GINLayer(nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim, output_dimension, n_head=2, dropout=0.1):
        super(GINLayer, self).__init__()

        n_neighbors_features = n_neighbors_features + time_dim + n_edge_features
        hidden_dim = n_node_features
        self.transform = nn.Linear(n_neighbors_features, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.epsilon = nn.Parameter(torch.Tensor([0.0]))
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, src_node_features, neighbors_features, neighbors_time_features, edge_features, neighbors_padding_mask, weights):

        neighbor_embeddings = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)
        neighbor_embeddings = self.transform(neighbor_embeddings)
        neighbor_embeddings = neighbor_embeddings * weights[:, :, None]

        neighbor_embeddings = neighbor_embeddings * (~neighbors_padding_mask).unsqueeze(-1).float()
        neighbor_agg = neighbor_embeddings.sum(dim=1)
        
        out = (1 + self.epsilon) * src_node_features + neighbor_agg
        out = self.mlp(out)
        out = self.layer_norm(out)
        return out

