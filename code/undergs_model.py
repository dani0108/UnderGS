import numpy as np
import torch
import torch.nn as nn
from .util import MergeLayer
from .temporal_structure_learning import get_embedding_module
from .util import TimeEncode


class UnderGS(torch.nn.Module):
    def __init__(self, node_features, edge_features, device, dropout, node_dimension, time_dimension, backbone_type, args):
        super(UnderGS, self).__init__()

        if node_features is not None:
            node_raw_features = torch.from_numpy(node_features.astype(np.float32))
            n = node_raw_features.shape[1]
            node_feat_map = torch.nn.Linear(n, node_dimension)
            node_features = node_feat_map(node_raw_features).detach().numpy()
        else:
            node_features = np.zeros((args.n_nodes, node_dimension), dtype = np.float32)
        self.node_features = torch.from_numpy(node_features).float().to(device)

        edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        n_edge_features = edge_raw_features.shape[1]
        node_dimension = node_dimension
        time_encoder = TimeEncode(dimension = time_dimension)
        self.affinity_score = MergeLayer(node_dimension, node_dimension, node_dimension, 1)

        self.embedding_module = get_embedding_module(backbone_type = backbone_type,
                                                     node_features = node_features,
                                                     edge_features = edge_raw_features,
                                                     time_encoder = time_encoder,
                                                     n_node_features = node_dimension,
                                                     n_edge_features = n_edge_features,
                                                     n_time_features = time_dimension,
                                                     device = device,
													 dropout = dropout,
                                                     args = args)


    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, train):

        if negative_nodes is not None:
            nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        else:
            nodes = np.concatenate([source_nodes, destination_nodes])
        current_timestamp = edge_times[0]
        update_nodes = np.union1d(source_nodes, destination_nodes)
        unique_nodes, unique_indices = np.unique(nodes, return_index = True)

        n_samples = len(source_nodes)
        nodes = np.array(nodes,dtype=np.int32)
        edges = np.stack([source_nodes, destination_nodes], axis = 1)
        unique_edges, unique_edge_indices, unique_edge_mapping = np.unique(edges, axis = 0, return_index = True, return_inverse = True)

        node_embedding = self.embedding_module.compute_embedding(source_nodes = nodes,
                                                                 update_node_list = update_nodes,
                                                                 unique_nodes = unique_nodes,
                                                                 unique_indices = unique_indices,
                                                                 timestamps = current_timestamp,
                                                                 edge_idxs = edge_idxs,
                                                                 unique_edge_indices = unique_edge_indices,
                                                                 n_samples = n_samples,
                                                                 train = train)

        self.node_features[unique_nodes] = node_embedding.detach()
        neg_num = len(negative_nodes) // n_samples
        source_node_embedding = self.node_features[source_nodes]
        destination_node_embedding = self.node_features[destination_nodes]
        negative_node_embedding = self.node_features[negative_nodes]

        source_node_sum = source_node_embedding.repeat(neg_num + 1, 1)
        score = self.affinity_score(source_node_sum,
                                    torch.cat([destination_node_embedding, negative_node_embedding]))
        score = score.squeeze(dim = 0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        return pos_score.sigmoid(), neg_score.sigmoid()
