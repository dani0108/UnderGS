import torch
from torch import nn
from .util import tcns
from .backbone_model import MLPMixer, GATLayer, MaxPoolingAggregator, GINLayer


def get_embedding_module(backbone_type, node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout, args):
    if backbone_type == "gcn" or backbone_type == "mlp_mixer":
        return GraphDiffusionEmbedding(node_features=node_features,
                                       edge_features=edge_features,
                                       time_encoder=time_encoder,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       device=device,
                                       dropout=dropout,
                                       args=args,
                                       backbone_type=backbone_type)
    if backbone_type == "graph_attention" or backbone_type == "graphsage" or backbone_type == "gin":
        return GraphModuleEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    time_encoder=time_encoder,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    device=device,
                                    dropout=dropout,
                                    args=args, 
                                    backbone_type=backbone_type)                                  
    else:
        raise ValueError("Backbone {} not supported".format(backbone_type))


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout):
        super(EmbeddingModule, self).__init__()

        self.node_features = torch.from_numpy(node_features).float().to(device)    
        self.edge_features = edge_features
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features

        self.time_encoder = time_encoder
        self.dropout = dropout
        self.device = device


class GraphDiffusionEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout, args, backbone_type):
        super(GraphDiffusionEmbedding, self).__init__(node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout)

        neighbor_embedding_dimension = n_node_features + n_time_features +n_edge_features

        self.fc1 = torch.nn.Linear(neighbor_embedding_dimension, n_node_features)
        self.fc2 = torch.nn.Linear(n_node_features, n_node_features)
        self.drop = nn.Dropout(dropout)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        self.fc1_source = torch.nn.Linear(n_node_features, n_node_features)
        self.fc2_source = torch.nn.Linear(n_node_features, n_node_features)
        torch.nn.init.xavier_normal_(self.fc1_source.weight)
        torch.nn.init.xavier_normal_(self.fc2_source.weight)
        self.act = torch.nn.ReLU()

        self.filter_ratio = args.filter_ratio
        self.layer_norm = nn.LayerNorm(n_node_features)
        self.tcns = tcns(args.n_nodes, args.topk, args.lambda_1, args.lambda_2, device=self.device)

        # gcn | mlp_mixer
        self.backbone_type = backbone_type
        self.update_node_feature = torch.nn.Linear(n_node_features + n_node_features, n_node_features).to(device)
        if self.backbone_type == "mlp_mixer":
            self.mlp_mixer = MLPMixer(num_tokens=args.topk, num_channels=n_node_features)


    def compute_embedding(self, unique_edge_indices, source_nodes, update_node_list, unique_nodes, unique_indices, timestamps, edge_idxs, n_samples, train):

        selected_node, selected_edge_idxs, selected_delta_time, selected_weight = self.tcns.update_tcns(source_nodes, update_node_list, timestamps, edge_idxs, n_samples, unique_edge_indices)
        
        if self.filter_ratio != 0:
            subset_delta_time = selected_delta_time[:2 * n_samples]  
            nonzero_indices = subset_delta_time.nonzero(as_tuple=True)
            num_nonzero = len(nonzero_indices[0])  
            num_to_zero = int(self.filter_ratio * num_nonzero)  
            random_indices = torch.randperm(num_nonzero)[:num_to_zero]
            selected_positions = (nonzero_indices[0][random_indices], nonzero_indices[1][random_indices])
            selected_weight[selected_positions] = 0

        selected_node = selected_node[unique_indices]
        selected_edge_idxs = selected_edge_idxs[unique_indices]
        selected_delta_time = selected_delta_time[unique_indices]
        selected_weight = selected_weight[unique_indices]

        nodes_0 = torch.from_numpy(unique_nodes).long().to(self.device)
        source_embeddings = self.node_features[nodes_0]
        source_embeddings=self.transform_source(source_embeddings)
        node_features = self.node_features[selected_node]
        edge_features = self.edge_features[selected_edge_idxs, :] 

        time_embeddings = self.time_encoder(selected_delta_time)
        neighbor_embeddings = torch.cat([node_features,edge_features,time_embeddings], dim=-1)
        neighbor_embeddings=self.transform(neighbor_embeddings)
        weights_sum=torch.sum(selected_weight,dim=1)
        weights_sum[weights_sum < 1e-6] = 1e-6 
        weights=selected_weight/weights_sum.unsqueeze(1) 
        weights[weights_sum==0]=0
        neighbor_embeddings=neighbor_embeddings*weights[:,:,None]

        if self.backbone_type == "mlp_mixer":
            neighbor_embeddings=self.mlp_mixer(neighbor_embeddings)
        neighbor_embeddings=torch.sum(neighbor_embeddings,dim=1)
        embeddings = torch.cat((source_embeddings,neighbor_embeddings),dim=1)
        embeddings = self.update_node_feature(embeddings)
        if self.backbone_type == "mlp_mixer":
            embeddings = self.layer_norm(embeddings)
        with torch.no_grad():
            self.node_features[nodes_0] = embeddings
        return embeddings

    def transform_source(self, x):
        h = self.act(self.fc1_source(x))
        h = self.drop(h)
        output = self.fc2_source(h)    
        return output

    def transform(self, x):
        h = self.act(self.fc1(x))
        h = self.drop(h)
        output = self.fc2(h)
        return output


class GraphModuleEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout, args, backbone_type, n_layers=1, n_heads=2):
        super(GraphModuleEmbedding, self).__init__(node_features, edge_features, time_encoder, n_node_features, n_edge_features, n_time_features, device, dropout)

        self.n_layers = n_layers
        self.filter_ratio = args.filter_ratio
        self.tcns = tcns(args.n_nodes, args.topk, args.lambda_1, args.lambda_2, device=self.device)

        if backbone_type == "graphsage":
            self.layers = torch.nn.ModuleList([
                MaxPoolingAggregator(
                    n_node_features=n_node_features,
                    n_neighbors_features=n_node_features,
                    n_edge_features=n_edge_features,
                    time_dim=n_time_features,
                    n_head=n_heads,
                    dropout=dropout,
                    output_dimension=n_node_features)
                for _ in range(n_layers)
            ])
        elif backbone_type == "graph_attention":
            self.layers = torch.nn.ModuleList([
                GATLayer(
                    n_node_features=n_node_features,
                    n_neighbors_features=n_node_features,
                    n_edge_features=n_edge_features,
                    time_dim=n_time_features,
                    n_head=n_heads,
                    dropout=dropout,
                    output_dimension=n_node_features)
                for _ in range(n_layers)
            ])
        elif backbone_type == "gin":
            self.layers = torch.nn.ModuleList([
                GINLayer(
                    n_node_features=n_node_features,
                    n_neighbors_features=n_node_features,
                    n_edge_features=n_edge_features,
                    time_dim=n_time_features,
                    n_head=n_heads,
                    dropout=dropout,
                    output_dimension=n_node_features)
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")


    def aggregate(self, source_node_features, neighbor_embeddings, edge_time_embeddings, edge_features, mask, weights):
        source_embedding = source_node_features
        for n_layer in range(self.n_layers):
            attention_model = self.layers[n_layer]
            source_embedding = attention_model(source_embedding, neighbor_embeddings, edge_time_embeddings, edge_features, mask, weights)
        return source_embedding


    def compute_embedding(self, unique_edge_indices, source_nodes, update_node_list, unique_nodes, unique_indices, timestamps, edge_idxs, n_samples, train):

        selected_node, selected_edge_idxs, selected_delta_time, selected_weight = self.tcns.update_tcns(source_nodes, update_node_list, timestamps, edge_idxs, n_samples, unique_edge_indices)

        selected_node = selected_node[unique_indices]
        selected_edge_idxs = selected_edge_idxs[unique_indices]
        selected_delta_time = selected_delta_time[unique_indices]
        selected_weight = selected_weight[unique_indices]

        nodes_0 = torch.from_numpy(unique_nodes).long().to(self.device)
        neighbor_features = self.node_features[selected_node]  
        edge_features = self.edge_features[selected_edge_idxs]  
        time_embeddings = self.time_encoder(selected_delta_time) 
        source_embeddings = self.node_features[nodes_0]  

        if self.filter_ratio != 0:
            subset_delta_time = selected_delta_time[:2 * n_samples]  
            nonzero_indices = subset_delta_time.nonzero(as_tuple=True)
            num_nonzero = len(nonzero_indices[0]) 
            num_to_zero = int(self.filter_ratio * num_nonzero) 
            random_indices = torch.randperm(num_nonzero)[:num_to_zero]
            selected_positions = (nonzero_indices[0][random_indices], nonzero_indices[1][random_indices])
            selected_weight[selected_positions] = 0

        weights_sum=torch.sum(selected_weight,dim=1)
        weights=selected_weight/weights_sum.unsqueeze(1) 
        weights[weights_sum==0]=0
        aggregated_embeddings = self.aggregate(
            source_node_features=source_embeddings,
            neighbor_embeddings=neighbor_features,
            edge_time_embeddings=time_embeddings,
            edge_features=edge_features,
            mask=(selected_weight == 0),
            weights=weights
        )

        self.node_features.data[nodes_0] = aggregated_embeddings.detach()
        return aggregated_embeddings

