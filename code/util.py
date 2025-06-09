import numpy as np
import torch
from torch import nn
from numba.experimental import jitclass
from numba.typed import Dict
from collections import defaultdict
import random
import math
from numba import types, typed
import numba as nb


def create_batches_from_graphs(graphs, batch_size):
    new_graphs = []
    for graph in graphs:
        graph_batches = []  
        if len(graph) <= batch_size:
            graph_batches.append(graph)  
        else:
            for i in range(0, len(graph), batch_size):
                graph_batches.append(graph[i:i + batch_size])
        new_graphs.append(graph_batches)  
    return new_graphs


def group_by_timestamps(data):
    grouped_data = defaultdict(list)
    for i in range(len(data.sources)):
        grouped_data[data.timestamps[i]].append(i)
    return grouped_data


def group_data(data):
    grouped_data = group_by_timestamps(data)
    graphs = [grouped_data[key] for key in sorted(grouped_data.keys())]
    return graphs


class TimeEncode(nn.Module):
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dimension, dtype=np.float32))).reshape(self.dimension, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dimension))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class EarlyStopMonitor(object):
    def __init__(self, max_round=5):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None

    def early_stop_check(self, curr_val):
        if self.last_best is None:
            self.last_best = curr_val
        elif curr_val > self.last_best:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


class tcns:
    def __init__(self, num_nodes, k, lambda_1, lambda_2, device):
        self.num_nodes = num_nodes
        self.k = k
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device

        self.tcns_keys = torch.full((num_nodes, k, 3), -1, dtype=torch.long, device=device) 
        self.tcns_weights = torch.zeros((num_nodes, k), dtype=torch.float32, device=device)

    def reset_tcns(self):
        self.tcns_keys.fill_(-1)
        self.tcns_weights.zero_()

    def reset_batch_tcns(self, node_list):
        self.tcns_keys[node_list] = -1
        self.tcns_weights[node_list] = 0.0

    def snapshot_decay(self, lambda_2):
        self.tcns_weights *= lambda_2

    def backup(self):
        return {
            'tcns_keys': self.tcns_keys.clone().detach(),
            'tcns_weights': self.tcns_weights.clone().detach()
        }

    def restore(self, backup):
        self.tcns_keys = backup['tcns_keys'].clone().detach()
        self.tcns_weights = backup['tcns_weights'].clone().detach()

    def update_tcns(self, source_nodes, update_node_list, current_timestamp, edge_idxs, n_samples, unique_edge_indices):

        tcns_keys = self.tcns_keys.detach().cpu().clone()
        tcns_weights = self.tcns_weights.detach().cpu().clone()

        container = Container(self.num_nodes, self.k, self.lambda_1, self.lambda_2)
        container.load_tcns(tcns_keys.numpy(), tcns_weights.numpy())

        batch_node, batch_edge_idxs, batch_delta_time, batch_weight = container.maintain(
            source_nodes,
            update_node_list,
            current_timestamp,
            edge_idxs,
            n_samples,
            unique_edge_indices
        )

        batch_node = torch.from_numpy(batch_node).long().to(self.device)       
        batch_edge_idxs = torch.from_numpy(batch_edge_idxs).long().to(self.device) 
        batch_delta_time = torch.from_numpy(batch_delta_time).float().to(self.device)
        batch_weight = torch.from_numpy(batch_weight).float().to(self.device)

        return batch_node, batch_edge_idxs, batch_delta_time, batch_weight




key_type = nb.typeof((1, 1, 1))
container_dict = nb.typed.Dict.empty(
    key_type = key_type,
    value_type = types.float64,
)
list_dict = typed.List()
list_dict.append(container_dict)
spec_container = [
    ('num_nodes', types.int64),
    ('k', types.int64),
    ('container_list', nb.typeof(list_dict)),
    ('lambda_1', types.float64),
    ('lambda_2', types.float64),
]

@jitclass(spec_container)
class Container:
    def __init__(self, num_nodes, k, lambda_1, lambda_2):
        self.num_nodes = num_nodes
        self.k = k
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        container_list = typed.List()
        for _ in range(self.num_nodes):
            container_dict = nb.typed.Dict.empty(
                key_type = key_type,
                value_type = types.float64,
            )
            container_list.append(container_dict)
        self.container_list = container_list

    def load_tcns(self, keys_tensor, weights_tensor):
        for node_id in range(self.num_nodes):
            new_dict = typed.Dict.empty(key_type=key_type, value_type=types.float64)
            for i in range(self.k):
                edge_idx, neighbor, timestamp = keys_tensor[node_id, i]
                weight = weights_tensor[node_id, i]
                if edge_idx == -1:
                    continue  
                new_dict[(int(edge_idx), int(neighbor), int(timestamp))] = float(weight)
            self.container_list[node_id] = new_dict

    def extract_content(self, content, current_timestamp, k, node_list, edge_idxs_list, delta_time_list, weight_list, position):
        if len(content) != 0:
            tmp_nodes = np.zeros(k, dtype=np.int32)
            tmp_edge_idxs = np.zeros(k, dtype=np.int32)
            tmp_timestamps = np.zeros(k, dtype=np.float32)
            tmp_weights = np.zeros(k, dtype=np.float32)

            for j, ((edge_idx, node, timestamp), weight) in enumerate(content.items()):
                if j >= k:
                    break
                tmp_nodes[j] = node
                tmp_edge_idxs[j] = edge_idx
                tmp_timestamps[j] = timestamp
                tmp_weights[j] = weight

            tmp_timestamps = current_timestamp - tmp_timestamps
            node_list[position] = tmp_nodes
            edge_idxs_list[position] = tmp_edge_idxs
            delta_time_list[position] = tmp_timestamps
            weight_list[position] = tmp_weights

    def maintain(self, source_nodes, update_node_list, current_timestamp, edge_idxs, n_samples, unique_edge_indices):
        n_edges = n_samples
        n_nodes = len(update_node_list)

        node_index_map = {node: idx for idx, node in enumerate(update_node_list)}
        interaction_num = np.zeros(n_nodes, dtype=np.int32)

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            if source in node_index_map and target in node_index_map:
                interaction_num[node_index_map[source]] += 1
                interaction_num[node_index_map[target]] += 1

        SCI = np.zeros(n_nodes, dtype=np.float64)
        sci_factor = math.e

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            if source not in node_index_map or target not in node_index_map:
                continue
            s_idx = node_index_map[source]
            t_idx = node_index_map[target]
            n_s = interaction_num[s_idx]
            n_t = interaction_num[t_idx]
            if n_s == n_t:
                SCI[s_idx] += sci_factor
                SCI[t_idx] += sci_factor
            elif n_s < n_t:
                SCI[s_idx] += sci_factor ** (n_s / n_t)
                SCI[t_idx] += (1 / sci_factor) ** (n_t / n_s)
            else:
                SCI[s_idx] += (1 / sci_factor) ** (n_s / n_t)
                SCI[t_idx] += sci_factor ** (n_t / n_s)

        container_list = self.container_list

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            edge_idx = edge_idxs[i]
            timestamp = current_timestamp
            pairs = [(source, target), (target, source)] if source != target else [(source, target)]

            for idx, (s1, s2) in enumerate(pairs):
                s1_container = container_list[s1].copy()

                if len(container_list[s2]) == 0:
                    s1_container[(edge_idx, s2, timestamp)] = SCI[node_index_map[s2]]
                else:
                    s2_container = container_list[s2]
                    for key, _ in s2_container.items():
                        if key[1] in node_index_map:
                            s1_container[key] = SCI[node_index_map[key[1]]] * self.lambda_1
                    s1_container[(edge_idx, s2, timestamp)] = SCI[node_index_map[s2]]

                updated_container = typed.Dict.empty(key_type=key_type, value_type=types.float64)
                if len(s1_container) <= self.k:
                    updated_container = s1_container
                else:
                    keys = list(s1_container.keys())
                    values = np.array(list(s1_container.values()))
                    inds = np.argsort(values)[-self.k:]
                    for ind in inds:
                        key = keys[ind]
                        updated_container[key] = s1_container[key]

                if idx == 0:
                    new_s1_container = updated_container
                else:
                    new_s2_container = updated_container

            if source != target:
                container_list[source] = new_s1_container
                container_list[target] = new_s2_container
            else:
                container_list[source] = new_s1_container

        batch_node = np.zeros((len(source_nodes), self.k), dtype=np.int32)
        batch_edge_idxs = np.zeros((len(source_nodes), self.k), dtype=np.int32)
        batch_delta_time = np.zeros((len(source_nodes), self.k), dtype=np.float32)
        batch_weight = np.zeros((len(source_nodes), self.k), dtype=np.float32)

        for i in unique_edge_indices:
            self.extract_content(container_list[source_nodes[i]], current_timestamp, self.k,
                              batch_node, batch_edge_idxs, batch_delta_time, batch_weight, i)
            self.extract_content(container_list[source_nodes[i + n_edges]], current_timestamp, self.k,
                              batch_node, batch_edge_idxs, batch_delta_time, batch_weight, i + n_edges)

        return batch_node, batch_edge_idxs, batch_delta_time, batch_weight