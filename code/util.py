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


key_type = nb.typeof((1, 1, 1))
tcns_dict = nb.typed.Dict.empty(
    key_type = key_type,
    value_type = types.float64,
)
list_dict = typed.List()
list_dict.append(tcns_dict)
spec_tcns = [
    ('num_nodes', types.int64),
    ('k', types.int64),
    ('tcns_list', nb.typeof(list_dict)),
    ('lambda_1', types.float64),
    ('lambda_2', types.float64),
]


@jitclass(spec_tcns)
class tcns:
    def __init__(self, num_nodes, k, lambda_1, lambda_2):
        self.num_nodes = num_nodes
        self.k = k
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def reset_tcns(self):
        tcns_list = typed.List()
        for _ in range(self.num_nodes):
            tcns_dict = nb.typed.Dict.empty(
                key_type = key_type,
                value_type = types.float64,
            )
            tcns_list.append(tcns_dict)
        self.tcns_list = tcns_list

    def reset_batch_tcns(self, node_list):
        for node_id in node_list:
            if 0 <= node_id < len(self.tcns_list):
                tcns_dict = nb.typed.Dict.empty(
                    key_type = key_type,
                    value_type = types.float64,
                )
                self.tcns_list[node_id] = tcns_dict

    def snapshot_decay(self, lambda_2):
        for node in range(self.num_nodes):
            tcns_dict = self.tcns_list[node]
            for key in tcns_dict:
                tcns_dict[key] *= lambda_2

    def backup(self):
        return self.tcns_list.copy()

    def restore(self, backup):
        self.tcns_list = backup

    def extract_tcns(self, tcns, current_timestamp, k, node_list, edge_idxs_list, delta_time_list, weight_list, position):
        if len(tcns) != 0:
            tmp_nodes = np.zeros(k, dtype = np.int32)
            tmp_edge_idxs = np.zeros(k, dtype = np.int32)
            tmp_timestamps = np.zeros(k, dtype = np.float32)
            tmp_weights = np.zeros(k, dtype = np.float32)

            for j, ((edge_idx, node, timestamp), weight) in enumerate(tcns.items()):
                tmp_nodes[j] = node
                tmp_edge_idxs[j] = edge_idx
                tmp_timestamps[j] = timestamp
                tmp_weights[j] = weight

            tmp_timestamps = current_timestamp - tmp_timestamps
            node_list[position] = tmp_nodes
            edge_idxs_list[position] = tmp_edge_idxs
            delta_time_list[position] = tmp_timestamps
            weight_list[position] = tmp_weights

    def streaming_topk(self, source_nodes, update_node_list, current_timestamp, edge_idxs, n_samples, unique_edge_indices):

        n_edges = n_samples
        n_nodes = len(update_node_list)

        node_index_map = {node: idx for idx, node in enumerate(update_node_list)}
        index_node_map = {idx: node for node, idx in node_index_map.items()}
        interaction_num = np.zeros(n_nodes, dtype = np.int32)

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            if source in node_index_map and target in node_index_map:
                source_idx = node_index_map[source]
                target_idx = node_index_map[target]
                interaction_num[source_idx] += 1
                interaction_num[target_idx] += 1

        # Structural Cohesiveness-based Importance
        SCI = np.zeros(n_nodes, dtype = np.float64)
        sci_factor = math.e

        for i in unique_edge_indices:
            source = source_nodes[i]
            source_index = node_index_map[source]
            target = source_nodes[i + n_edges]
            target_index = node_index_map[target]
            if interaction_num[source_index] == interaction_num[target_index]:
                SCI[source_index] += sci_factor ** (interaction_num[source_index] / interaction_num[target_index])
                SCI[target_index] += sci_factor ** (interaction_num[target_index] / interaction_num[source_index])
            elif interaction_num[source_index] < interaction_num[target_index]:
                SCI[source_index] += sci_factor ** (interaction_num[source_index] / interaction_num[target_index])
                SCI[target_index] += (1 / sci_factor) ** (interaction_num[target_index] / interaction_num[source_index])
            elif interaction_num[source_index] > interaction_num[target_index]:
                SCI[source_index] += (1 / sci_factor) ** (interaction_num[source_index] / interaction_num[target_index])
                SCI[target_index] += sci_factor ** (interaction_num[target_index] / interaction_num[source_index])

        tcns_list = self.tcns_list

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            timestamp = current_timestamp
            edge_idx = edge_idxs[i]
            pairs = [(source, target), (target, source)] if source != target else [(source, target)]

            for index, pair in enumerate(pairs):
                s1 = pair[0]
                s2 = pair[1]

                if tcns_list[s1] == 0:
                    s1_tcns = nb.typed.Dict.empty(
                        key_type = key_type,
                        value_type = types.float64,
                    )
                else:
                    s1_tcns = tcns_list[s1].copy()

                if tcns_list[s2] == 0:
                    s1_tcns[(edge_idx, s2, timestamp)] = SCI[node_index_map[s2]]
                else:
                    s2_tcns = tcns_list[s2]
                    for key, value in s2_tcns.items():
                        if key[1] not in node_index_map:
                            continue
                        s1_tcns[key] = SCI[node_index_map[key[1]]] * self.lambda_1
                    new_key = (edge_idx, s2, timestamp)
                    s1_tcns[new_key] = SCI[node_index_map[s2]]

                updated_tcns = nb.typed.Dict.empty(
                    key_type = key_type,
                    value_type = types.float64
                )

                tcns_size = len(s1_tcns)
                if tcns_size <= self.k:
                    updated_tcns = s1_tcns
                else:
                    keys = list(s1_tcns.keys())
                    values = np.array(list(s1_tcns.values()))
                    inds = np.argsort(values)[-self.k:]
                    for ind in inds:
                        key = keys[ind]
                        value = values[ind]
                        updated_tcns[key] = value

                if index == 0:
                    new_s1_tcns = updated_tcns
                else:
                    new_s2_tcns = updated_tcns

            if source != target:
                tcns_list[source] = new_s1_tcns
                tcns_list[target] = new_s2_tcns
            else:
                tcns_list[source] = new_s1_tcns

        batch_node = np.zeros((len(source_nodes), self.k), dtype = np.int32)
        batch_edge_idxs = np.zeros((len(source_nodes), self.k), dtype = np.int32)
        batch_delta_time = np.zeros((len(source_nodes), self.k), dtype = np.float32)
        batch_weight = np.zeros((len(source_nodes), self.k), dtype = np.float32)

        for i in unique_edge_indices:
            source = source_nodes[i]
            target = source_nodes[i + n_edges]
            fake = source_nodes[i + 2 * n_edges::n_edges]
            timestamp = current_timestamp
            edge_idx = edge_idxs[i]
            pairs = [(source, target), (target, source)] if source != target else [(source, target)]

            self.extract_tcns(tcns_list[source], timestamp, self.k, batch_node, batch_edge_idxs, batch_delta_time, batch_weight, i)
            self.extract_tcns(tcns_list[target], timestamp, self.k, batch_node, batch_edge_idxs, batch_delta_time, batch_weight, i + n_edges)
            for j in range(len(fake)):
                self.extract_tcns(tcns_list[fake[j]], timestamp, self.k, batch_node, batch_edge_idxs, batch_delta_time, batch_weight, i + 2 * n_edges + j * n_edges)

        return batch_node, batch_edge_idxs, batch_delta_time, batch_weight
