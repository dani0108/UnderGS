import os
import random
import numpy as np
import pandas as pd
from .util import group_data, create_batches_from_graphs


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def sample(self, ratio):
        data_size = self.n_interactions
        sample_size = int(ratio * data_size)
        sample_inds = random.sample(range(data_size), sample_size)
        sample_inds = np.sort(sample_inds)
        sources = self.sources[sample_inds]
        destination = self.destinations[sample_inds]
        timestamps = self.timestamps[sample_inds]
        edge_idxs = self.edge_idxs[sample_inds]
        labels = self.labels[sample_inds]
        return Data(sources, destination, timestamps, edge_idxs, labels)


def load_feat(d):
    node_feats = None
    if os.path.exists('data/{}/ml_{}_node.npy'.format(d,d)):
        node_feats = np.load('data/{}/ml_{}_node.npy'.format(d,d)) 
        node_feats = node_feats.astype(np.float32)
    edge_feats = None
    if os.path.exists('data/{}/ml_{}.npy'.format(d,d)):
        edge_feats = np.load('data/{}/ml_{}.npy'.format(d,d))
    return node_feats, edge_feats


def get_data(dataset_name, dataset_seed):

    graph_df = pd.read_csv('data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    max_ts = graph_df.ts.max()
    train_time = float(int(max_ts * 0.7))
    val_time = float(int(max_ts * 0.8))    

    sources = graph_df.u.values
    destinations = graph_df.i.values
    timestamps = graph_df.ts.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(dataset_seed)
    node_set = set(sources) | set(destinations)
    n_nodes = len(node_set)
    n_edges = len(sources)

    train_mask = timestamps <= train_time
    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])
    val_mask = (timestamps > train_time) & (timestamps <= val_time)
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_mask = timestamps > val_time
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])
    full_train_mask = timestamps <= val_time
    full_train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])

    print_dataset_statistics(full_data, train_data, val_data, test_data)
    return full_data, full_train_data, train_data, val_data, test_data, n_nodes, n_edges


def print_dataset_statistics(full_data, train_data, val_data, test_data):

    print("Dataset Overview >>>")
    datasets = [
        ("Training dataset", train_data, f"{train_data.timestamps.min()+1} - {train_data.timestamps.max()+1}"),
        ("Validation dataset", val_data, f"{val_data.timestamps.min()+1} - {val_data.timestamps.max()+1}"),
        ("Test dataset", test_data, f"{test_data.timestamps.min()+1} - {test_data.timestamps.max()+1}")
    ]

    col_widths = [25, 15, 15, 16]
    header = f"{'Dataset':<{col_widths[0]}}{'Interactions':<{col_widths[1]}}{'Unique nodes':<{col_widths[2]}}{'Snapshot Range':<{col_widths[3]}}"
    separator = "-" * sum(col_widths)

    print(separator)
    print(header)
    print(separator)

    for name, data, time_range in datasets:
        row = f"{name:<{col_widths[0]}}{data.n_interactions:<{col_widths[1]}}{data.n_unique_nodes:<{col_widths[2]}}{time_range:<{col_widths[3]}}"
        print(row)

    print(separator)


def get_large_graphs(train_data, val_data, test_data, args):
    train_graphs = group_data(train_data)
    val_graphs = group_data(val_data)
    test_graphs = group_data(test_data)
    train_batches = create_batches_from_graphs(train_graphs, args.bs)
    val_batches = create_batches_from_graphs(val_graphs, args.bs)
    test_batches = create_batches_from_graphs(test_graphs, args.bs)
    return train_batches, val_batches, test_batches


def get_graphs(train_data, val_data, test_data):
    return group_data(train_data), group_data(val_data), group_data(test_data)


