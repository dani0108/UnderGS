import os
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy
import torch.optim as optim
from code.undergs_model import UnderGS
from code.data_processing import get_data, load_feat, get_large_graphs
from code.util import EarlyStopMonitor, RandEdgeSampler
from code.evaluation import val, test_large_graph


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Dataset name', default='reddit-title')
parser.add_argument('--dataset_seed', type=int, default=822, help='Random seed for sampling')
parser.add_argument('--node_dim', type=int, default=128, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=8, help='Dimensions of the time embedding')

parser.add_argument('--gpu', type=int, default=2, help='Idx for the gpu to use')
parser.add_argument('--bs', type=int, default=200, help='Batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--drop_out', type=float, default=0, help='Dropout probability')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--backbone_type', type=str, default='gcn', choices=['gcn', 'mlp_mixer', 'graph_attention', 'graphsage', 'gin'], help='Select the backbone architecture')

parser.add_argument('--topk', type=int, default=16, help='Keep the most influential neighbors')
parser.add_argument('--filter_ratio', type=float, default=0, help='Time-aware filter ratio')
parser.add_argument('--lambda_1', type=float, default=0.1, help='Hop decay factor')
parser.add_argument('--lambda_2', type=float, default=0.0001, help='Snapshot decay factor')

args = parser.parse_args()
print("Model Configuration >>>", vars(args))

torch.manual_seed(args.dataset_seed)
np.random.seed(args.dataset_seed)

full_data, full_train_data, train_data, val_data, test_data, n_nodes, n_edges = get_data(args.data, args.dataset_seed)
args.n_nodes = n_nodes + 1
args.n_edges = n_edges + 1
node_feats, edge_feats = load_feat(args.data)
if edge_feats is None:
    edge_feats = np.zeros((args.n_edges, 1))

train_batches, val_batches, test_batches = get_large_graphs(train_data, val_data, test_data, args)

train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_train_data.sources, full_train_data.destinations, seed=0)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

model = UnderGS(node_features=node_feats, edge_features=edge_feats, device=device,
				dropout=args.drop_out,
				node_dimension=args.node_dim, time_dimension=args.time_dim,
				backbone_type=args.backbone_type,
				args=args)
model = model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

early_stopper = EarlyStopMonitor()

best_param = {'best_auc': 0, 'best_state': None}
run_epoch_time = 0

for epoch in range(args.n_epoch):
	# test
	if epoch == 22:
		print("Pass")
		break
	t_epoch_train_start = time.time()
	model.embedding_module.tcns.reset_tcns()
	total_loss = 0

	for snapshot_idx in range(len(train_batches)):
		model.embedding_module.snapshot_idx = snapshot_idx

		losses = torch.tensor(0.0).to(device)
		n_batch = len(train_batches[snapshot_idx])

		for batch_idx in range(n_batch):
			sample_inds = train_batches[snapshot_idx][batch_idx]
			sources_batch = train_data.sources[sample_inds]
			destinations_batch = train_data.destinations[sample_inds]
			timestamps_batch = train_data.timestamps[sample_inds]
			edge_idxs_batch = train_data.edge_idxs[sample_inds]
			size = len(sources_batch)
			_, negatives_batch = train_rand_sampler.sample(size)

			model = model.train()
			pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch, timestamps_batch, edge_idxs_batch, train=True)
			with torch.no_grad():
				pos_label = torch.ones(size, dtype=torch.float, device=device)
				neg_label = torch.zeros(size, dtype=torch.float, device=device)
			loss = criterion(pos_prob.squeeze(dim=1), pos_label) + criterion(neg_prob.squeeze(dim=1), neg_label)
			losses = losses + loss

		model.embedding_module.tcns.snapshot_decay(args.lambda_2)

		optimizer.zero_grad()
		losses = losses / n_batch
		total_loss = total_loss + losses
		losses.backward()
		optimizer.step()

	scheduler.step()
	total_loss = total_loss / len(train_batches)

	all_auc = []

	for snapshot_idx in range(len(val_batches)):
		n_batch = len(val_batches[snapshot_idx])

		for batch_idx in range(n_batch):
			sample_inds = val_batches[snapshot_idx][batch_idx]

			train_tcns_backup = model.embedding_module.tcns.backup()
			loss, val_auc = val(model=model, negative_edge_sampler=val_rand_sampler, data=val_data,
			                    sample_inds=sample_inds, device=device, criterion=criterion, args=args)
			model.embedding_module.tcns.restore(train_tcns_backup)
			all_auc.append(val_auc)

		model.embedding_module.tcns.snapshot_decay(args.lambda_2)

	t_epoch_train_end = time.time()
	train_time = t_epoch_train_end - t_epoch_train_start
	run_epoch_time += train_time
	print('Epoch: {}, val auc: {}, loss: {}, time: {}'.format(epoch, np.mean(all_auc), total_loss, train_time))

	val_metric = np.mean(all_auc)
	if early_stopper.early_stop_check(val_metric):
		break
	else:
		if epoch == early_stopper.best_epoch:
			best_param = {'best_auc': all_auc, 'best_state': deepcopy(model.state_dict())}

model.load_state_dict(best_param['best_state'])
model.eval()

test_results = test_large_graph(model=model, negative_edge_sampler=test_rand_sampler, data=test_data, test_batches=test_batches, criterion=criterion, args=args)
print("Training Summary >>> Total Time: {:.2f}s | Epochs: {:d} | Test AUC: {:.2f} | Test MRR: {:.2f}".format(
        run_epoch_time, epoch + 1, test_results[0] * 100, test_results[1] * 100))