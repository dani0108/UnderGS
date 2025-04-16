import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score


def val(model, negative_edge_sampler, data, sample_inds, device, criterion, args):

    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    sources_batch = data.sources[sample_inds]
    destinations_batch = data.destinations[sample_inds]
    timestamps_batch = data.timestamps[sample_inds]
    edge_idxs_batch = data.edge_idxs[sample_inds]

    size = len(sources_batch)
    _, negative_samples = negative_edge_sampler.sample(size)
    pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, train=False)
    with torch.no_grad():
        pos_label = torch.ones(size, dtype=torch.float, device=device)
        neg_label = torch.zeros(size, dtype=torch.float, device=device)
    loss = criterion(pos_prob.squeeze(dim=1), pos_label) + criterion(neg_prob.squeeze(dim=1), neg_label)

    pos_prob = pos_prob.cpu().detach().numpy()
    neg_prob = neg_prob.cpu().detach().numpy()
    pred_score = np.concatenate([pos_prob, neg_prob])
    true_label = np.concatenate([np.ones(size), np.zeros(size)])
    val_auc = roc_auc_score(true_label, pred_score)

    return loss, val_auc


def test(model, negative_edge_sampler, data, test_graphs, criterion, args, neg_sample_size=100):

    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_auc, val_mrr = [], []
    for snapshot_idx in range(len(test_graphs)):
        sample_inds = test_graphs[snapshot_idx]

        sources_batch = data.sources[sample_inds]
        destinations_batch = data.destinations[sample_inds]
        timestamps_batch = data.timestamps[sample_inds]
        edge_idxs_batch = data.edge_idxs[sample_inds]

        assert negative_edge_sampler.seed is not None
        negative_edge_sampler.reset_random_state()
        size = len(sources_batch)
        _, negative_samples = negative_edge_sampler.sample(size)
        pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, train=False)
        pos_prob = pos_prob.cpu().detach().numpy()
        neg_prob = neg_prob.cpu().detach().numpy()
        pred_score = np.concatenate([pos_prob, neg_prob])
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        val_auc.append(roc_auc_score(true_label, pred_score))

        assert negative_edge_sampler.seed is not None
        negative_edge_sampler.reset_random_state()
        size = size * neg_sample_size
        _, negative_samples = negative_edge_sampler.sample(size)
        pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, train=False)
        pos_prob = pos_prob.cpu().detach().numpy()
        neg_prob = neg_prob.cpu().detach().numpy()
        mrr = compute_mrr(pos_prob, neg_prob)
        val_mrr.append(mrr)

        model.embedding_module.tcns.snapshot_decay(args.lambda_2)

    return [np.mean(val_auc), np.mean(val_mrr)]



def test_large_graph(model, negative_edge_sampler, data, test_batches, criterion, args, neg_sample_size=100):

    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()
    
    val_auc, val_mrr = [], []
    for snapshot_idx in range(len(test_batches)):
        n_batch = len(test_batches[snapshot_idx])

        for batch_idx in range(n_batch):
            sample_inds = test_batches[snapshot_idx][batch_idx]

            sources_batch = data.sources[sample_inds]
            destinations_batch = data.destinations[sample_inds]
            timestamps_batch = data.timestamps[sample_inds]
            edge_idxs_batch = data.edge_idxs[sample_inds]

            assert negative_edge_sampler.seed is not None
            negative_edge_sampler.reset_random_state()
            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)
            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, train=False)
            pos_prob = pos_prob.cpu().detach().numpy()
            neg_prob = neg_prob.cpu().detach().numpy()
            pred_score = np.concatenate([pos_prob, neg_prob])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            val_auc.append(roc_auc_score(true_label, pred_score))

            assert negative_edge_sampler.seed is not None
            negative_edge_sampler.reset_random_state()
            size = size * neg_sample_size
            _, negative_samples = negative_edge_sampler.sample(size)
            pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch, negative_samples, timestamps_batch, edge_idxs_batch, train=False)
            pos_prob = pos_prob.cpu().detach().numpy()
            neg_prob = neg_prob.cpu().detach().numpy()
            mrr = compute_mrr(pos_prob, neg_prob)
            val_mrr.append(mrr)

        model.embedding_module.tcns.snapshot_decay(args.lambda_2)

    return [np.mean(val_auc), np.mean(val_mrr)]


def compute_mrr(pred_pos, pred_neg):
    y_pred_pos, y_pred_neg = pred_pos.flatten(), pred_neg.flatten()
    num_pos = y_pred_pos.shape[0]
    reciprocal_ranks = []

    for i in range(num_pos):  
        corresponding_neg_samples = y_pred_neg[i::num_pos]  
        combined_scores = np.concatenate(([y_pred_pos[i]], corresponding_neg_samples))
        sorted_indices = np.argsort(-combined_scores, axis=0)
        rank_of_positive = np.where(sorted_indices == 0)[0][0]
        reciprocal_ranks.append(1 / (rank_of_positive + 1))  

    mrr = np.mean(reciprocal_ranks)  
    return mrr
