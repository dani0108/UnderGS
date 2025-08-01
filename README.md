Understanding Evolving Graph Structures for Large Discrete-Time Dynamic Graph Representation
=============================================================================

## Requirements
```sh
pip install -r requirements.txt
```

## Dataset
We collect 7 discrete-time dynamic graphs for model evaluation.
1. Wikipedia is downloaded from https://doi.org/10.5281/zenodo.7008205.
2. The remaining datasets are obtained from https://github.com/snap-stanford/roland.

## Preprocessing
generate discrete-time dynamic graphs
```sh
python preprocessing/preprocess_data.py --data DATASET
```

## Usage

1. Key Arguments (Tunable)

| Argument                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `--neural_network`  | Choice of neural network: `gcn`, `gin`, `graph_attention`, `graphsage`, `mlp_mixer`. |
| `--filter_ratio`         | Ratio of historical neighbors to randomly mask. Range: `0` to `1`. |
| `--lambda_1`             | Hop decay factor. Lower values represent stronger decay over hop distance. Range: `0` to `1`. |
| `--lambda_2`             | Snapshot decay factor controlling how fast influence decays over snapshot. Smaller values represent faster decay. Range: `0` to `1`. |

2. Running Example 

Small Dataset
```sh
    python train.py --data bitcoinotc --lr 0.001 --filter_ratio 0.1 --neural_network gcn
```
Large Dataset
```sh
    python train_large_graph.py --data reddit-body --lr 0.0005 --filter_ratio 0.4 --neural_network gcn
```
