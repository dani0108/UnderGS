import os
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import argparse


def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(data_name) as f:
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            ts = float(e[3])
            label_list.append(0)
            feat = float(e[2])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(idx)
            feat_l.append(feat)

        u_list = np.array(u_list)
        i_list = np.array(i_list)
        ts_list = np.array(ts_list)
        feat_l = np.array(feat_l)

        rating_scaler = MinMaxScaler()
        feat_l = rating_scaler.fit_transform(feat_l.reshape(-1, 1)).ravel()

        ind = np.argsort(ts_list)
        u_list = u_list[ind]
        i_list = i_list[ind]
        ts_list = ts_list[ind]
        t_min = np.min(ts_list)
        ts_list = ts_list - t_min

        unique_u = np.unique(u_list)
        unique_i = np.unique(i_list)

        max_id = max(np.max(unique_u), np.max(unique_i)) + 1
        bitmap = np.zeros((max_id,), dtype=int)
        mapper = np.zeros((max_id,), dtype=int)

        for i in unique_u:
            bitmap[i] += 1
        for i in unique_i:
            bitmap[i] += 1

        counter = 0
        for index, val in enumerate(bitmap):
            if val != 0:
                mapper[index] = counter
                counter += 1

        for index, val in enumerate(u_list):
            u_list[index] = mapper[val]

        for index, val in enumerate(i_list):
            i_list[index] = mapper[val]

        unique_u = np.unique(u_list)
        unique_i = np.unique(i_list)

        df = pd.DataFrame({'u': u_list,
                           'i': i_list,
                           'ts': ts_list,
                           'label': label_list,
                           'idx': idx_list})
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        df['ts'] = df['ts'].dt.to_period('W').astype(str)
        df['ts'], _ = pd.factorize(df['ts'])
        max_t = df['ts'].nunique()
        print(f'u {len(unique_u)}, i {len(unique_i)}, size {max_t}, max_u {np.max(unique_u)}, max_i {np.max(unique_i)}')

    return df, feat_l


def reindex(df):
    new_df = df.copy()
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def run(data_name):
    Path("data/").mkdir(parents=True, exist_ok=True)
    PATH = f'./data/{data_name}/{data_name}.csv'
    OUT_DF = f'./data/{data_name}/ml_{data_name}.csv'
    OUT_FEAT = f'./data/{data_name}/ml_{data_name}.npy'

    df, feat = preprocess(PATH)
    new_df = reindex(df)
    new_df.to_csv(OUT_DF)

    feat = feat.reshape(-1, 1) if len(feat.shape) == 1 else feat
    empty = np.zeros((1, feat.shape[1]))
    feat = np.vstack([empty, feat])
    np.save(OUT_FEAT, feat)


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='bitcoinotc')
args = parser.parse_args()
run(args.data)
