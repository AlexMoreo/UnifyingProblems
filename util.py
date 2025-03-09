import quapy as qp
import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame
import torch

from lascal import Ece

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def datasets(top_length_k=10):
    datasets_selected, _ = list(zip(*sorted([
        (dataset, len(qp.datasets.fetch_UCIBinaryLabelledCollection(dataset)))
        for dataset in qp.datasets.UCI_BINARY_DATASETS
    ], key=lambda x:x[1])[-top_length_k:]))
    return datasets_selected


def posterior_probabilities(h, X):
    if hasattr(h, "predict_proba"):
        P = h.predict_proba(X)
    else:
        dec_scores = h.decision_function(X)
        if dec_scores.dnim == 1:
            dec_scores = np.vstack([-dec_scores, dec_scores]).T
        P = scipy.special.softmax(dec_scores, axis=1)
    return P


def cal_error(conf_scores, y, arelogits=False):
    # expected_cal_error = Ece(adaptive_bins=True, n_bins=15, p=2, classwise=False)
    expected_cal_error = Ece(adaptive_bins=False, version='other', n_bins=15, p=2, classwise=False)

    if not arelogits:
        assert np.isclose(conf_scores.sum(axis=1), 1).all(), \
            "conf_scores are assumed to be posterior probabilities, but they don't sum up to 1"
        logits = prob2logits(conf_scores)
    else:
        logits = torch.from_numpy(conf_scores)

    ece = expected_cal_error(
        logits=logits,
        labels=torch.from_numpy(y),
    ).item()

    return ece * 100



def cap_error(acc_true, acc_estim):
    return abs(acc_true-acc_estim)



def prob2logits(P, asnumpy=False):
    logits = torch.log(torch.from_numpy(P))
    if asnumpy:
        logits = logits.numpy()
    return logits


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def accuracy_from_contingency_table(ct):
    return np.diagonal(ct).sum() / ct.sum()


def count_successes(df: DataFrame, baselines, value, expected_repetitions=100):
    datasets = df.dataset.unique()
    methods = df.method.unique()
    ids = sorted(df.id.unique())
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_baselines = len(baselines)
    n_experiments = len(ids)
    outcomes = np.zeros(shape=(n_datasets, n_methods, n_experiments))

    # collect all results in a tensor, in order of experiment idx
    for i, dataset in enumerate(datasets):
        df_data = df[df['dataset']==dataset]
        for j, method in enumerate(methods):
            df_data_method = df_data[df_data['method']==method]
            assert len(df_data_method)==expected_repetitions
            for id, val in zip(df_data_method.id.values, df_data_method[value].values):
                outcomes[i,j,id]=val

    baselines_idx = {baseline: np.where(methods==baseline)[0][0] for baseline in baselines}

    count = {}
    for j, method in enumerate(methods):
        if method in baselines: continue
        # counts how many times the method has improved over 0, 1, 2, ... #baselines baselines
        method_successes_count = {b:0 for b in range(n_baselines+1)}
        for i, dataset in enumerate(datasets):
            method_scores = outcomes[i,j]
            method_dataset_successes = np.zeros(shape=expected_repetitions, dtype=int)
            for baseline in baselines:
                baseline_scores = outcomes[i,baselines_idx[baseline]]
                successes = (method_scores <= baseline_scores)*1
                method_dataset_successes += successes
            for b in range(n_baselines+1):
                times_better_than_b_baselines = sum(method_dataset_successes>b)
                method_successes_count[b] = method_successes_count[b] + times_better_than_b_baselines
        # report the counts as fractions over the total
        method_successes_count = {b:v/(n_datasets*expected_repetitions) for b,v in method_successes_count.items()}
        count[method] = method_successes_count
    return count





