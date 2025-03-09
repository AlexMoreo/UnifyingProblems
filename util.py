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

def count_successes(df: DataFrame, baselines):
    datasets = df.dataset.unique()
    methods = df.method.unique()
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_experiments = df.id
    outcomes = np.zeros(shape=())
