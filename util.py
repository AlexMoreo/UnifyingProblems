import quapy as qp
import numpy as np
import scipy
import pandas as pd

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