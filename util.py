import os

import numpy as np
import scipy
import pandas as pd
from pandas import DataFrame
import torch
from scipy.stats import binom
from sklearn.base import BaseEstimator

from lascal import Ece
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


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

def get_ranks(df: DataFrame, value, expected_repetitions=100):
    from scipy.stats import rankdata

    datasets = sorted(df.dataset.unique())
    methods = sorted(df.method.unique())
    if 'classifier' not in df.columns:
        df['classifier']='lr' # the only classifier used in quantification experiments
    classifiers = sorted(df.classifier.unique())
    # classifiers = df.classifier.unique() if 'classifier' in df.columns else ['lr']
    # ids = sorted(df.id.unique())
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_classifiers = len(classifiers)
    n_experiments = expected_repetitions * n_classifiers * n_datasets

    version=1
    print(f'{version=}')


    if version==2:
        df_sorted2 = df.sort_values(by=["method", "dataset", "classifier", "id"])
        matrix2 = df_sorted2.pivot_table(index="method", columns=["dataset", "classifier", "id"], values=value)
        outcomes2 = np.round(matrix2.to_numpy(), decimals=8)
        by_dataset2 = [{'method': method, 'dataset': f'{idx}', 'score': 1-val} for m, method in enumerate(methods) for idx, val in enumerate(outcomes2[m])]
        df2 = pd.DataFrame(by_dataset2)
        ranks2 = rankdata(outcomes2, axis=0, method='average')
        ave_ranks2 = ranks2.mean(axis=1)
        method_ranks2 = {method: ave_ranks2[i] for i, method in enumerate(methods)}
        print(method_ranks2)
        print(df2.pivot_table(
            index="dataset",
            columns="method",
            values="score"
        ))
        return method_ranks2, df2

    else:

        # collect all results in a tensor of shape (n_methods, total_experiments), in order of experiment idx
        outcomes = np.zeros(shape=(n_methods, n_experiments))
        by_dataset = []
        for j, method in enumerate(methods):
            df_method = df[df['method']==method]
            method_results = []
            for i, dataset in enumerate(datasets):
                df_data_method = df_method[df_method['dataset']==dataset]
                method_dataset_results = []
                for k, classifier in enumerate(classifiers):
                    if classifier is not None:
                        df_data_method_cls = df_data_method[df_data_method['classifier']==classifier]
                    else:
                        df_data_method_cls = df_data_method
                    if len(df_data_method_cls)!=expected_repetitions:
                        raise ValueError(f'unexpected length of dataframe {len(df_data_method_cls)}')
                    for id, val in zip(df_data_method_cls.id.values, df_data_method_cls[value].values):
                        method_dataset_results.append(val)
                        # by_dataset.append({
                        #     'method': method,
                        #     'dataset': classifier+'_'+dataset+'_'+str(id),
                        #     'score': 1-val
                        # })
                        by_dataset.append({
                            'method': method,
                            'dataset': dataset+'_'+classifier,
                            'score': 1-val
                        })
                method_results.extend(method_dataset_results)
            outcomes[j]=np.asarray(method_results)

        outcomes = np.round(outcomes, decimals=8)  # otherwise, ties are not treated correctly due to float error precision
        ranks = rankdata(outcomes, axis=0, method='average')
        ave_ranks = ranks.mean(axis=1)
        method_ranks = {method:ave_ranks[i] for i, method in enumerate(methods)}
        df = pd.DataFrame(by_dataset)
        print(method_ranks)
        print(df.pivot_table(
            index="dataset",
            columns="method",
            values="score"
        ))
        return method_ranks, df


def count_successes(df: DataFrame, baselines, value, expected_repetitions=100, p_val=0.05):
    datasets = df.dataset.unique()
    methods = df.method.unique()
    classifiers = df.classifier.unique() if 'classifier' in df.columns else [None]
    ids = sorted(df.id.unique())
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_baselines = len(baselines)
    n_classifiers = len(classifiers)
    n_experiments = len(ids)
    outcomes = np.zeros(shape=(n_datasets, n_methods, n_experiments*n_classifiers))

    # collect all results in a tensor of shape (n_datasets, n_methods, total_experiments), in order of experiment idx
    for i, dataset in enumerate(datasets):
        df_data = df[df['dataset']==dataset]
        for j, method in enumerate(methods):
            df_data_method = df_data[df_data['method']==method]
            for k, classifier in enumerate(classifiers):
                if classifier is not None:
                    df_data_method_cls = df_data_method[df_data_method['classifier']==classifier]
                else:
                    df_data_method_cls = df_data_method
                if len(df_data_method_cls)!=expected_repetitions:
                    # print(f'unexpected length of dataframe {len(df_data_method_cls)}')
                    raise ValueError(f'unexpected length of dataframe {len(df_data_method_cls)}')
                for id, val in zip(df_data_method_cls.id.values, df_data_method_cls[value].values):
                    outcomes[i,j,k*n_experiments+id]=val

    baselines_idx = {baseline: np.where(methods==baseline)[0][0] for baseline in baselines}

    count = {}
    reject_H0 = {}
    for j, method in enumerate(methods):
        if method in baselines: continue
        # counts how many times the method has improved over 0, 1, 2, ... #baselines baselines
        method_successes_count = {b:0 for b in range(1,n_baselines+1)}
        ave = []
        for i, dataset in enumerate(datasets):
            method_scores = outcomes[i,j]
            method_dataset_successes = np.zeros(shape=expected_repetitions*n_classifiers, dtype=int)
            for baseline in baselines:
                baseline_scores = outcomes[i,baselines_idx[baseline]]
                successes = (method_scores <= baseline_scores)*1
                method_dataset_successes += successes
            ave.append(np.mean(method_dataset_successes))
            for b in range(1, n_baselines+1):
                times_better_than_b_baselines = sum(method_dataset_successes>=b)
                method_successes_count[b] = method_successes_count[b] + times_better_than_b_baselines
        # report the counts as fractions over the total
        total_experiments = n_datasets*expected_repetitions*n_classifiers
        # establish the theoretical probability of success assuming the method and the baselines are not different
        # i.e., for 3 baselines, P(M>1 method) = 75%, P(M>2 methods) = 50%, P(M>3 methods) = 25%
        rand_expected_success = {b:1.-b/(n_baselines+1) for b in range(1, n_baselines+1)}
        # method_successes_stat  = {
        #     b:1-binom.cdf(v-1, total_experiments, rand_expected_success[b])<p_val for b, v in method_successes_count.items()
        # }
        method_successes_stat = {}
        for b, v in method_successes_count.items():
            # we only keep testing for b baselines if the test was passed for b-1 baselines
            keep_testing = (b==1) or (method_successes_stat[b-1]==True)
            if keep_testing:
                test = 1 - binom.cdf(v - 1, total_experiments, rand_expected_success[b])
                reject = test < p_val
            else:
                reject = False
            method_successes_stat[b] = reject
        method_successes_prop = {b:v/total_experiments for b,v in method_successes_count.items()}

        count[method] = method_successes_prop
        reject_H0[method] = method_successes_stat

        ave = np.mean(ave)
        count[method]['ave']=ave

    return count, reject_H0


def empirical_baseline(df, baselines, value, n_permutations=1000, expected_repetitions=100):
    """
    Get empirical expected success rates under the null hypothesis via permutations
    """
    datasets = df.dataset.unique()
    methods = df.method.unique()
    classifiers = df.classifier.unique() if 'classifier' in df.columns else [None]
    n_baselines = len(baselines)
    n_experiments = expected_repetitions * len(classifiers)

    outcomes = np.zeros((len(datasets), len(methods), n_experiments))

    for i, dataset in enumerate(datasets):
        df_data = df[df['dataset'] == dataset]
        for j, method in enumerate(methods):
            df_data_method = df_data[df_data['method'] == method]
            for k, classifier in enumerate(classifiers):
                if classifier is not None:
                    df_data_method_cls = df_data_method[df_data_method['classifier'] == classifier]
                else:
                    df_data_method_cls = df_data_method
                outcomes[i, j, k*n_experiments:(k+1)*n_experiments] = df_data_method_cls[value].values

    success_rates = {b: [] for b in range(1, n_baselines + 1)}

    for _ in range(n_permutations):
        permuted_outcomes = outcomes.copy()
        for i in range(len(datasets)):
            np.random.shuffle(permuted_outcomes[i])

        for j, method in enumerate(methods):
            if method in baselines:
                continue
            method_successes = np.zeros(n_experiments, dtype=int)
            for baseline in baselines:
                baseline_idx = np.where(methods == baseline)[0][0]
                successes = (permuted_outcomes[:, j, :] <= permuted_outcomes[:, baseline_idx, :]) * 1
                method_successes += successes

            for b in range(1, n_baselines + 1):
                success_rates[b].append(np.mean(method_successes >= b))

    empirical_expectation = {b: np.mean(success_rates[b]) for b in success_rates}

    return empirical_expectation


def isometric_binning(nbins, xlow=0, xhigh=1, eps=1e-5):
    bin_edges = np.linspace(xlow, xhigh, nbins+1)
    bin_edges[0] -= eps
    bin_edges[-1] += eps
    return bin_edges


def isodense_binning(nbins, data, eps=1e-5):
    bin_edges = np.percentile(data, np.linspace(0, 100, nbins + 1))
    bin_edges[0] -= eps
    bin_edges[-1] += eps
    return bin_edges


def impute_nanvalues_via_interpolation(coordinates, values):
    if np.isnan(values).any():
        nans_idx = np.isnan(values)
        nans_coord = coordinates[nans_idx]
        interp_values = np.interp(nans_coord, xp=coordinates[~nans_idx], fp=values[~nans_idx])
        values[nans_idx] = interp_values
    return values


def impose_monotonicity(values):
    # in-place
    for i in range(1, len(values)-1):
        values[i] = max(values[i], values[i-1])
    return values


def smooth(values):
    # average smoothing length=3, in-place
    values[1:-1]=np.mean(np.vstack([values[:-2], values[1:-1], values[2:]]), axis=0)
    return values


class PrecomputedClassifier(BaseEstimator):
    """
    A 'fake' classifier that stores all the precomputed predictions (as generated by, e.g., a BERT model)
    and simply returns the corresponding value when required.
    """
    
    def __init__(self):
        super().__init__()
        self.idx = np.empty(0)
        self.covariates = None
        self.posteriors = None
        self.logits = None
        self.predictions = np.empty(0)


    def feed(self, X, P, L):
        new_items = X.shape[0]
        stored_items = len(self.idx)
        new_index = np.arange(new_items) + stored_items
        self.idx = np.concatenate([self.idx, new_index])
        
        def stack(stored, V):
            return V if stored is None else np.vstack([stored, V])
        
        self.covariates = stack(self.covariates, X)
        self.posteriors = stack(self.posteriors, P)
        self.logits = stack(self.logits, L)
        self.predictions = np.concatenate([self.predictions, np.argmax(P,axis=1)])

        return new_index.reshape(-1,1)

    def fit(self, X, y):
        raise ValueError("this classifier is already fit; use feed to store the precomputed values")
        return self
    
    def predict_proba(self, idx):
        return self.posteriors[idx.flatten()]
    
    def decision_function(self, idx):
        return self.logits[idx.flatten()]
    
    def predict(self, idx):
        return self.predictions[idx.flatten()]
    
    @property
    def classes_(self):
        return np.asarray([0,1])
    
def makepath(path):
    parent = Path(path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_text(path, text):
    makepath(path)
    with open(path, 'wt') as foo:
        foo.write(text)






