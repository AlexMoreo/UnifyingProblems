import itertools
import os
from glob import glob
import pandas as pd
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import util
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import quapy as qp
from commons import uci_datasets
from model.classifier_accuracy_predictors import *
from commons import SAMPLE_SIZE, REPEATS
from tqdm import tqdm


def binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=1, title=None, show_std=True, legend=True,
                    train_prev=None, savepath=None, method_order=None, num_bins=10):

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0, 1], [0, 1], '--k', label='ideal', zorder=1)

    order = list(zip(method_names, true_prevs, estim_prevs))
    if method_order is not None:
        table = {method_name:[true_prev, estim_prev] for method_name, true_prev, estim_prev in order}
        order  = [(method_name, *table[method_name]) for method_name in method_order]

    NUM_COLORS = len(method_names)
    if NUM_COLORS>10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for method, true_prev, estim_prev in order:
        bin_indices = np.digitize(true_prev[:,pos_class], bins=bin_edges, right=True)

        estim_prev = estim_prev[:,pos_class]

        y_ave = np.array([estim_prev[bin_indices == i].mean() for i in range(1, num_bins + 1)])
        y_std = np.array([estim_prev[bin_indices == i].std() for i in range(1, num_bins + 1)])

        ax.errorbar(bin_centers, y_ave, fmt='-', marker='o', label=method, markersize=3, zorder=2)

        if show_std:
            ax.fill_between(bin_centers, y_ave - y_std, y_ave + y_std, alpha=0.1)

    if train_prev is not None:
        train_prev = train_prev[pos_class]
        ax.scatter(train_prev, train_prev, c='c', label='tr-prev', linewidth=2, edgecolor='k', s=100, zorder=3)


    ax.set(xlabel='true prevalence', ylabel='estimated prevalence', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    os.makedirs(Path(savepath).parent, exist_ok=True)
    plt.savefig(savepath)

task = 'classifier_accuracy_prediction'
dataset_shift = 'label_shift'



def cap_methods(h:BaseEstimator, Xva, yva):

    # CAP methods
    yield 'Naive', NaiveIID(classifier=h).fit(Xva, yva)
    yield 'ATC', ATC(h).fit(Xva, yva)
    val_prot = UPP(val, sample_size=SAMPLE_SIZE, repeats=REPEATS, random_state=0, return_type='labelled_collection')
    yield 'DoC', DoC(h, protocol=val_prot).fit(Xva, yva)
    yield 'LEAP', LEAP(classifier=h, q_class=KDEyML(classifier=h)).fit(Xva, yva)


    # Calibration 2 CAP
    yield 'TransCal-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=TransCalCalibrator, probs2logits=True, Ftr=Xtr, ytr=ytr).fit(Xva, yva)
    # yield 'Cpcs-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=CpcsCalibrator, probs2logits=True, Ftr=Xtr, ytr=ytr).fit(Xva, yva)
    yield 'LasCal-a-P', LasCal2CAP(classifier=h, probs2logits=False).fit(Xva, yva)

    # Quantification 2 CAP
    yield 'PACC-a', Quant2CAP(classifier=h, quantifier_class=PACC).fit(Xva, yva)
    yield 'KDEy-a', Quant2CAP(classifier=h, quantifier_class=KDEyML).fit(Xva, yva)
    yield 'EMQ-a', Quant2CAP(classifier=h, quantifier_class=EMQ).fit(Xva, yva)


def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    yield 'mlp', MLPClassifier()

pbar = tqdm(itertools.product(datasets_selected, classifiers()), total=len(datasets_selected) * n_classifiers)
for dataset, (cls_name, h) in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy
    h.fit(Xtr, ytr)

    Xva, yva = val.Xy

    # sample generation protocol ("artificial prevalence protocol" -- generates prior probability shift)
    app = UPP(test, sample_size=SAMPLE_SIZE, repeats=REPEATS, return_type='labelled_collection')

    for name, cap_method in cap_methods(h, Xva, yva):
        if name not in methods_order:
            methods_order.append(name)

        result_method_dataset_path = join(result_dir, f'{name}_{dataset}_{cls_name}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            method_dataset_results = []
            for id, test_shifted in tqdm(enumerate(app()), total=app.total(), desc=f'model={name}-h={cls_name}'):
                Xte, yte = test_shifted.Xy

                y_pred = h.predict(Xte)
                acc_true = accuracy(y_true=yte, y_pred=y_pred)

                acc_estim = cap_method.predict(Xte)
                err_cap = cap_error(acc_true=acc_true, acc_estim=acc_estim)

                shift = qp.error.ae(test_shifted.prevalence(), train_prev)
                result = ResultRow(dataset=dataset, id=id, method=name, classifier=cls_name, shift=shift,
                                   err=err_cap)
                method_dataset_results.append(asdict(result))

            report = pd.DataFrame(method_dataset_results)
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

    path = f'figures/diagonal/quantification_{dataset}.pdf'
    binary_diagonal(
        method_names=method_names,
        true_prevs=true_prevs,
        estim_prevs=estim_prevs,
        show_std=True,
        savepath=path,
        train_prev=train_prev,
        method_order=include_methods,
        num_bins=15
    )
