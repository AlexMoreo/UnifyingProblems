import itertools
import os
from glob import glob
import pandas as pd
import util
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import quapy as qp
from commons import uci_datasets

def included_method(file_path, include_methods):
    name = Path(file_path).name.split('_')[0]
    for method in include_methods:
        if method == name:
            return True
    return False


def included_dataset(file_path, dataset=None):
    if dataset is not None:
        return dataset in file_path
    return True


def include_result(file_path, include_methods, dataset):
    if not included_method(file_path, include_methods):
        return False
    if not included_dataset(file_path, dataset):
        return False
    return True


def load_report(path):
    def str2prev_arr(strprev):
        within = strprev.strip('[]').split()
        float_list = [float(p) for p in within]
        float_list[-1] = 1. - sum(float_list[:-1])
        return np.asarray(float_list)

    df = pd.read_csv(path)
    df['true-prev'] = df['true-prev'].apply(str2prev_arr)
    df['estim-prev'] = df['estim-prev'].apply(str2prev_arr)
    return df


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

task = 'quantification'
dataset_shift = 'label_shift'

input_dir = f'./results/{task}/{dataset_shift}/repeats_100_samplesize_250'

include_methods = [
    'CC',
    # 'PCC',
    # 'PACC',
    # 'EMQ',
    'KDEy',
    # 'ATC-q',
    'DoC-q',
    'LEAP-q',
    'LasCal-q-P',
    # 'TransCal-q-P',
    'Cpcs-q-P'
]

for dataset in uci_datasets()+[None]:

    if dataset is not None:
        data = qp.datasets.fetch_UCIBinaryDataset(dataset)
        train_prev = data.training.prevalence()
    else:
        train_prev = None

    results = pd.concat([load_report(result_file) for result_file in glob(input_dir + '/*.csv') if include_result(result_file, include_methods, dataset)])

    method_names, true_prevs, estim_prevs = [], [], []

    for method in results.method.unique():
        method_names.append(method)
        df_sel = results[results['method']==method]
        true_prevs.append(np.vstack(df_sel['true-prev'].values))
        estim_prevs.append(np.vstack(df_sel['estim-prev'].values))

    if dataset is None:
        dataset='all'
    path = f'figures/diagonal/{task}/{dataset}.pdf'
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
