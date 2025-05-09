import os
from glob import glob
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from model.classifier_accuracy_predictors import *


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

def include_cls(file_path:str, classifier):
    if file_path.endswith(classifier+'.csv'):
        return True
    if len(classifier)>5 and classifier in file_path:
        return True
    return False


def include_result(file_path, include_methods, dataset, classifier):
    if not included_method(file_path, include_methods):
        return False
    if not included_dataset(file_path, dataset):
        return False
    if not include_cls(file_path, classifier):
        return False
    return True


def cap_diagonal(method_names, true_acc, estim_acc, naive=None, title=None, legend=True, savepath=None, method_order=None):

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0, 1], [0, 1], '--k', label='ideal', zorder=1)

    order = list(zip(method_names, true_acc, estim_acc))
    if method_order is not None:
        table = {method_name:[true_acc, estim_acc] for method_name, true_acc, estim_acc in order}
        order  = [(method_name, *table[method_name]) for method_name in method_order]

    NUM_COLORS = len(method_names)
    if NUM_COLORS>10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])

    if naive is not None:
        ax.axhline(y=0.5, color="gray", linestyle="-.", linewidth=1.5, label="Naive")

    for method, true_acc, estim_acc in order:
        ax.scatter(x=true_acc, y=estim_acc, marker='o', label=method, zorder=2, alpha=0.5)

    ax.set(xlabel='true Accuracy', ylabel='estimated Accuracy', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)


    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    os.makedirs(Path(savepath).parent, exist_ok=True)
    plt.savefig(savepath)

task = 'classifier_accuracy_prediction'
dataset_shift = 'label_shift'
# dataset_shift = 'covariate_shift'

input_dir = f'./results/{task}/{dataset_shift}/repeats_100_samplesize_250'

if dataset_shift == 'label_shift':
    include_methods = [
        'Naive',
        'ATC',
        'DoC',
        'LEAP',
        'Cpcs-a-S',
        'PACC-a'
    ]

    classifiers=['lr', 'knn', 'nb', 'mlp']

    datasets = ['cmc.3']

else:
    include_methods = [
        'Naive',
        'ATC',
        'DoC',
        'LEAP',
        'Cpcs-a-S'
    ]

    classifiers = ['distilbert-base-uncased']

    datasets = ['imdb__rt']

for dataset, classifier in product(datasets, classifiers):

    results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir + '/*.csv') if include_result(result_file, include_methods, dataset, classifier)])

    method_names, true_acc, estim_acc = [], [], []

    naive=None
    for method in results.method.unique():
        df_sel = results[results['method'] == method]
        if method == 'Naive':
            naive = df_sel['estim_acc'].values[0]
        else:
            method_names.append(method)
            true_acc.append(df_sel['true_acc'].values)
            estim_acc.append(df_sel['estim_acc'].values)

    if dataset is None:
        dataset='all'
    path = f'figures/diagonal/{task}/{dataset_shift}_{dataset}_{classifier}.png'
    if 'Naive' in include_methods:
        include_methods.remove('Naive')
    cap_diagonal(
        method_names=method_names,
        true_acc=true_acc,
        estim_acc=estim_acc,
        naive=naive,
        savepath=path,
        method_order=include_methods,
    )
