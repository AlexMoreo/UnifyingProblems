from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC, EMQ, AggregativeQuantifier, CC, PCC, KDEyML
from quapy.method.base import BaseQuantifier
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.protocol import UPP, ArtificialPrevalenceProtocol
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
from os.path import join
import pathlib
from util import datasets
import quapy as qp
from tqdm import tqdm
import numpy as np

from model.quantifiers import *

REPEATS = 100
result_dir = f'results/quantification/label_shift/repeats_{REPEATS}'
os.makedirs(result_dir, exist_ok=True)

datasets_selected = datasets(top_length_k=10)


def new_labelshift_protocol(X, y, classes):
    lc = LabelledCollection(X, y, classes=classes)
    app = ArtificialPrevalenceProtocol(
        lc,
        sample_size=len(lc),
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return app


def quantifiers(classifier):
    yield 'Naive', MaximumLikelihoodPrevalenceEstimation()
    yield 'CC', CC(classifier)
    yield 'PCC', PCC(classifier)
    yield 'PACC', PACC(classifier)
    yield 'EMQ', EMQ(classifier)
    yield 'KDEy', KDEyML(classifier)
    yield 'ATC-q', ATC2Quant(classifier)
    yield 'DoC-q', DoC2Quant(classifier, protocol_constructor=new_labelshift_protocol)
    # yield 'LasCal-q', LasCal2Quant(classifier)
    # yield 'PACC(LasCal)', PACCLasCal(classifier)
    # yield 'EMQ(LasCal)', EMQLasCal(classifier)


def fit_quantifier(quant, train, val):
    if isinstance(quant, AggregativeQuantifier):
        quant.fit(train, fit_classifier=False, val_split=val)
    elif isinstance(quant, BaseQuantifier):
        quant.fit(val)
    else:
        raise ValueError(f'{quant}: unrecognized object')


print('Datasets:', datasets_selected)
print('Repeats:', REPEATS)

all_results = []

pbar = tqdm(datasets_selected, total=len(datasets_selected))
for dataset in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()

    train, val = train.split_stratified(0.5, random_state=0)
    app = UPP(test, sample_size=len(test), repeats=REPEATS, random_state=0)
    qp.environ['SAMPLE_SIZE'] = len(test)

    h = LogisticRegression()
    h.fit(*train.Xy)

    for name, quant in quantifiers(classifier=h):
        result_method_dataset_path = join(result_dir, f'{name}_{dataset}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            fit_quantifier(quant, train, val)
            report = qp.evaluation.evaluation_report(quant, protocol=app, error_metrics=['ae', 'rae'])
            true_prevs = np.vstack(report['true-prev'])
            report['shift'] = qp.error.ae(true_prevs, np.tile(train_prev, (REPEATS, 1)))
            report['method'] = name
            report['dataset'] = dataset
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
df.pop('true-prev')
df.pop('estim-prev')
pivot = df.pivot_table(index='dataset', columns='method', values='ae')
print(df)
print(pivot)
print(pivot.mean(axis=0))
