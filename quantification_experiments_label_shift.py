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

import util
from util import datasets
import quapy as qp
from tqdm import tqdm
import numpy as np
from model.quantifiers import *

from commons import REPEATS, SAMPLE_SIZE, EXPERIMENT_FOLDER


result_dir = f'results/quantification/label_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)


datasets_selected = datasets(top_length_k=10)


def new_labelshift_protocol(X, y, classes):
    lc = LabelledCollection(X, y, classes=classes)
    app = ArtificialPrevalenceProtocol(
        lc,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return app


def quantifiers(classifier, Xtr, ytr):
    # quantification methods
    #yield 'Naive', MaximumLikelihoodPrevalenceEstimation()
    yield 'CC', CC(classifier)
    yield 'PCC', PCC(classifier)
    yield 'PACC', PACC(classifier)
    yield 'EMQ', EMQ(classifier)
    yield 'KDEy', KDEyML(classifier)

    # CAP methods
    yield 'ATC-q', ATC2Quant(classifier)
    yield 'DoC-q', DoC2Quant(classifier, protocol_constructor=new_labelshift_protocol)
    yield 'LEAP-q', LEAP2Quant(classifier)

    # Calibration methods
    yield 'Cpcs-q-S', Cpcs2Quant(classifier, Xtr, ytr, prob2logits=True)
    yield 'Cpcs-q-P', Cpcs2Quant(classifier, Xtr, ytr, prob2logits=False)

    yield 'TransCal-q', Transcal2Quant(classifier, Xtr, ytr, prob2logits=True)
    yield 'TransCal-q-P', Transcal2Quant(classifier, Xtr, ytr, prob2logits=False)

    yield 'LasCal-q', LasCal2Quant(classifier, prob2logits=True)
    yield 'LasCal-q-P', LasCal2Quant(classifier, prob2logits=False)

    yield 'Head2Tail-q', HeadToTail2Quant(classifier, Xtr, ytr, prob2logits=True)
    yield 'Head2Tail-q-P', HeadToTail2Quant(classifier, Xtr, ytr, prob2logits=False)
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
methods_order = []

pbar = tqdm(datasets_selected, total=len(datasets_selected))
for dataset in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()

    train, val = train.split_stratified(0.5, random_state=0)
    app = UPP(test, sample_size=SAMPLE_SIZE, repeats=REPEATS, random_state=0)
    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    h = LogisticRegression()
    h.fit(*train.Xy)

    for name, quant in quantifiers(h, *train.Xy):
        if name not in methods_order:
            methods_order.append(name)

        result_method_dataset_path = join(result_dir, f'{name}_{dataset}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            fit_quantifier(quant, train, val)
            report = qp.evaluation.evaluation_report(quant, protocol=app, error_metrics=['ae', 'rae'])
            true_prevs = np.vstack(report['true-prev'])
            report['id'] = np.arange(REPEATS)
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


from new_table import LatexTable

table = LatexTable.from_dataframe(df, method='method', benchmark='dataset', value='ae')
table.name = 'quantification_pps'
table.reorder_methods(methods_order)
table.format.configuration.show_std=False
table.latexPDF('./tables/quantification_label_shift.pdf')

