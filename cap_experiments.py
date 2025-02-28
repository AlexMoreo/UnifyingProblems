from quapy.method.aggregative import KDEyML, EMQ
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

from model.classifier_accuracy_predictors import *
from util import accuracy, cap_error
import quapy as qp
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from util import datasets
from dataclasses import dataclass, asdict
import pandas as pd


"""
Methods:
- Naive [baseline]
- LEAP, ATC, DoC [proper CAP methods]
- Lascal2Cap [a method from calibration]
- PACC2C [a method from quantification]
- HDC2C [a method from calibration that comes from quantification]
"""

DOC_VAL_SAMPLES = 50
REPEATS = 100

result_dir = f'results/classifier_accuracy_prediction/label_shift/repeats_{REPEATS}'
os.makedirs(result_dir, exist_ok=True)

datasets_selected = datasets(top_length_k=10)

@dataclass
class ResultRow:
    dataset: str
    id: int
    method: str
    shift: float
    err: float

def cap_methods(h:BaseEstimator, Xva, yva):

    # CAP methods
    yield 'Naive', NaiveIID(classifier=h).fit(Xva, yva)
    yield 'LEAP', LEAP(classifier=h, q_class=KDEyML(classifier=h)).fit(Xva, yva)
    yield 'ATC', ATC(h).fit(Xva, yva)

    val_prot = UPP(val, sample_size=len(val), repeats=DOC_VAL_SAMPLES, random_state=0, return_type='labelled_collection')
    yield 'DoC', DoC(h, protocol=val_prot).fit(Xva, yva)

    # Calibration 2 CAP
    yield 'LasCal-a', LasCal2CAP(classifier=h).fit(Xva, yva)
    yield 'HDc-a', HDC2CAP(classifier=h).fit(Xva, yva)

    # Quantification 2 CAP
    yield 'PACC-a', Quant2CAP(classifier=h, quantifier_class=PACC).fit(Xva, yva)
    yield 'KDEy-a', Quant2CAP(classifier=h, quantifier_class=KDEyML).fit(Xva, yva)
    yield 'EMQ-a', Quant2CAP(classifier=h, quantifier_class=EMQ).fit(Xva, yva)


all_results = []

pbar = tqdm(datasets_selected, total=len(datasets_selected))
for dataset in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy

    h = LogisticRegression()
    h.fit(Xtr, ytr)

    Xva, yva = val.Xy

    # sample generation protocol ("artificial prevalence protocol" -- generates prior probability shift)
    app = UPP(test, sample_size=len(test), repeats=REPEATS, return_type='labelled_collection')

    for name, cap_method in cap_methods(h, Xva, yva):
        result_method_dataset_path = join(result_dir, f'{name}_{dataset}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            method_dataset_results = []
            for id, test_shifted in tqdm(enumerate(app()), total=app.total(), desc=f'model={name}'):
                Xte, yte = test_shifted.Xy

                y_pred = h.predict(Xte)
                acc_true = accuracy(y_true=yte, y_pred=y_pred)

                acc_estim = cap_method.predict(Xte)
                err_cap = cap_error(acc_true=acc_true, acc_estim=acc_estim)

                shift = qp.error.ae(test_shifted.prevalence(), train_prev)
                result = ResultRow(dataset=dataset, id=id, method=name, shift=shift, err=err_cap)
                method_dataset_results.append(asdict(result))

            report = pd.DataFrame(method_dataset_results)
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index='dataset', columns='method', values='err')
print(df)
print(pivot)
print(pivot.mean(axis=0))
