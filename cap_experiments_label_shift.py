import pickle

from quapy.method.aggregative import KDEyML, EMQ
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from model.classifier_accuracy_predictors import *
from model.classifier_calibrators import CpcsCalibrator, HeadToTailCalibrator
from itertools import product
from util import accuracy, cap_error
import quapy as qp
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from dataclasses import dataclass, asdict
import pandas as pd
from commons import REPEATS, SAMPLE_SIZE, EXPERIMENT_FOLDER, uci_datasets, new_artif_prev_protocol

result_dir = f'results/classifier_accuracy_prediction/label_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)


datasets_selected = uci_datasets(top_length_k=10)

@dataclass
class ResultRow:
    dataset: str
    id: int
    method: str
    classifier: str
    shift: float
    err: float
    true_acc: float
    estim_acc: float


def cap_methods(h:BaseEstimator, Xva, yva):

    # CAP methods
    yield 'Naive', NaiveIID(classifier=h).fit(Xva, yva)
    yield 'ATC', ATC(h).fit(Xva, yva)
    yield 'DoC', DoC(h, protocol=new_artif_prev_protocol(*val.Xy)).fit(Xva, yva)
    yield 'LEAP', LEAP(classifier=h, q_class=KDEyML(classifier=h)).fit(Xva, yva)

    # Calibration 2 CAP
    yield 'Cpcs-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=CpcsCalibrator, probs2logits=True, Ftr=Xtr, ytr=ytr).fit(Xva, yva)
    yield 'TransCal-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=TransCalCalibrator, probs2logits=True, Ftr=Xtr, ytr=ytr).fit(Xva, yva)
    yield 'LasCal-a-P', LasCal2CAP(classifier=h, probs2logits=False).fit(Xva, yva)

    yield 'HDc-a-sm-mono', DMCal2CAP(classifier=h).fit(Xva, yva)

    # Quantification 2 CAP
    yield 'PACC-a', Quant2CAP(classifier=h, quantifier_class=PACC).fit(Xva, yva)
    yield 'EMQ-a', Quant2CAP(classifier=h, quantifier_class=EMQ).fit(Xva, yva)
    yield 'KDEy-a', Quant2CAP(classifier=h, quantifier_class=KDEyML).fit(Xva, yva)


def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    yield 'mlp', MLPClassifier()


all_results = []
methods_order = []

n_classifiers = len([c for _,c in classifiers()])
pbar = tqdm(product(datasets_selected, classifiers()), total=len(datasets_selected)*n_classifiers)
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
                result = ResultRow(dataset=dataset, id=id, method=name, classifier=cls_name, shift=shift, err=err_cap, true_acc=acc_true, estim_acc=acc_estim)
                method_dataset_results.append(asdict(result))

            report = pd.DataFrame(method_dataset_results)
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)

from new_table import LatexTable

tables = []
for classifier_name, _ in classifiers():
    df_h = df[df['classifier']==classifier_name]
    print(df_h)
    table = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='err')
    table.name = f'cap_pps_{classifier_name}'
    table.reorder_methods(methods_order)
    table.format.configuration.show_std=False
    table.format.configuration.side_columns = True
    tables.append(table)


LatexTable.LatexPDF(f'./tables/cap_label_shift.pdf', tables)
