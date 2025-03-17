from quapy.method.aggregative import KDEyML, EMQ, PCC
from quapy.protocol import NPP
from sklearn.linear_model import LogisticRegression

from model.classifier_accuracy_predictors import *
from model.classifier_calibrators import CpcsCalibrator, HeadToTailCalibrator
from util import accuracy, cap_error
import quapy as qp
from tqdm import tqdm
import numpy as np
import os
from os.path import join
from util import PrecomputedClassifier
from dataclasses import dataclass, asdict
import pandas as pd
from commons import *


result_dir = f'results/classifier_accuracy_prediction/covariate_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)


@dataclass
class ResultRow:
    dataset: str  # '{source}->{target}'
    source: str
    target: str
    id: int
    shift: float
    method: str
    classifier: str
    err: float


def cap_methods(h:BaseEstimator, setup: Setup, x_val_idx):

    Ftr = setup.train.hidden
    ytr = setup.train.labels
    
    Fva = setup.valid.hidden
    yva = setup.valid.labels
    

    # CAP methods
    yield 'Naive', NaiveIID(classifier=h).fit(x_val_idx, yva)
    yield 'ATC', ATC(h).fit(x_val_idx, yva)
    val_lc = LabelledCollection(x_val_idx, yva)
    val_prot = NPP(val_lc, sample_size=SAMPLE_SIZE, repeats=REPEATS, random_state=0, return_type='labelled_collection')
    yield 'DoC', DoC(h, protocol=val_prot).fit(x_val_idx, yva)
    yield 'LEAP', LEAP(classifier=h, q_class=KDEyML(classifier=h)).fit(x_val_idx, yva)
    yield 'LEAP-PCC', LEAP(classifier=h, q_class=PCC(classifier=h)).fit(x_val_idx, yva)

    # Calibration 2 CAP
    yield 'TransCal-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=TransCalCalibrator, probs2logits=True, Ftr=Ftr, ytr=ytr).fit(x_val_idx, yva, hidden=Fva)
    yield 'TransCal-a-P', CalibratorCompound2CAP(classifier=h, calibrator_cls=TransCalCalibrator, probs2logits=False, Ftr=Ftr, ytr=ytr).fit(x_val_idx, yva, hidden=Fva)
    yield 'Cpcs-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=CpcsCalibrator, probs2logits=True, Ftr=Ftr, ytr=ytr).fit(x_val_idx, yva, hidden=Fva)
    yield 'Cpcs-a-P', CalibratorCompound2CAP(classifier=h, calibrator_cls=CpcsCalibrator, probs2logits=False, Ftr=Ftr, ytr=ytr).fit(x_val_idx, yva, hidden=Fva)
    #yield 'Head2Tail-a-S', CalibratorCompound2CAP(classifier=h, calibrator_cls=HeadToTailCalibrator, probs2logits=True, Ftr=Xtr, ytr=ytr).fit(x_val_idx, yva)
    ## yield 'Head2Tail-a-P', CalibratorCompound2CAP(classifier=h, calibrator_cls=HeadToTailCalibrator, probs2logits=False, Ftr=Xtr, ytr=ytr).fit(Xva, yva)
    yield 'LasCal-a-S', LasCal2CAP(classifier=h, probs2logits=True).fit(x_val_idx, yva)
    yield 'LasCal-a-P', LasCal2CAP(classifier=h, probs2logits=False).fit(x_val_idx, yva)
    ## yield 'HDc-a', HDC2CAP(classifier=h).fit(Xva, yva)

    # Quantification 2 CAP
    yield 'PCC-a', Quant2CAP(classifier=h, quantifier_class=PCC).fit(x_val_idx, yva)
    yield 'PACC-a', Quant2CAP(classifier=h, quantifier_class=PACC).fit(x_val_idx, yva)
    yield 'KDEy-a', Quant2CAP(classifier=h, quantifier_class=KDEyML).fit(x_val_idx, yva)
    yield 'EMQ-a', Quant2CAP(classifier=h, quantifier_class=EMQ).fit(x_val_idx, yva)
    yield 'EMQ-BCTS-a', Quant2CAP(classifier=h, quantifier_class=EMQ.EMQ_BCTS).fit(x_val_idx, yva)



all_results = []
methods_order = []

pbar = tqdm(iterate_datasets_covariate_shift(), total=total_setups_covariate_shift)
for setup in pbar:
    description = f'[{setup.model}]::{setup.source}->{setup.target}'

    h = PrecomputedClassifier()
    x_val_idx = h.feed(X=setup.valid.hidden, P=setup.valid.posteriors, L=setup.valid.logits)
    
    for name, cap_method in cap_methods(h, setup, x_val_idx):
        if name not in methods_order:
            methods_order.append(name)
        
        result_method_setup_path = join(result_dir, f'{name}_{setup.model}_{setup.source}__{setup.target}.csv')
        if os.path.exists(result_method_setup_path):
            report = pd.read_csv(result_method_setup_path)
        else:
            method_setup_results = []
            err_ave = []
            for idx, (test_sample, shift) in  enumerate(yield_random_samples(setup.in_test, setup.out_test, repeats=REPEATS, samplesize=SAMPLE_SIZE)):
                x_test_idx = h.feed(X=test_sample.hidden, P=test_sample.posteriors, L=test_sample.logits)

                yte = test_sample.labels
                y_pred = h.predict(x_test_idx)
                acc_true = accuracy(y_true=yte, y_pred=y_pred)

                acc_estim = cap_method.predict(x_test_idx, hidden=test_sample.hidden)
                err_cap = cap_error(acc_true=acc_true, acc_estim=acc_estim)
                err_ave.append(err_cap)
                pbar.set_description(description + f' {name} ({idx}/{REPEATS}) AE-ave={np.mean(err_ave):.5f}')

                result = ResultRow(
                    dataset=f'{setup.source}->{setup.target}',
                    source=setup.source,
                    target=setup.target,
                    id=idx,
                    shift=shift,
                    method=name,
                    classifier=setup.model,
                    err=err_cap)
                method_setup_results.append(asdict(result))

            report = pd.DataFrame(method_setup_results)
            report.to_csv(result_method_setup_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index='dataset', columns='method', values='err')
print(df)
print(pivot)
print(pivot.mean(axis=0))

from new_table import LatexTable

tables = []
for classifier_name in models:
    df_h = df[df['classifier']==classifier_name]
    table = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='err')
    table.name = f'cap_cv_AE_{classifier_name}'
    table.reorder_methods(methods_order)
    table.format.configuration.show_std=False
    table.format.configuration.side_columns = True
    tables.append(table)
LatexTable.LatexPDF('./tables/cap_covariate_shift.pdf', tables=tables)
