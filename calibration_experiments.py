from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from model.classifier_calibrators import *
from tqdm import tqdm
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import DistributionMatchingY
from quapy.protocol import UPP
from dataclasses import dataclass, asdict
from util import cal_error, datasets
import os
from os.path import join

"""
Methods:
- Naive (uncal) [baseline]- 
- Lascal [a proper method from calibration, for label shift]
- TransCal [a proper method from calibration, for covariate?]
- Cpcs [a proper method from calibration, for covariate?]
- HeadToTail [a proper method from calibration, for covariate and long-tailed distributions]
- HDC [a method from quantification (which works very well)]
- Missing: [a method from CAP, qué tal un LEAP?]

"""

# There should be clear whether Xva, Pva, yva is representative from P or from Q

REPEATS = 100
result_dir = f'results/calibration/label_shift/repeats_{REPEATS}'
os.makedirs(result_dir, exist_ok=True)

datasets_selected = datasets(top_length_k=10)

@dataclass
class ResultRow:
    dataset: str
    id: int
    method: str
    shift: float
    ece: float


def calibration_methods(classifier, Pva, yva, train):
    yield 'Uncal', UncalibratedWrap()
    # yield 'Platt', PlattScaling().fit(Pva, yva)
    # yield 'Isotonic', IsotonicCalibration().fit(Pva, yva)
    yield 'EM', EM(train.prevalence())
    yield 'EM-BCTS', EMBCTSCalibration()
    yield 'CPCS-S', CpcsCalibrator(prob2logits=True)
    # yield 'CPCS-P', CpcsCalibrator(prob2logits=False)
    yield 'Head2Tail-S', HeadToTailCalibrator(prob2logits=True)
    # yield 'Head2Tail-P', HeadToTailCalibrator(prob2logits=False)
    yield 'TransCal-S', TransCalCalibrator(prob2logits=True)
    # yield 'TransCal-P', TransCalCalibrator(prob2logits=False)
    yield 'LasCal-S', LasCalCalibration(prob2logits=True) #convert them to logits
    # yield 'LasCal-P', LasCalCalibration(prob2logits=False) #do not convert to logits

    dm = DistributionMatchingY(classifier=classifier, nbins=10)
    preclassified = LabelledCollection(Pva, yva)
    dm.aggregation_fit(classif_predictions=preclassified, data=val)
    yield 'HDcal', HellingerDistanceCalibration(dm)
    yield 'HDcal-sm', HellingerDistanceCalibration(dm, smooth=True)
    yield 'HDcal-sm-mono', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
    yield 'HDcal-sm-mono2', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
    yield 'HDcal-mono', HellingerDistanceCalibration(dm, smooth=False, monotonicity=True)


def calibrate(model, Xtr, ytr, Xva, Pva, yva, Xte, Pte):
    if isinstance(model, CalibratorSimple):
        return model.calibrate(Pte)
    elif isinstance(model, CalibratorSourceTarget):
        return model.calibrate(Pva, yva, Pte)
    elif isinstance(model, CalibratorCompound):
        return model.calibrate(Xtr, ytr, Xva, Pva, yva, Xte, Pte)
    else:
        raise ValueError(f'unrecognized calibrator method {model}')


all_results = []

pbar = tqdm(datasets_selected, total=len(datasets_selected))
for dataset in pbar:
    if dataset in ['ctg.1', 'spambase']:
        print('SKIPPING CTG.1, SPAMBASE')
        continue
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy

    lr = LogisticRegression()
    lr.fit(Xtr, ytr)

    Xva, yva = val.Xy
    Pva = lr.predict_proba(Xva)

    # sample generation protocol ("artificial prevalence protocol" -- generates prior probability shift)
    app = UPP(test, sample_size=len(test), repeats=REPEATS, return_type='labelled_collection')

    for name, calibrator in calibration_methods(lr, Pva, yva, train):
        result_method_dataset_path = join(result_dir, f'{name}_{dataset}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            method_dataset_results = []
            for id, test_shifted in tqdm(enumerate(app()), total=app.total(), desc=f'model={name}'):
                Xte, yte = test_shifted.Xy
                Pte = lr.predict_proba(Xte)

                Pte_cal = calibrate(calibrator, Xtr, ytr, Xva, Pva, yva, Xte, Pte)
                # print(f'Pte_cal={Pte_cal.shape}')
                # print(f'yte={yte[:10]}')
                ece_cal = cal_error(Pte_cal, yte)

                shift = qp.error.ae(test_shifted.prevalence(), train_prev)
                result = ResultRow(dataset=dataset, id=id, method=name, shift=shift, ece=ece_cal)
                method_dataset_results.append(asdict(result))

            report = pd.DataFrame(method_dataset_results)
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index='dataset', columns='method', values='ece')
print(df)
print(pivot)
print(pivot.mean(axis=0))
