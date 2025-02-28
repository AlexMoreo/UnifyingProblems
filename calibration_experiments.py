from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from itertools import product

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
from sklearn.metrics import brier_score_loss

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
    classifier: str
    shift: float
    ece: float
    brier: float


def calibration_methods(classifier, Pva, yva, train):
    yield 'Uncal', UncalibratedWrap()
    yield 'Platt', PlattScaling().fit(Pva, yva)
    # yield 'Isotonic', IsotonicCalibration().fit(Pva, yva)
    yield 'EM', EM(train.prevalence())
    # yield 'EM-BCTS', EMBCTSCalibration()
    yield 'CPCS-S', CpcsCalibrator(prob2logits=True)
    # yield 'CPCS-P', CpcsCalibrator(prob2logits=False)
    yield 'Head2Tail-S', HeadToTailCalibrator(prob2logits=True)
    # yield 'Head2Tail-P', HeadToTailCalibrator(prob2logits=False)
    yield 'TransCal-S', TransCalCalibrator(prob2logits=True)
    # yield 'TransCal-P', TransCalCalibrator(prob2logits=False)
    yield 'LasCal-S', LasCalCalibration(prob2logits=True) #convert them to logits
    # yield 'LasCal-P', LasCalCalibration(prob2logits=False) #do not convert to logits

    for nbins in [8, 20]: #, 25, 30, 35, 40]:
        dm = DistributionMatchingY(classifier=classifier, nbins=nbins)
        preclassified = LabelledCollection(Pva, yva)
        dm.aggregation_fit(classif_predictions=preclassified, data=val)
        yield f'HDcal{nbins}', HellingerDistanceCalibration(dm)
        yield f'HDcal{nbins}-sm', HellingerDistanceCalibration(dm, smooth=True)
        yield f'HDcal{nbins}-sm-mono', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
        # yield f'HDcal{nbins}-sm-mono-wrong', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True, postsmooth=True)
        yield f'HDcal{nbins}-mono', HellingerDistanceCalibration(dm, smooth=False, monotonicity=True)
    yield 'PACC-cal', PACCcal(Pva, yva)
    yield 'PACC-cal(soft)', PACCcal(Pva, yva, post_proc='softmax')
    yield 'NaiveUncertain', NaiveUncertain()
    yield 'NaiveTrain', NaiveUncertain(train_prev)


def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    # yield 'dt', DecisionTreeClassifier()
    # yield 'svm', SVC()
    yield 'mlp', MLPClassifier()


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

n_classifiers = len([c for _,c in classifiers()])
pbar = tqdm(product(datasets_selected, classifiers()), total=len(datasets_selected)*n_classifiers)
for dataset, (cls_name, cls) in pbar:
    if dataset in ['ctg.1', 'spambase', 'yeast']:
        print('SKIPPING CTG.1, SPAMBASE, YEAST')
        continue
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy
    cls.fit(Xtr, ytr)

    Xva, yva = val.Xy
    Pva = cls.predict_proba(Xva)

    # sample generation protocol ("artificial prevalence protocol" -- generates prior probability shift)
    app = UPP(test, sample_size=len(test), repeats=REPEATS, return_type='labelled_collection')

    for name, calibrator in calibration_methods(cls, Pva, yva, train):
        result_method_dataset_path = join(result_dir, f'{name}_{dataset}_{cls_name}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            method_dataset_results = []
            for id, test_shifted in tqdm(enumerate(app()), total=app.total(), desc=f'model={name}'):
                Xte, yte = test_shifted.Xy
                Pte = cls.predict_proba(Xte)

                Pte_cal = calibrate(calibrator, Xtr, ytr, Xva, Pva, yva, Xte, Pte)
                # print(f'Pte_cal={Pte_cal.shape}')
                # print(f'yte={yte[:10]}')
                ece_cal = cal_error(Pte_cal, yte)
                brier_score = brier_score_loss(y_true=yte, y_proba=Pte_cal[:,1])

                shift = qp.error.ae(test_shifted.prevalence(), train_prev)
                result = ResultRow(dataset=dataset, id=id, method=name, classifier=cls_name, shift=shift, ece=ece_cal, brier=brier_score)
                method_dataset_results.append(asdict(result))

            report = pd.DataFrame(method_dataset_results)
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index=['classifier'], columns='method', values='ece')
# print(df)
print('ECE')
print(pivot)
# print(pivot.mean(axis=0))

pivot = df.pivot_table(index=['classifier'], columns='method', values='brier')
# print(df)
print('Brier Score')
print(pivot)
# print(pivot.mean(axis=0))

# df['cal']=df['ece']*df['brier']
# pivot = df.pivot_table(index=['classifier'], columns='method', values='cal')
# print(pivot)
# print(pivot.mean(axis=0))


