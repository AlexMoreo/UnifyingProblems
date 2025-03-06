from itertools import product
import os
from dataclasses import dataclass
from quapy.protocol import NaturalPrevalenceProtocol
import torch
from os.path import join
import numpy as np
from sklearn.metrics import brier_score_loss
from sympy.multipledispatch.dispatcher import source

from model.classifier_calibrators import *
from util import cal_error
from scipy.special import softmax


REPEATS = 100
SAMPLESIZE = 250
result_dir = f'results/calibration/covariate_shift/repeats_{REPEATS}_samplesize={SAMPLESIZE}'
os.makedirs(result_dir, exist_ok=True)


@dataclass
class ResultRow:
    dataset: str  # '{source}->{target}'
    source: str
    target: str
    id: int
    method: str
    classifier: str
    # shift: float
    ece: float
    brier: float

@dataclass
class Dataset:
    hidden: np.ndarray
    logits: np.ndarray
    labels: np.ndarray

@dataclass
class Setup:
    model: str
    source: str
    target: str
    train: Dataset
    valid: Dataset
    test: Dataset

sentiment_datasets = ['imdb', 'rt', 'yelp']
models = ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base']

def calibrators():
    # valid_posteriors = softmax(valid_logits, axis=1)
    # test_posteriors = softmax(test_logits, axis=1)
    # yield 'EM', EM(train_prevalence=np.mean(train_y))
    # yield 'EM', PACCcal(softmax(valid_logits, axis=1), valid_y)
    yield 'TransCal', TransCalCalibrator(prob2logits=False)
    # yield 'Head2Tail', HeadToTailCalibrator(prob2logits=False)
    yield 'CPCS', CpcsCalibrator(prob2logits=False)
    yield 'LasCal', LasCalCalibration(prob2logits=False)

def get_calibrated_posteriors(calibrator, train, valid, test):
    if isinstance(calibrator, CalibratorCompound):
        calib_posteriors = calibrator.calibrate(
            Ftr=train.hidden, ytr=train.labels,
            Fsrc=valid.hidden, Zsrc=valid.logits, ysrc=valid.labels,
            Ftgt=test.hidden, Ztgt=test.logits
        )
    elif isinstance(calibrator, CalibratorSourceTarget):
        calib_posteriors = calibrator.calibrate(
            Zsrc=valid.logits, ysrc=valid.labels, Ztgt=test.logits
        )
    elif isinstance(calibrator, CalibratorSimple):
        calib_posteriors = calibrator.calibrate(P=softmax(test.logits, axis=1))

    return calib_posteriors





def iterate_datasets():

    def load_dataset(path, domain, splitname):
        hidden = torch.load(join(path, f'{domain}.{splitname}.hidden_states.pt')).numpy()
        logits = torch.load(join(path, f'{domain}.{splitname}.logits.pt')).numpy()
        labels = torch.load(join(path, f'{domain}.{splitname}.labels.pt')).numpy()
        return Dataset(hidden=hidden, logits=logits, labels=labels)

    for source in sentiment_datasets:
        for model in models:
            path = f'./neural_training/embeds/{source}/{model}'

            train = load_dataset(path, 'source', 'train')
            valid = load_dataset(path, 'source', 'validation')

            for target in sentiment_datasets:
                if target == source:
                    target_prefix = 'source'
                else:
                    target_prefix = f'target_{target}'

                test = load_dataset(path, target_prefix, 'test')

                yield Setup(model=model, source=source, target=target, train=train, valid=valid, test=test)


def yield_random_samples(test: Dataset, repeats, samplesize):
    np.random.seed(0)
    indexes = []
    test_length = len(test.labels)
    for _ in range(repeats):
        indexes.append(np.random.choice(test_length, size=samplesize, replace=True))
    for index in indexes:
        sample_hidden = test.hidden[index]
        sample_logits = test.logits[index]
        sample_labels = test.labels[index]
        yield Dataset(hidden=sample_hidden, logits=sample_logits, labels=sample_labels)


all_results = []
method_order = []

for setup in iterate_datasets():

    for idx, test_sample in enumerate(yield_random_samples(setup.test, repeats=REPEATS, samplesize=SAMPLESIZE)):

        print(f'[{setup.model}]::{setup.source}->{setup.target}')

        for cal_name, calibrator in calibrators():

            calib_posteriors = get_calibrated_posteriors(calibrator, setup.train, setup.valid, test_sample)

            ece = cal_error(calib_posteriors, test_sample.labels, arelogits=False)
            brier_score = brier_score_loss(y_true=test_sample.labels, y_proba=calib_posteriors[:, 1])

            print(f'\t{cal_name} got ECE={ece:.5f}')
            result = ResultRow(
                dataset=f'{setup.source}->{setup.target}',
                source=setup.source,
                target=setup.target,
                id=idx,
                method=cal_name,
                ece=ece,
                brier=brier_score
            )






