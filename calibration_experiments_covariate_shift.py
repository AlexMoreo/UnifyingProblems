from itertools import product
import os
from dataclasses import dataclass, asdict
from quapy.protocol import NaturalPrevalenceProtocol
import torch
from os.path import join
import numpy as np
from sklearn.metrics import brier_score_loss
from sympy.multipledispatch.dispatcher import source
import pandas as pd
from tqdm import tqdm

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
    posteriors: np.ndarray
    labels: np.ndarray
    prevalence: np.ndarray

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


def calibrators(setup):
    # valid_posteriors = softmax(valid_logits, axis=1)
    # test_posteriors = softmax(test_logits, axis=1)
    yield 'EM', EM(train_prevalence=setup.train.prevalence)
    # yield 'EM', PACCcal(softmax(valid_logits, axis=1), valid_y)
    yield 'TransCal', TransCalCalibrator(prob2logits=False)
    #yield 'Head2Tail', HeadToTailCalibrator(prob2logits=False)
    yield 'CPCS', CpcsCalibrator(prob2logits=False)
    yield 'LasCal', LasCalCalibration(prob2logits=False)

    #for nbins in [8]: #20, 25, 30, 35, 40]:
    #    dm = DistributionMatchingY(classifier=classifier, nbins=nbins)
    #    preclassified = LabelledCollection(Pva, yva)
    #    dm.aggregation_fit(classif_predictions=preclassified, data=val)
        # yield f'HDcal{nbins}', HellingerDistanceCalibration(dm)
        # yield f'HDcal{nbins}-sm', HellingerDistanceCalibration(dm, smooth=True)
    #    yield f'HDcal{nbins}-sm-mono', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
        # yield f'HDcal{nbins}-sm-mono-wrong', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True, postsmooth=True)
        # yield f'HDcal{nbins}-mono', HellingerDistanceCalibration(dm, smooth=False, monotonicity=True)

    Pva = setup.valid.posteriors
    yva = setup.valid.labels
    yield 'PACC-cal', PACCcal(Pva, yva)
    yield 'PACC-cal(soft)', PACCcal(Pva, yva, post_proc='softmax')
    yield 'PACC-cal(soft)2', PACCcal(Pva, yva, post_proc='softmax')
    yield 'PACC-cal(log)', PACCcal(Pva, yva, post_proc='logistic')
    yield 'PACC-cal(log)2', PACCcal(Pva, yva, post_proc='logistic')
    yield 'PACC-cal(iso)', PACCcal(Pva, yva, post_proc='isotonic')
    yield 'PACC-cal(iso)2', PACCcal(Pva, yva, post_proc='isotonic')

    # yield 'Bin-PACC', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC).fit(Pva, yva)
    # yield 'Bin-PACC2', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=2).fit(Pva, yva)
    # yield 'Bin-PACC5', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5).fit(Pva, yva)
    #yield 'Bin2-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5).fit(Xva, yva)
    # yield 'Bin-EM', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ).fit(Pva, yva)
    # yield 'Bin-EM2', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=2).fit(Pva, yva)
    # yield 'Bin-EM5', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5).fit(Pva, yva)
    #yield 'Bin2-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5).fit(Xva, yva)
    #yield 'Bin2-DM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=DistributionMatchingY, nbins=5).fit(Xva, yva)


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

    def load_dataset(path, domain, splitname, reduce=None, random_seed=0):
        hidden = torch.load(join(path, f'{domain}.{splitname}.hidden_states.pt')).numpy()
        logits = torch.load(join(path, f'{domain}.{splitname}.logits.pt')).numpy()
        labels = torch.load(join(path, f'{domain}.{splitname}.labels.pt')).numpy()
        if reduce is not None and isinstance(reduce,int) and reduce<len(labels):
            np.random.seed(random_seed)
            sel_idx = np.random.choice(reduce, size=reduce, replace=False)
            hidden = hidden[sel_idx]
            logits = logits[sel_idx]
            labels = labels[sel_idx]
        posteriors = softmax(logits, axis=1)
        prevalence = F.prevalence_from_labels(labels, classes=[0,1])
        return Dataset(hidden=hidden, logits=logits, posteriors=posteriors, prevalence=prevalence, labels=labels)

    for source in sentiment_datasets:
        for model in models:
            path = f'./neural_training/embeds/{source}/{model}'

            train = load_dataset(path, 'source', 'train', reduce=5000)
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
        sample_posteriors = test.posteriors[index]
        sample_prevalence = F.prevalence_from_labels(sample_labels, classes=[0,1])
        yield Dataset(hidden=sample_hidden, logits=sample_logits, labels=sample_labels, posteriors=sample_posteriors, prevalence=sample_prevalence)


total_setups = len(models)*(len(sentiment_datasets)**2)
all_results = []
method_order = []

pbar = tqdm(iterate_datasets(), total=total_setups)
for setup in pbar:
    description = f'[{setup.model}]::{setup.source}->{setup.target}'

    for cal_name, calibrator in calibrators(setup):
        result_method_setup_path = join(result_dir, f'{cal_name}_{setup.model}_{setup.source}__{setup.target}.csv')
        if os.path.exists(result_method_setup_path):
            report = pd.read_csv(result_method_setup_path)
        else:
            method_setup_results = []
            ece_ave = []
            for idx, test_sample in enumerate(yield_random_samples(setup.test, repeats=REPEATS, samplesize=SAMPLESIZE)):
                calib_posteriors = get_calibrated_posteriors(calibrator, setup.train, setup.valid, test_sample)

                ece = cal_error(calib_posteriors, test_sample.labels, arelogits=False)
                brier_score = brier_score_loss(y_true=test_sample.labels, y_proba=calib_posteriors[:, 1])
                ece_ave.append(ece)
                pbar.set_description(description + f' {cal_name} ({idx}/{REPEATS}) ECE-ave={np.mean(ece_ave):.5f}')

                result = ResultRow(
                    dataset=f'{setup.source}->{setup.target}',
                    source=setup.source,
                    target=setup.target,
                    id=idx,
                    method=cal_name,
                    classifier=setup.model,
                    ece=ece,
                    brier=brier_score
                )

                method_setup_results.append(asdict(result))

            report = pd.DataFrame(method_setup_results)
            report.to_csv(result_method_setup_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index=['classifier', 'dataset'], columns='method', values='ece')
print('ECE')
print(pivot)

pivot = df.pivot_table(index=['classifier', 'dataset'], columns='method', values='brier')
print('Brier score')
print(pivot)







