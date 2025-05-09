import pandas as pd
from itertools import product

from model.classifier_calibrators import *
from tqdm import tqdm
import quapy as qp
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from dataclasses import dataclass, asdict
from util import cal_error
import os
from os.path import join
from sklearn.metrics import brier_score_loss
from model.classifier_accuracy_predictors import ATC, DoC, LEAP
from commons import REPEATS, SAMPLE_SIZE, EXPERIMENT_FOLDER, uci_datasets, new_artif_prev_protocol, classifiers


result_dir = f'results/calibration/label_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)

datasets_selected = uci_datasets(top_length_k=10)

@dataclass
class ResultRow:
    dataset: str
    id: int
    method: str
    classifier: str
    shift: float
    ece: float
    brier: float


def calibration_methods(classifier):

    # proper calibration methods
    yield 'Platt', PlattScaling().fit(Pva, yva)

    yield 'Head2Tail-P', HeadToTailCalibrator(prob2logits=False, n_components=50).fit(
        Ftr=Xtr, ytr=ytr,
        Fsrc=Xva, Zsrc=Pva, ysrc=yva
    )
    yield 'CPCS-P', CpcsCalibrator(prob2logits=False)
    yield 'TransCal-S', TransCalCalibrator(prob2logits=True)
    yield 'LasCal-P', LasCalCalibration(prob2logits=False)

    # from quantification
    yield 'EM', EMQ_Calibrator(train.prevalence())
    yield 'EM-BCTS', EMQ_BCTS_Calibrator().fit(Pva, yva)
    yield 'EMLasCal', EMQ_LasCal_Calibrator(train.prevalence(), prob2logits=True)

    yield f'HDcal8-sm-mono', DistributionMatchingCalibration(classifier, nbins=8).fit(Pva, yva)

    yield 'PACC-cal(soft)', PacCcal(Pva, yva, post_proc='softmax')

    yield 'Bin6-PACC5', Quant2Calibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva) # smooth and mono
    yield 'Bin6-EM5', Quant2Calibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva)
    yield 'Bin6-KDEy5', Quant2Calibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva)

    # from cap
    yield 'Bin2-ATC6', CAP2Calibrator(classifier=classifier, cap_method=ATC(classifier), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)
    yield 'Bin2-DoC6', CAP2Calibrator(classifier=classifier, cap_method=DoC(classifier, protocol=new_artif_prev_protocol(Xva, yva, [0, 1])), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)
    yield 'Bin2-LEAP6', CAP2Calibrator(classifier=classifier, cap_method=LEAP(classifier, KDEyML(classifier=classifier)), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)


def calibrate(model, Xtr, ytr, Xva, Pva, yva, Xte, Pte):
    if isinstance(model, CalibratorSimple):
        return model.calibrate(Pte)
    elif isinstance(model, CalibratorSourceTarget):
        return model.calibrate(Pva, yva, Pte)
    elif isinstance(model, CalibratorCompound):
        return model.calibrate(Xtr, ytr, Xva, Pva, yva, Xte, Pte)
    elif isinstance(model, CalibratorTarget):
        return model.calibrate(Xte, Pte)
    else:
        raise ValueError(f'unrecognized calibrator method {model}')


if __name__ == '__main__':
    all_results = []
    method_order = []

    n_classifiers = len([c for _,c in classifiers()])
    pbar = tqdm(product(datasets_selected, classifiers()), total=len(datasets_selected)*n_classifiers)
    for dataset, (cls_name, cls) in pbar:
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
        app = UPP(test, sample_size=SAMPLE_SIZE, repeats=REPEATS, return_type='labelled_collection')

        for name, calibrator in calibration_methods(cls):
            if name not in method_order:
                method_order.append(name)

            result_method_dataset_path = join(result_dir, f'{name}_{dataset}_{cls_name}.csv')
            if os.path.exists(result_method_dataset_path):
                report = pd.read_csv(result_method_dataset_path)
            else:
                method_dataset_results = []
                for id, test_shifted in tqdm(enumerate(app()), total=app.total(), desc=f'model={name}'):
                    Xte, yte = test_shifted.Xy
                    Pte = cls.predict_proba(Xte)

                    Pte_cal = calibrate(calibrator, Xtr, ytr, Xva, Pva, yva, Xte, Pte)
                    ece_cal = cal_error(Pte_cal, yte)
                    brier_score = brier_score_loss(y_true=yte, y_proba=Pte_cal[:,1])

                    shift = qp.error.ae(test_shifted.prevalence(), train_prev)
                    result = ResultRow(dataset=dataset, id=id, method=name, classifier=cls_name, shift=shift, ece=ece_cal, brier=brier_score)
                    method_dataset_results.append(asdict(result))

                report = pd.DataFrame(method_dataset_results)
                report.to_csv(result_method_dataset_path, index=False)

            all_results.append(report)

    df = pd.concat(all_results)

