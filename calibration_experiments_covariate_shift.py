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
from quapy.method.aggregative import KDEyML
from quapy.method.aggregative import PCC
from model.classifier_calibrators import *
from model.classifier_accuracy_predictors import ATC, DoC, LEAP
from util import cal_error, PrecomputedClassifier
from scipy.special import softmax
from commons import *


result_dir = f'results/calibration/covariate_shift/{EXPERIMENT_FOLDER}'
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
    ece: float
    brier: float



def calibrators(setup):
    Pva = setup.valid.posteriors
    yva = setup.valid.labels

    # proper calibration methods
    yield 'Platt', PlattScaling().fit(Pva, yva)

    # yield '?', PACCcal(softmax(valid_logits, axis=1), valid_y)
    yield 'TransCal', TransCalCalibrator(prob2logits=False)
    yield 'Head2Tail', HeadToTailCalibrator(prob2logits=True, n_components=50).fit(
        Ftr=setup.train.hidden, ytr=setup.train.labels,
        Fsrc=setup.valid.hidden, Zsrc=setup.valid.posteriors, ysrc=setup.valid.labels
    )
    yield 'CPCS', CpcsCalibrator(prob2logits=False)
    yield 'LasCal', LasCalCalibration(prob2logits=False)

    yield 'EM', EM(train_prevalence=setup.train.prevalence)
    yield 'EM-BCTS', EMQ_BCTS_Calibrator().fit(Pva, yva)
    yield 'EM-TransCal', EMTransCal(train_prevalence=setup.train.prevalence, prob2logits=False)

    h = PrecomputedClassifier()
    val_idx = h.feed(X=setup.valid.hidden, P=setup.valid.posteriors, L=setup.valid.logits)

    for nbins in [8]: #20, 25, 30, 35, 40]:
        dm = DistributionMatchingY(classifier=h, nbins=nbins)
        preclassified = LabelledCollection(Pva, yva)
        dm.aggregation_fit(classif_predictions=preclassified, data=None)
        # yield f'HDcal{nbins}', HellingerDistanceCalibration(dm)
        # yield f'HDcal{nbins}-sm', HellingerDistanceCalibration(dm, smooth=True)
        yield f'HDcal{nbins}-sm-mono', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
        # yield f'HDcal{nbins}-sm-mono-wrong', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True, postsmooth=True)
        # yield f'HDcal{nbins}-mono', HellingerDistanceCalibration(dm, smooth=False, monotonicity=True)
    

    # yield 'PACC-cal(clip)', PACCcal(Pva, yva, post_proc='clip')
    yield 'PACC-cal(soft)', PACCcal(Pva, yva, post_proc='softmax')
    # yield 'PACC-cal(log)', PACCcal(Pva, yva, post_proc='logistic')
    # yield 'PACC-cal(iso)', PACCcal(Pva, yva, post_proc='isotonic')

    yield 'Bin6-PCC5', QuantifyCalibrator(classifier=h, quantifier_cls=PCC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono
    yield 'Bin6-PACC5', QuantifyCalibrator(classifier=h, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono
    yield 'Bin6-EM5', QuantifyCalibrator(classifier=h, quantifier_cls=EMQ, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva)
    yield 'Bin6-KDEy5', QuantifyCalibrator(classifier=h, quantifier_cls=KDEyML, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva)

    yield 'Bin2-ATC6', CAPCalibrator(classifier=h, cap_method=ATC(h), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)

    def new_labelshift_protocol(X, y, classes):
        lc = LabelledCollection(X, y, classes=classes)
        app = NaturalPrevalenceProtocol(
            lc,
            sample_size=SAMPLE_SIZE,
            repeats=REPEATS,
            return_type='labelled_collection',
            random_state=0
        )
        return app


    yield 'Bin2-DoC6', CAPCalibrator(classifier=h, cap_method=DoC(h, protocol=new_labelshift_protocol(val_idx,yva,[0,1])), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin2-LEAP6', CAPCalibrator(classifier=h, cap_method=LEAP(h, KDEyML(classifier=h)), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin2-LEAP-PCC-6', CAPCalibrator(classifier=h, cap_method=LEAP(h, PCC(classifier=h)), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)



def get_calibrated_posteriors(calibrator, train, valid, test):
    if isinstance(calibrator, CalibratorCompound): # working with logits
        calib_posteriors = calibrator.calibrate(
            Ftr=train.hidden, ytr=train.labels,
            Fsrc=valid.hidden, Zsrc=valid.logits, ysrc=valid.labels,
            Ftgt=test.hidden, Ztgt=test.logits
        )
    elif isinstance(calibrator, CalibratorSourceTarget): # working with logits
        calib_posteriors = calibrator.calibrate(
            Zsrc=valid.logits, ysrc=valid.labels, Ztgt=test.logits
        )
    elif isinstance(calibrator, CalibratorTarget): # working with posteriors
        test_X_idx = calibrator.classifier.feed(X=test.hidden, P=test.posteriors, L=test.logits)
        calib_posteriors = calibrator.calibrate(
            Ftgt=test_X_idx, Ztgt=test.posteriors
        )
    elif isinstance(calibrator, CalibratorSimple): # working with posteriors
        calib_posteriors = calibrator.calibrate(test.posteriors)

    return calib_posteriors


all_results = []
method_order = []

pbar = tqdm(iterate_datasets_covariate_shift(), total=total_setups_covariate_shift)
for setup in pbar:
    description = f'[{setup.model}]::{setup.source}->{setup.target}'

    for cal_name, calibrator in calibrators(setup):
        if cal_name not in method_order:
            method_order.append(cal_name)

        result_method_setup_path = join(result_dir, f'{cal_name}_{setup.model}_{setup.source}__{setup.target}.csv')
        if os.path.exists(result_method_setup_path):
            report = pd.read_csv(result_method_setup_path)
        else:
            method_setup_results = []
            ece_ave = []
            for idx, (test_sample, shift) in enumerate(yield_random_samples(setup.in_test, setup.out_test, repeats=REPEATS, samplesize=SAMPLE_SIZE)):
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
                    shift=shift,
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

from new_table import LatexTable

tables = []
for classifier_name in models:
    df_h = df[df['classifier']==classifier_name]
    table_ece = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='ece')
    table_ece.name = f'calibration_cv_ECE_{classifier_name}'
    table_ece.reorder_methods(method_order)
    table_ece.format.configuration.show_std=False
    table_ece.format.configuration.side_columns = True
    tables.append(table_ece)

    table_brier = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='brier')
    table_brier.name = f'calibration_cv_brier_{classifier_name}'
    table_brier.reorder_methods(method_order)
    table_brier.format.configuration.show_std = False
    table_brier.format.configuration.side_columns = True
    tables.append(table_brier)

LatexTable.LatexPDF(f'./tables/calibration_covariate_shift.pdf', tables)







