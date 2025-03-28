from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from itertools import product

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import util
from model.classifier_calibrators import *
from tqdm import tqdm
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import DistributionMatchingY, KDEyML
from quapy.protocol import UPP, ArtificialPrevalenceProtocol
from dataclasses import dataclass, asdict
from util import cal_error
import os
from os.path import join
from sklearn.metrics import brier_score_loss
from model.classifier_accuracy_predictors import ATC, DoC, LEAP
from commons import REPEATS, SAMPLE_SIZE, EXPERIMENT_FOLDER, uci_datasets

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
    yield 'Uncal', UncalibratedWrap()
    # yield 'NaiveUncertain', NaiveUncertain()
    # yield 'NaiveTrain', NaiveUncertain(train_prev)

    # proper calibration methods
    yield 'Platt', PlattScaling().fit(Pva, yva)
    yield 'Isotonic', IsotonicCalibration().fit(Pva, yva)
    yield 'CPCS-S', CpcsCalibrator(prob2logits=True)
    yield 'CPCS-P', CpcsCalibrator(prob2logits=False)
    # yield 'Head2Tail-S', HeadToTailCalibrator(prob2logits=True).fit(
    #     Ftr=Xtr, ytr=ytr,
    #     Fsrc=Xva, Zsrc=Pva, ysrc=yva
    # )
    # yield 'Head2Tail-P', HeadToTailCalibrator(prob2logits=False)
    yield 'TransCal-S', TransCalCalibrator(prob2logits=True)
    yield 'TransCal-P', TransCalCalibrator(prob2logits=False)
    yield 'LasCal-S', LasCalCalibration(prob2logits=True) #convert them to logits
    yield 'LasCal-P', LasCalCalibration(prob2logits=False) #do not convert to logits

    # from quantification
    yield 'EM', EM(train.prevalence())
    # yield 'EM-BCTS', EMBCTSCalibration()
    for nbins in [8]: #20, 25, 30, 35, 40]:
        dm = DistributionMatchingY(classifier=classifier, nbins=nbins)
        preclassified = LabelledCollection(Pva, yva)
        dm.aggregation_fit(classif_predictions=preclassified, data=val)
        # yield f'HDcal{nbins}', HellingerDistanceCalibration(dm)
        # yield f'HDcal{nbins}-sm', HellingerDistanceCalibration(dm, smooth=True)
        yield f'HDcal{nbins}-sm-mono', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True)
        # yield f'HDcal{nbins}-sm-mono-wrong', HellingerDistanceCalibration(dm, smooth=True, monotonicity=True, postsmooth=True)
        # yield f'HDcal{nbins}-mono', HellingerDistanceCalibration(dm, smooth=False, monotonicity=True)
    yield 'PACC-cal(clip)', PACCcal(Pva, yva)
    yield 'PACC-cal(soft)', PACCcal(Pva, yva, post_proc='softmax')
    # yield 'PACC-cal(log)', PACCcal(Pva, yva, post_proc='logistic')
    # yield 'PACC-cal(iso)', PACCcal(Pva, yva, post_proc='isotonic')

    # yield 'Bin-PACC', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC).fit(Pva, yva)
    # yield 'Bin-PACC2', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=2).fit(Pva, yva)
    # yield 'Bin-PACC5', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5).fit(Pva, yva)
    #yield 'Bin2-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5).fit(Xva, yva)
    #yield 'Bin3-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5).fit(Xva, yva) # with interpolation
    #yield 'Bin4-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, dedicated=True).fit(Xva, yva) # dedicated quantifiers for bin
    #yield 'Bin5-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, dedicated=False).fit(Xva, yva) # fill nans via interpolation
    yield 'Bin6-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva) # smooth and mono
    #yield 'Bin6-PACC10', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=10, smooth=True, monotonicity=True).fit(Xva, yva)  # smooth and mono
    #yield 'Bin7-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva) # isodense
    #yield 'Bin9-PACC5', QuantifyCalibrator(classifier=classifier, quantifier_cls=PACC, nbins=5, dedicated=True, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva) # dedicated but fit_classifier=False
    
    # yield 'Bin-EM', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ).fit(Pva, yva)
    # yield 'Bin-EM2', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=2).fit(Pva, yva)
    # yield 'Bin-EM5', QuantifyBinsCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5).fit(Pva, yva)
    #yield 'Bin2-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5).fit(Xva, yva)
    #yield 'Bin3-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5).fit(Xva, yva)
    #yield 'Bin4-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, dedicated=True).fit(Xva, yva)
    #yield 'Bin5-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, dedicated=False).fit(Xva, yva)
    yield 'Bin6-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva)
    #yield 'Bin6-EM10', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=10, smooth=True, monotonicity=True).fit(Xva, yva)
    #yield 'Bin7-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva)
    #yield 'Bin9-EM5', QuantifyCalibrator(classifier=classifier, quantifier_cls=EMQ, nbins=5, dedicated=True, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva) # dedicated but fit_classifier=False


    #yield 'Bin2-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5).fit(Xva, yva)
    #yield 'Bin3-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5).fit(Xva, yva)
    #yield 'Bin4-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, dedicated=True).fit(Xva, yva)
    #yield 'Bin5-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, dedicated=False).fit(Xva, yva)
    yield 'Bin6-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, smooth=True, monotonicity=True).fit(Xva, yva)
    #yield 'Bin6-KDEy10', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=10, smooth=True, monotonicity=True).fit(Xva, yva)
    #yield 'Bin7-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva)
    #yield 'Bin9-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, dedicated=True, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva) # dedicated but fit_classifier=False
    #yield 'Bin10-KDEy5', QuantifyCalibrator(classifier=classifier, quantifier_cls=KDEyML, nbins=5, dedicated=True, smooth=True, monotonicity=True, isometric=False).fit(Xva, yva) # dedicated but fit_classifier=False

    # from cap
    #yield 'Bin-ATC6', CAPCalibrator(classifier=classifier, cap_method=ATC(classifier), nbins=6).fit(Xva, yva)
    yield 'Bin2-ATC6', CAPCalibrator(classifier=classifier, cap_method=ATC(classifier), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)

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

    #yield 'Bin-DoC6', CAPCalibrator(classifier=classifier, cap_method=DoC(classifier, protocol=new_labelshift_protocol(Xva,yva,[0,1])), nbins=6).fit(Xva, yva)
    yield 'Bin2-DoC6', CAPCalibrator(classifier=classifier, cap_method=DoC(classifier, protocol=new_labelshift_protocol(Xva,yva,[0,1])), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)
    #yield 'Bin2-DoC12', CAPCalibrator(classifier=classifier, cap_method=DoC(classifier, protocol=new_labelshift_protocol(Xva, yva, [0, 1])), nbins=12, monotonicity=True, smooth=True).fit(Xva, yva)
    
    #yield 'Bin-LEAP6', CAPCalibrator(classifier=classifier, cap_method=LEAP(classifier, KDEyML(classifier=classifier)), nbins=6).fit(Xva, yva)
    yield 'Bin2-LEAP6', CAPCalibrator(classifier=classifier, cap_method=LEAP(classifier, KDEyML(classifier=classifier)), nbins=6, monotonicity=True, smooth=True).fit(Xva, yva)
    #yield 'Bin2-LEAP12', CAPCalibrator(classifier=classifier, cap_method=LEAP(classifier, KDEyML(classifier=classifier)), nbins=12, monotonicity=True, smooth=True).fit(Xva, yva)



def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    yield 'mlp', MLPClassifier()


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

    from new_table import LatexTable

    tables = []
    for classifier_name, _ in classifiers():
        df_h = df[df['classifier']==classifier_name]
        table_ece = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='ece')
        table_ece.name = f'calibration_pps_ECE_{classifier_name}'
        table_ece.reorder_methods(method_order)
        table_ece.format.configuration.show_std=False
        table_ece.format.configuration.side_columns = True
        tables.append(table_ece)

        table_brier = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='brier')
        table_brier.name = f'calibration_pps_brier_{classifier_name}'
        table_brier.reorder_methods(method_order)
        table_brier.format.configuration.show_std = False
        table_brier.format.configuration.side_columns = True
        tables.append(table_brier)

    LatexTable.LatexPDF(f'./tables/calibration_label_shift.pdf', tables)

