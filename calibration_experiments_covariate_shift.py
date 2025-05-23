import os
from dataclasses import dataclass, asdict
from sklearn.metrics import brier_score_loss
import pandas as pd
from tqdm import tqdm
from quapy.method.aggregative import KDEyML
from quapy.method.aggregative import PCC
from model.classifier_calibrators import *
from model.classifier_accuracy_predictors import ATC, DoC, LEAP
from util import cal_error, PrecomputedClassifier
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

    yield 'TransCal', TransCalCalibrator(prob2logits=False)
    # yield 'Head2Tail', HeadToTailCalibrator(prob2logits=True, n_components=50).fit(
    #     Ftr=setup.train.hidden, ytr=setup.train.labels,
    #     Fsrc=setup.valid.hidden, Zsrc=setup.valid.posteriors, ysrc=setup.valid.labels
    # )
    yield 'CPCS', CpcsCalibrator(prob2logits=False)
    yield 'LasCal', LasCalCalibration(prob2logits=False)

    yield 'EM', EMQ_Calibrator(train_prevalence=setup.train.prevalence)
    yield 'EM-BCTS', EMQ_BCTS_Calibrator().fit(Pva, yva)
    yield 'EM-TransCal', EMQ_TransCal_Calibrator(train_prevalence=setup.train.prevalence, prob2logits=False)

    h = PrecomputedClassifier()
    val_idx = h.feed(X=setup.valid.hidden, P=setup.valid.posteriors, L=setup.valid.logits)

    yield f'HDcal8-sm-mono', DistributionMatchingCalibration(h, nbins=8).fit(Pva, yva)

    yield 'PACC-cal(soft)', PacCcal(Pva, yva, post_proc='softmax')

    yield 'Bin6-PCC5', Quant2Calibrator(classifier=h, quantifier_cls=PCC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono
    yield 'Bin6-PACC5', Quant2Calibrator(classifier=h, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono
    yield 'Bin6-EM5', Quant2Calibrator(classifier=h, quantifier_cls=EMQ, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva)
    yield 'Bin6-KDEy5', Quant2Calibrator(classifier=h, quantifier_cls=KDEyML, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva)

    yield 'Bin2-ATC6', CAP2Calibrator(classifier=h, cap_method=ATC(h), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin2-DoC6', CAP2Calibrator(classifier=h, cap_method=DoC(h, protocol=new_natur_prev_protocol(val_idx, yva, [0, 1])), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin2-LEAP6', CAP2Calibrator(classifier=h, cap_method=LEAP(h, KDEyML(classifier=h)), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin2-LEAP-PCC-6', CAP2Calibrator(classifier=h, cap_method=LEAP(h, PCC(classifier=h)), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)



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


if __name__ == '__main__':
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

