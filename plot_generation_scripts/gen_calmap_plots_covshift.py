import os

import matplotlib.pyplot as plt
import numpy as np
from pycalib.visualisations import plot_binary_reliability_diagram_gaps, plot_calibration_map
import pandas as pd
from quapy.data import LabelledCollection
from quapy.method.aggregative import DistributionMatchingY, PCC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from calibration_experiments_covariate_shift import get_calibrated_posteriors
from calibration_experiments_label_shift import calibrate
from commons import iterate_datasets_covariate_shift, total_setups_covariate_shift, yield_random_samples, REPEATS, \
    SAMPLE_SIZE, new_natur_prev_protocol
from gen_tables import replace_method
from model.classifier_accuracy_predictors import DoC
from model.classifier_calibrators import *
import quapy as qp
from pathlib import Path
import itertools
from util import makepath, PrecomputedClassifier
from tqdm import tqdm


def calibrator_iter(setup):
    Pva = setup.valid.posteriors
    yva = setup.valid.labels
    h = PrecomputedClassifier()
    val_idx = h.feed(X=setup.valid.hidden, P=setup.valid.posteriors, L=setup.valid.logits)

    yield 'Bin2-DoC6', CAP2Calibrator(classifier=h, cap_method=DoC(h, protocol=new_natur_prev_protocol(val_idx, yva, [0, 1])), nbins=6, monotonicity=True, smooth=True).fit(val_idx, yva)
    yield 'Bin6-PCC5', Quant2Calibrator(classifier=h, quantifier_cls=PCC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono
    yield 'Platt', PlattScaling().fit(Pva, yva)
    yield 'Bin6-PACC5', Quant2Calibrator(classifier=h, quantifier_cls=PACC, nbins=5, smooth=True, monotonicity=True).fit(val_idx, yva) # smooth and mono

pbar = tqdm(iterate_datasets_covariate_shift(), total=total_setups_covariate_shift)
for setup in pbar:
    description = f'[{setup.model}]::{setup.source}->{setup.target}'

    for cal_name, calibrator in calibrator_iter(setup):
        for idx, (test_sample, shift) in enumerate(yield_random_samples(setup.in_test, setup.out_test, repeats=REPEATS, samplesize=SAMPLE_SIZE)):
            if idx!=99: continue
            calib_posteriors = get_calibrated_posteriors(calibrator, setup.train, setup.valid, test_sample)

            x = test_sample.posteriors[:,1]
            y = calib_posteriors[:,1]
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            plt.plot(x, y, label=replace_method.get(cal_name, cal_name))

            outpath = (f'figures/calibration/covariate_shift/{setup.model}_{setup.source}-{setup.target}_calmap.png')
            makepath(outpath)

    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Uncalibrated decision score')
    plt.ylabel('Calibrated posterior probability')
    plt.legend()
    plt.gca().set_aspect('equal')  # Para que la diagonal quede a 45°
    plt.grid(True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.cla()



