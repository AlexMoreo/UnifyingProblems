import numpy as np
import torch
from quapy.data import LabelledCollection

from lascal import Calibrator
from methods import HeadToTail, Cpcs
from quapy.method.aggregative import DistributionMatchingY

EPSILON = 1e-7


# ----------------------------------------------------------
# Under Label Shift
# ----------------------------------------------------------
class LasCalCalibration:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def predict_proba(self, Str, ytr, Ste):

        calibrator = Calibrator(
            experiment_path=None,
            verbose=self.verbose,
            covariate=False,
        )

        Str = torch.from_numpy(Str)
        Ste = torch.from_numpy(Ste)
        ytr = torch.from_numpy(ytr)
        yte = None

        source_agg = {
            'y_logits': Str,
            'y_true': ytr
        }
        target_agg = {
            'y_logits': Ste,
            'y_true': yte
        }

        calibrated_agg = calibrator.calibrate(
            method_name='lascal',
            source_agg=source_agg,
            target_agg=target_agg,
            train_agg=None,
        )

        y_logits = calibrated_agg['target']['y_logits']
        Pte_calib = y_logits.softmax(-1).numpy()
        return Pte_calib


class HellingerDistanceCalibration:

    def __init__(self, hdy:DistributionMatchingY):
        self.hdy = hdy

    def predict_proba(self, Ste):
        dm = self.hdy
        nbins = dm.nbins
        estim_prev = dm.aggregate(Ste)
        hist_neg, hist_pos = dm.validation_distribution
        # because the histograms were computed wrt the posterior of the first class (the negative one!), we invert the order
        # which is equivalent to computing the histogram wrt the positive class
        hist_neg = hist_neg.flatten()[::-1]
        hist_pos = hist_pos.flatten()[::-1]
        hist_neg = hist_neg * estim_prev[0] + EPSILON
        hist_pos = hist_pos * estim_prev[1] + EPSILON
        corrected_posteriors_bins = hist_pos / (hist_neg + hist_pos)
        corrected_posteriors_bins = np.concatenate(([0.], corrected_posteriors_bins, [1.]))
        x_coords = np.concatenate(
            ([0.], (np.linspace(0., 1., nbins + 1)[:-1] + 0.5 / nbins), [1.]))  # this assumes binning=isometric
        uncalibrated_posteriors_pos = Ste[:,1]
        posteriors = np.interp(uncalibrated_posteriors_pos, x_coords, corrected_posteriors_bins)
        posteriors = np.asarray([1 - posteriors, posteriors]).T
        return posteriors


# ----------------------------------------------------------
# Under Covariate Shift
# ----------------------------------------------------------
# TransCal [Wang et al., 2020]
class CpcsCalibrator:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit(self, Str, ytr):
        cpcs = Cpcs()
        self.optim_temp = cpcs.find_best_T(torch.from_numpy(Str), torch.from_numpy(ytr), weight=1)
        print(self.optim_temp)
        if self.verbose:
            print(f"Temperature found with Cpcs is: {self.optim_temp}")

    def predict_proba(self, Ste):
        y_logits = Ste / self.optim_temp
        Pte_recalib = torch.from_numpy(y_logits).softmax(-1).numpy()
        return Pte_recalib


# HeadToTail [Chen and Su, 2023]
class HeadToTailCalibrator:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def predict_proba(self, Xtr, ytr, Xva, Sva, yva, Ste):
        head_to_tail = HeadToTail(
            num_classes=2,
            features=Xva,
            logits=Sva,
            labels=yva,
            train_features=Xtr,
            train_labels=ytr,
            verbose=self.verbose,
        )
        optim_temp = head_to_tail.find_best_T(Sva, yva)
        if self.verbose:
            print(f"Temperature found with HeadToTail is: {optim_temp[0]}")

        y_logits = Ste / optim_temp
        Pte_recalib = y_logits.softmax(-1).numpy()
        return Pte_recalib
