import numpy as np
import torch
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator

from calibrator import cal_acc_error, get_weight_feature_space
from lascal import Calibrator
from methods import HeadToTail, Cpcs, TransCal
from quapy.method.aggregative import DistributionMatchingY
from scipy.special import softmax
from abc import ABC, abstractmethod

EPSILON = 1e-7

class CalibratorSimple(ABC):
    @abstractmethod
    def calibrate(self, Z):
        ...

class CalibratorSourceTarget(ABC):
    @abstractmethod
    def calibrate(self, Zsrc, ysrc, Ztgt):
        ...

class CalibratorCompound(ABC):
    @abstractmethod
    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Fte, Zte):
        ...


# ----------------------------------------------------------
# Basic wrap that simply returns the posterior probabilities
# ----------------------------------------------------------
class UncalibratedWrap(CalibratorSimple):
    def __init__(self):
        pass

    def calibrate(self, Z):
        if not all(np.isclose(Z.sum(axis=1), 1, atol=1e-5)):
            return softmax(Z, axis=-1)
        else:
            return Z

# ----------------------------------------------------------
# Under Label Shift
# ----------------------------------------------------------

class LasCalCalibration(CalibratorSourceTarget):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def calibrate(self, Zsrc, ysrc, Ztgt):

        calibrator = Calibrator(
            experiment_path=None,
            verbose=self.verbose,
            covariate=False,
        )

        Zsrc = torch.from_numpy(Zsrc)
        Ztgt = torch.from_numpy(Ztgt)
        ysrc = torch.from_numpy(ysrc)
        yte = None

        source_agg = {
            'y_logits': Zsrc,
            'y_true': ysrc
        }
        target_agg = {
            'y_logits': Ztgt,
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


class HellingerDistanceCalibration(CalibratorSimple):

    def __init__(self, hdy:DistributionMatchingY):
        self.hdy = hdy

    def calibrate(self, Z):
        dm = self.hdy
        nbins = dm.nbins
        estim_prev = dm.aggregate(Z)
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
        uncalibrated_posteriors_pos = Z[:, 1]
        posteriors = np.interp(uncalibrated_posteriors_pos, x_coords, corrected_posteriors_bins)
        posteriors = np.asarray([1 - posteriors, posteriors]).T
        return posteriors


# ----------------------------------------------------------
# Under Covariate Shift
# ----------------------------------------------------------
# TransCal [Wang et al., 2020]
class TransCalCalibrator(CalibratorCompound):

    def __init__(self):
        pass

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Fte, Zte):
        Zsrc = torch.from_numpy(Zsrc)
        ysrc = torch.from_numpy(ysrc)
        optim_temp_source = Calibrator()._temperature_scale(
            logits=Zsrc,
            labels=ysrc
        )
        _, source_confidence, error_source_val = cal_acc_error(
            Zsrc / optim_temp_source, ysrc
        )
        weight = get_weight_feature_space(Ftr, Fte, Fsrc)
        # Find optimal temp with TransCal
        optim_temp = TransCal().find_best_T(
            Zte,
            weight,
            error_source_val,
            source_confidence.item(),
        )
        y_logits = Zte / optim_temp
        Pte_recalib = torch.from_numpy(y_logits).softmax(-1).numpy()
        return Pte_recalib


class CpcsCalibrator(CalibratorCompound):

    def __init__(self):
        pass

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Fte, Zte):
        weight = get_weight_feature_space(Ftr, Fte, Fsrc)
        # Find optimal temp with TransCal
        optim_temp = Cpcs().find_best_T(
            torch.from_numpy(Zsrc),
            torch.from_numpy(ysrc),
            weight
        )
        y_logits = Zte / optim_temp
        Pte_recalib = torch.from_numpy(y_logits).softmax(-1).numpy()
        return Pte_recalib


# HeadToTail [Chen and Su, 2023]
class HeadToTailCalibrator(CalibratorCompound):

    def __init__(self, verbose):
        self.verbose = verbose

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Fte, Zte):
        head_to_tail = HeadToTail(
            num_classes=2,
            features=Fsrc,
            logits=Zsrc,
            labels=ysrc,
            train_features=Ftr,
            train_labels=ytr,
            verbose=self.verbose,
        )
        optim_temp = head_to_tail.find_best_T(Zsrc, ysrc)
        if self.verbose:
            print(f"Temperature found with HeadToTail is: {optim_temp[0]}")

        y_logits = Zte / optim_temp
        # Pte_recalib = y_logits.softmax(-1).numpy()
        Pte_recalib = softmax(y_logits, axis=-1)
        return Pte_recalib
