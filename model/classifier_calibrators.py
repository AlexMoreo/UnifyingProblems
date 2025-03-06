import numpy as np
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
import quapy.functional as F
import util
from calibrator import cal_acc_error, get_weight_feature_space
from lascal import Calibrator
from methods import HeadToTail, Cpcs, TransCal
from quapy.method.aggregative import DistributionMatchingY, EMQ, ACC, PACC
from scipy.special import softmax
from abc import ABC, abstractmethod
from sklearn.calibration import _fit_calibrator

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
    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        ...

def np2tensor(scores, probability_to_logit=False):
    scores = torch.from_numpy(scores)
    if probability_to_logit:
        scores = torch.log(scores)
    return scores

def np_prob2logit(prob):
    return np2tensor(prob, probability_to_logit=True).numpy()

# ----------------------------------------------------------
# Baseline methods
# ----------------------------------------------------------
# Basic wrap that simply returns the posterior probabilities
class UncalibratedWrap(CalibratorSimple):
    def __init__(self):
        pass

    def calibrate(self, Z):
        if not all(np.isclose(Z.sum(axis=1), 1, atol=1e-5)):
            return softmax(Z, axis=-1)
        else:
            return Z


class NaiveUncertain(CalibratorSimple):
    def __init__(self, train_prev=None):
        self.train_prev = train_prev

    def calibrate(self, Z):
        if self.train_prev is None:
            uncertain = np.full_like(Z, fill_value=0.5)
        else:
            uncertain = np.tile(self.train_prev, (Z.shape[0],1))
        return uncertain


class PlattScaling(CalibratorSimple):

    def __init__(self):
        pass

    def fit(self, Zva, yva):
        self.calibrator = _SigmoidCalibration()
        self.calibrator.fit(Zva[:,1], yva)
        return self

    def calibrate(self, Z):
        if not hasattr(self, 'calibrator'):
            raise RuntimeError('calibrate called before fit')
        calibrated_pos = self.calibrator.predict(Z[:,1])
        calibrated = np.asarray([1-calibrated_pos, calibrated_pos]).T
        return calibrated


class IsotonicCalibration(CalibratorSimple):

    def __init__(self):
        pass

    def fit(self, Zva, yva):
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(Zva[:, 1], yva)
        return self

    def calibrate(self, Z):
        if not hasattr(self, 'calibrator'):
            raise RuntimeError('calibrate called before fit')
        calibrated_pos = self.calibrator.predict(Z[:, 1])
        calibrated = np.asarray([1 - calibrated_pos, calibrated_pos]).T
        return calibrated


# ----------------------------------------------------------
# Under Label Shift
# ----------------------------------------------------------

class LasCalCalibration(CalibratorSourceTarget):

    def __init__(self, verbose=False, prob2logits=True):
        self.verbose = verbose
        self.prob2logits = prob2logits

    def calibrate(self, Zsrc, ysrc, Ztgt):

        calibrator = Calibrator(
            experiment_path=None,
            verbose=self.verbose,
            covariate=False,
        )

        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)
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


class EMBCTSCalibration(CalibratorSourceTarget):

    def __init__(self, verbose=False, prob2logits=True):
        self.verbose = verbose
        self.prob2logits = prob2logits

    def calibrate(self, Zsrc, ysrc, Ztgt):

        calibrator = Calibrator(
            experiment_path=None,
            verbose=self.verbose,
            covariate=False,
        )

        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)
        yte = None

        source_agg = {
            'y_logits': Zsrc,
            'y_true': ysrc
        }
        target_agg = {
            'y_logits': Ztgt,
            'y_true': yte
        }

        try:
            calibrated_agg = calibrator.calibrate(
                method_name='em_alexandari',
                source_agg=source_agg,
                target_agg=target_agg,
                train_agg=None,
            )

            y_logits = calibrated_agg['target']['y_logits']
            Pte_calib = y_logits.softmax(-1).numpy()
            # print('good')
            return Pte_calib
        except AssertionError:
            # print('assertion')
            Pte = UncalibratedWrap().calibrate(Zsrc.numpy())
            # print(Zsrc.shape)
            # print(Pte.shape)
            return Pte


# -------------------------------------------------------------
# Based on Quantification
# -------------------------------------------------------------
class HellingerDistanceCalibration(CalibratorSimple):

    def __init__(self, hdy:DistributionMatchingY, smooth=False, monotonicity=False, postsmooth=False):
        self.hdy = hdy
        self.smooth = smooth
        self.postsmooth=postsmooth
        self.monotonicity = monotonicity

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
        calibration_map = hist_pos / (hist_neg + hist_pos)
        calibration_map = np.concatenate(([0.], calibration_map, [1.]))

        if self.monotonicity:
            for i in range(1,nbins-1):
                calibration_map[i] = max(calibration_map[i], calibration_map[i-1])
        if self.smooth:
            calibration_map[1:-1]=np.mean(np.vstack([calibration_map[:-2], calibration_map[1:-1], calibration_map[2:]]), axis=0)

        x_coords = np.concatenate(
            ([0.], (np.linspace(0., 1., nbins + 1)[:-1] + 0.5 / nbins), [1.]))  # this assumes binning=isometric
        uncalibrated_posteriors_pos = Z[:, 1]
        posteriors = np.interp(uncalibrated_posteriors_pos, x_coords, calibration_map)
        if self.postsmooth:
            posteriors[1:-1]=np.mean(np.vstack([posteriors[:-2], posteriors[1:-1], posteriors[2:]]), axis=0)
        posteriors = np.asarray([1 - posteriors, posteriors]).T
        return posteriors


class EM(CalibratorSimple):
    def __init__(self, train_prevalence):
        self.emq = EMQ()
        self.train_prevalence = train_prevalence

    def calibrate(self, P):
        priors, posteriors = EMQ.EM(tr_prev=self.train_prevalence, posterior_probabilities=P)
        return posteriors


class PACCcal(CalibratorSimple):
    POST_PROCESSING = ['clip', 'softmax', 'logistic', 'isotonic']
    def __init__(self, P, y, post_proc='clip'):
        assert post_proc in PACCcal.POST_PROCESSING, 'unknown post_proc method'
        PteCond = PACC.getPteCondEstim(classes=[0,1], y=y, y_=P)
        self.tpr = PteCond[1,1]
        self.fpr = PteCond[1,0]
        self.post_proc = post_proc
        if post_proc=='logistic':
            cal = self._transform(P)
            self.regressor = _SigmoidCalibration()
            self.regressor.fit(cal, y)
        elif post_proc == 'isotonic':
            cal = self._transform(P)
            self.regressor = IsotonicRegression(out_of_bounds="clip")
            self.regressor.fit(cal, y)

    def _transform(self, P):
        tpr, fpr = self.tpr, self.fpr
        denom = tpr - fpr
        if denom > 0:
            Ppos = P[:, 1]
            calib = (Ppos - fpr) / (tpr - fpr)
            return calib
        else:
            return P[:,1]

    def calibrate(self, P):
        Ppos = self._transform(P)
        if self.post_proc == 'clip':
            calib = qp.functional.as_binary_prevalence(Ppos, clip_if_necessary=True)
        elif (Ppos>1).any() or (Ppos<0).any():
            if self.post_proc == 'softmax':
                calib = softmax(np.asarray([1 - Ppos, Ppos]).T, axis=1)
            else:
                if (Ppos>1).any() or (Ppos<0).any():
                    Ppos = self.regressor.predict(Ppos)
                calib = np.asarray([1-Ppos,Ppos]).T
        return calib


class QuantifyBinsCalibrator_depr(CalibratorSimple): # does not work at all
    def __init__(self, classifier, quantifier_cls, nbins=8):
        self.classifier = classifier
        self.quantifier_cls = quantifier_cls
        self.bins=np.linspace(0, 1, nbins+1)
        self.bins[-1]+=1e-5

    def fit(self, P, y):
        posteriors = P[:,1]
        self.quantifiers=[]
        for bin_low, bin_high in zip(self.bins[:-1],self.bins[1:]):
            sel = np.logical_and(posteriors>=bin_low, posteriors<bin_high)
            if sum(sel)>0 and sum(y[sel]==1)>4 and sum(y[sel]==0)>4:
                lc = LabelledCollection(P[sel], y[sel], classes=[0,1])
                q_bin = self.quantifier_cls(clone(self.classifier)).fit(lc)
                self.quantifiers.append(q_bin)
            else:
                self.quantifiers.append(None)
        return self

    def calibrate(self, P):
        posteriors = P[:,1]
        calibrated = np.zeros_like(posteriors, dtype=float)
        for bin_low, bin_high, quant in zip(self.bins[:-1],self.bins[1:], self.quantifiers):
            sel = np.logical_and(posteriors>=bin_low, posteriors<bin_high)
            binsize = sum(sel)
            if quant is not None and binsize>0:
                prev = quant.quantify(P[sel])[1]
            else:
                prev = 0.5
            calibrated[sel] = prev

        calibrated = np.asarray([1-calibrated, calibrated]).T
        return calibrated


class QuantifyCalibrator(CalibratorCompound):
    def __init__(self, classifier, quantifier_cls, nbins=5):
        self.classifier = classifier
        self.quantifier = quantifier_cls()
        self.bins = np.linspace(0, 1, nbins + 1)
        self.bins[-1] += 1e-5

    def fit(self, X, y):
        self.quantifier.fit(LabelledCollection(X,y,classes=[0,1]))
        return self

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        posteriors = Ztgt[:,1]
        calibrated = np.zeros_like(posteriors, dtype=float)
        for bin_low, bin_high in zip(self.bins[:-1],self.bins[1:]):
            sel = np.logical_and(posteriors>=bin_low, posteriors<bin_high)
            binsize = sum(sel)
            if binsize>0:
                prev = self.quantifier.quantify(Ftgt[sel])[1]
            else:
                prev = 0.5
            calibrated[sel] = prev

        calibrated = np.asarray([1-calibrated, calibrated]).T
        return calibrated

# ----------------------------------------------------------
# Under Covariate Shift
# ----------------------------------------------------------
# TransCal [Wang et al., 2020]
class TransCalCalibrator(CalibratorCompound):

    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        Zsrc = np2tensor(Zsrc, self.prob2logits)
        Ztgt = np2tensor(Ztgt, self.prob2logits)
        ysrc = np2tensor(ysrc)

        optim_temp_source = Calibrator()._temperature_scale(
            logits=Zsrc,
            labels=ysrc
        )
        _, source_confidence, error_source_val = cal_acc_error(
            Zsrc / optim_temp_source, ysrc
        )
        weight = get_weight_feature_space(Ftr, Ftgt, Fsrc)
        # Find optimal temp with TransCal
        optim_temp = TransCal().find_best_T(
            Ztgt.numpy(),
            weight,
            error_source_val,
            source_confidence.item(),
        )
        y_logits = Ztgt / optim_temp
        Pte_recalib = y_logits.softmax(-1).numpy()
        return Pte_recalib


# CPCS [Park et al., 2020]
class CpcsCalibrator(CalibratorCompound):

    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):

        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)

        weight = get_weight_feature_space(Ftr, Ftgt, Fsrc)
        # Find optimal temp with TransCal

        optim_temp = Cpcs().find_best_T(Zsrc, ysrc, weight)
        y_logits = Ztgt / optim_temp
        Pte_recalib = y_logits.softmax(-1).numpy()
        return Pte_recalib


# HeadToTail [Chen and Su, 2023]
class HeadToTailCalibrator(CalibratorCompound):

    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        if self.prob2logits:
            Zsrc = np_prob2logit(Zsrc)
            Ztgt = np_prob2logit(Ztgt)

        head_to_tail = HeadToTail(
            num_classes=2,
            features=Fsrc,
            logits=Zsrc,
            labels=ysrc,
            train_features=Ftr,
            train_labels=ytr,
        )
        optim_temp = head_to_tail.find_best_T(Zsrc, ysrc)

        y_logits = Ztgt / optim_temp
        Pte_recalib = softmax(y_logits, axis=-1)
        return Pte_recalib


