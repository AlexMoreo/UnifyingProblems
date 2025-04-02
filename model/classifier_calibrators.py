import numpy as np
import quapy as qp
import torch
from abstention.calibration import TempScaling
from numpy.ma.core import shape
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
from sklearn.decomposition import PCA



EPSILON = 1e-7

class CalibratorSimple(ABC):
    @abstractmethod
    def calibrate(self, Z):
        ...

class CalibratorSourceTarget(ABC):
    @abstractmethod
    def calibrate(self, Zsrc, ysrc, Ztgt):
        ...

class CalibratorTarget(ABC):
    @abstractmethod
    def calibrate(self, Ftgt, Ztgt):
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

    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Zsrc, ysrc, Ztgt):

        calibrator = Calibrator(
            experiment_path=None,
            verbose=False,
            covariate=False,
        )

        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)
        yte = None

        try:
            calibrated_agg = calibrator.calibrate(
                method_name='lascal',
                source_agg={
                    'y_logits': Zsrc,
                    'y_true': ysrc
                },
                target_agg={
                    'y_logits': Ztgt,
                    'y_true': yte
                },
                train_agg=None,
            )
            y_logits = calibrated_agg['target']['y_logits']
            Pte_calib = y_logits.softmax(-1).numpy()
        except Exception:
            Ztgt = Ztgt.numpy()
            if np.isclose(Ztgt.sum(axis=1), 1).all():
                Pte_calib = Ztgt
            else:
                Pte_calib = softmax(Ztgt, axis=1)

        return Pte_calib


class EMQ_LasCal_Calibrator(LasCalCalibration):

    def __init__(self, train_prevalence, prob2logits=True):
        self.emq = EMQ()
        self.train_prevalence = train_prevalence
        self.prob2logits = prob2logits

    def calibrate(self, Zsrc, ysrc, Ztgt):
        P_lascal = super().calibrate(Zsrc, ysrc, Ztgt)
        priors, posteriors = EMQ.EM(tr_prev=self.train_prevalence, posterior_probabilities=P_lascal)
        return posteriors


class EMQ_BCTS_Calibrator(CalibratorSimple):

    def __init__(self):
        pass

    def fit(self, Pva, yva):
        try:
            self.calibration_function = TempScaling(bias_positions='all')(
                Pva, np.eye(2)[yva], posterior_supplied=True
            )
            self.train_prevalence = F.prevalence_from_probabilities(
                self.calibration_function(Pva)
            )
        except Exception:
            print('abstension raised an error; backing up to EMQ')
            self.calibration_function = lambda P:P
            self.train_prevalence = F.prevalence_from_labels(yva, classes=[0,1])
        return self

    def calibrate(self, P):
        if not np.isclose(P.sum(axis=1), 1).all():
            raise ValueError('P does not seem to be an array of posterior probabilities')
        P = self.calibration_function(P)
        priors, calib_posteriors = EMQ.EM(self.train_prevalence, P)
        return calib_posteriors


# ----------------------------------------------------------
# Under Covariate Shift
# ----------------------------------------------------------
# TransCal [Wang et al., 2020]
class TransCalCalibrator(CalibratorCompound):

    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)

        optim_temp_source = Calibrator()._temperature_scale(
            logits=Zsrc,
            labels=ysrc
        )
        _, source_confidence, error_source_val = cal_acc_error(
            Zsrc / optim_temp_source, ysrc
        )
        try:
            weight = get_weight_feature_space(Ftr, Ftgt, Fsrc)
        except ValueError:
            weight = np.ones(shape=(Fsrc.shape[0],1), dtype=float)
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

        try:
            weight = get_weight_feature_space(Ftr, Ftgt, Fsrc)
        except ValueError:
            weight = np.ones(shape=(Fsrc.shape[0],1), dtype=float)
        # Find optimal temp with TransCal

        optim_temp = Cpcs().find_best_T(Zsrc, ysrc, weight)
        y_logits = Ztgt / optim_temp
        Pte_recalib = y_logits.softmax(-1).numpy()
        return Pte_recalib


# HeadToTail [Chen and Su, 2023]
class HeadToTailCalibrator(CalibratorSimple):

    def __init__(self, prob2logits=True, n_components=None):
        self.prob2logits = prob2logits
        self.n_components = n_components

    def fit(self, Ftr, ytr, Fsrc, Zsrc, ysrc):
        if self.prob2logits:
            Zsrc = np_prob2logit(Zsrc)

        if self.n_components is not None and Ftr.shape[1]>self.n_components:
            print(f'reducing to {self.n_components} principal components')
            pca = PCA(n_components=self.n_components)
            Ftr = pca.fit_transform(Ftr)
            Fsrc = pca.transform(Fsrc)

        head_to_tail = HeadToTail(
            num_classes=2,
            features=Fsrc,
            logits=Zsrc,
            labels=ysrc,
            train_features=Ftr,
            train_labels=ytr,
        )
        self.optim_temp = head_to_tail.find_best_T(Zsrc, ysrc)
        return self

    def calibrate(self, Ztgt):
        if self.prob2logits:
            Ztgt = np_prob2logit(Ztgt)

        y_logits = Ztgt / self.optim_temp
        Pte_recalib = softmax(y_logits, axis=-1)
        return Pte_recalib
    

class EMQ_TransCal_Calibrator(TransCalCalibrator):

    def __init__(self, train_prevalence, prob2logits=True):
        self.emq = EMQ()
        self.train_prevalence = train_prevalence
        self.prob2logits = prob2logits

    def calibrate(self, Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt):
        P_calib = super().calibrate(Ftr, ytr, Fsrc, Zsrc, ysrc, Ftgt, Ztgt)
        priors, posteriors = EMQ.EM(tr_prev=self.train_prevalence, posterior_probabilities=P_calib)
        return posteriors


# -------------------------------------------------------------
# Based on Quantification
# -------------------------------------------------------------
class DistributionMatchingCalibration(CalibratorSimple):

    def __init__(self, classifier, smooth=True, monotonicity=True, nbins=8):
        self.h = classifier
        self.nbins = nbins
        self.smooth = smooth
        self.monotonicity = monotonicity

    def fit(self, P, y):
        self.dm = DistributionMatchingY(classifier=self.h, nbins=self.nbins)
        labelled_posteriors = LabelledCollection(P, y)
        self.dm.aggregation_fit(classif_predictions=labelled_posteriors, data=None)
        return self

    def calibrate(self, Z):
        dm = self.dm
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
            calibration_map = util.impose_monotonicity(calibration_map)
        if self.smooth:
            calibration_map = util.smooth(calibration_map)

        x_coords = np.concatenate(
            ([0.], (np.linspace(0., 1., nbins + 1)[:-1] + 0.5 / nbins), [1.]))  # this assumes binning=isometric
        uncalibrated_posteriors_pos = Z[:, 1]
        posteriors = np.interp(uncalibrated_posteriors_pos, x_coords, calibration_map)
        posteriors = np.asarray([1 - posteriors, posteriors]).T
        return posteriors


class EMQ_Calibrator(CalibratorSimple):
    def __init__(self, train_prevalence):
        self.emq = EMQ()
        self.train_prevalence = train_prevalence

    def calibrate(self, P):
        priors, posteriors = EMQ.EM(tr_prev=self.train_prevalence, posterior_probabilities=P)
        return posteriors


class PacCcal(CalibratorSimple):
    
    POST_PROCESSING = ['clip', 'softmax']

    def __init__(self, P, y, post_proc='clip'):
        assert post_proc in PacCcal.POST_PROCESSING, \
            f'unknown post_proc method; valid ones are {PacCcal.POST_PROCESSING}'
        PteCond = PACC.getPteCondEstim(classes=[0,1], y=y, y_=P)
        self.tpr = PteCond[1,1]
        self.fpr = PteCond[1,0]
        self.post_proc = post_proc

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
            # softmax, only if out of bounds
            calib = softmax(np.asarray([1 - Ppos, Ppos]).T, axis=1)
        return calib


class Quant2Calibrator(CalibratorTarget):
    def __init__(self, classifier, quantifier_cls, nbins=5, dedicated=False, monotonicity=False, smooth=False, isometric=True):
        self.classifier = classifier
        self.quantifier_cls = quantifier_cls
        self.nbins=nbins
        self.dedicated = dedicated
        self.monotonicity = monotonicity
        self.smooth = smooth
        self.isometric = isometric

    def fit(self, X, y):
        posteriors = self.classifier.predict_proba(X)[:,1]
        
        if self.isometric:
            self.bins = util.isometric_binning(self.nbins)
        else:
            self.bins = util.isodense_binning(self.nbins, posteriors)

        self.quantifiers_bin = []
        if self.dedicated:
            # learns one dedicated adjustment for each bin
            for bin_low, bin_high in zip(self.bins[:-1],self.bins[1:]):
                sel = np.logical_and(posteriors>=bin_low, posteriors<bin_high)
                #print(f'#sel {sum(sel)} in which #positives {sum(y[sel]==1)} and #negative {sum(y[sel]==0)}')
                if sum(sel)>0:
                    if sum(y[sel]==1)>4 and sum(y[sel]==0)>4:
                        quantifier_bin = self.quantifier_cls(self.classifier)
                        quantifier_bin.fit(LabelledCollection(X[sel], y[sel], classes=[0,1]), fit_classifier=False)
                        self.quantifiers_bin.append(quantifier_bin)
                    else:
                        self.quantifiers_bin.append(np.mean(y[sel]))
                else:
                    self.quantifiers_bin.append(None)
        else:
            quantifier = self.quantifier_cls(self.classifier)
            quantifier.fit(LabelledCollection(X,y,classes=[0,1]), fit_classifier=False)
            # the same quantifier for all bins
            for b in range(len(self.bins)-1):
                self.quantifiers_bin.append(quantifier)
        return self

    def calibrate(self, Ftgt, Ztgt):
        posteriors = Ztgt[:,1]
        calibration_coord = []
        calibrated_values = []
        for bin_low, bin_high, quantifier in zip(self.bins[:-1],self.bins[1:], self.quantifiers_bin):
            sel = np.logical_and(posteriors>=bin_low, posteriors<bin_high)
            binsize = sum(sel)
            if binsize>0:
                bin_inner = np.mean(posteriors[sel])
            else:
                bin_inner = (bin_low + bin_high) / 2

            prev = np.nan 
            if binsize>0:
                if isinstance(quantifier, float):
                    prev = quantifier
                elif quantifier is not None:
                    prev = quantifier.quantify(Ftgt[sel])[1]
                
            calibration_coord.append(bin_inner)
            calibrated_values.append(prev)

        calibration_coord = np.asarray([0.] + calibration_coord + [1.])
        calibrated_values = np.asarray([0.] + calibrated_values + [1.])

        calibrated_values = util.impute_nanvalues_via_interpolation(calibration_coord, calibrated_values)

        if self.monotonicity:
            calibrated_values = util.impose_monotonicity(calibrated_values)
        if self.smooth:
            calibrated_values = util.smooth(calibrated_values)

        calibrated_posteriors = np.interp(posteriors, xp=calibration_coord, fp=calibrated_values)

        calibrated = np.asarray([1 - calibrated_posteriors, calibrated_posteriors]).T
        return calibrated


# ---------------------------------------------------------------
# Based on CAP
# ---------------------------------------------------------------
class CAP2Calibrator(CalibratorTarget):

    def __init__(self, classifier, cap_method, nbins=6, monotonicity=True, smooth=True):
        assert nbins%2==0, f'unexpected number of bins {nbins}; use an odd number of bins'
        self.classifier = classifier
        self.cap = cap_method
        self.bins = np.linspace(0, 1, nbins + 1)
        self.bins[-1] += 1e-5
        self.monotonicity = monotonicity
        self.smooth = smooth

    def fit(self, X, y):
        self.cap.fit(X, y)
        return self

    def calibrate(self, Ftgt, Ztgt):
        posteriors = Ztgt[:, 1]
        calibration_coord = []
        calibrated_values = []
        for bin_low, bin_high in zip(self.bins[:-1], self.bins[1:]):
            bin_center = (bin_low+bin_high)/2
            sel = np.logical_and(posteriors >= bin_low, posteriors < bin_high)
            binsize = sum(sel)
            if binsize > 0:
                estim_acc = self.cap.predict(Ftgt[sel])
                estim_positives = estim_acc if bin_center>0.5 else (1.-estim_acc)
            else:
                estim_positives = np.nan
            calibration_coord.append(bin_center)
            calibrated_values.append(estim_positives)

        calibration_coord = np.asarray([0.] + calibration_coord + [1.])
        calibrated_values = np.asarray([0.] + calibrated_values + [1.])

        calibrated_values = util.impute_nanvalues_via_interpolation(calibration_coord, calibrated_values)

        if self.monotonicity:
            calibrated_values = util.impose_monotonicity(calibrated_values)
        if self.smooth:
            calibrated_values = util.smooth(calibrated_values)

        calibrated_posteriors = np.interp(posteriors, xp=calibration_coord, fp=calibrated_values)

        calibrated = np.asarray([1 - calibrated_posteriors, calibrated_posteriors]).T
        return calibrated


