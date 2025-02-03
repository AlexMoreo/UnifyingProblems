import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeSoftQuantifier
import quapy.functional as F
from sklearn.base import BaseEstimator

from model.classifier_calibrators import LasCalCalibration


# -----------------------------------------------------
# Quantifiers
# -----------------------------------------------------

class OracleQuantifier:

    def __init__(self):
        pass

    def quantify(self, X, y):
        prevalence = F.prevalence_from_labels(y, classes=[0,1])
        return prevalence


class OracleQuantifierFromCalibrator:

    def __init__(self, oracle_cal: 'OracleCalibrator'):
        self.oracle_cal = oracle_cal

    def quantify(self, X, y):
        hcal = self.oracle_cal.calibrate(X, y)
        posteriors = hcal.predict_proba(X)
        return posteriors.mean(axis=0)


class OracleQuantifierFromCAP:

    def __init__(self, oracle_cap: 'OracleCAP'):
        self.oracle_cap = oracle_cap

    def quantify(self, X, y):
        y_hat = self.oracle_cap.h.predict(X)

        n = len(y)

        X_pos = X[y_hat==1]
        y_pos = y[y_hat==1]
        n_pos = (y_hat==1).sum()

        X_neg = X[y_hat==0]
        y_neg = y[y_hat==0]
        n_neg = n-n_pos

        alpha = self.oracle_cap
        acc_plus = acc_minus = 0
        if n_pos>0:
            acc_plus = alpha.predict_accuracy(X_pos, y_pos)
        if n_neg>0:
            acc_minus = alpha.predict_accuracy(X_neg, y_neg)

        pos_prev = acc_plus * (n_pos / n) + (1-acc_minus)*(n_neg/n)
        prevalence = F.as_binary_prevalence(positive_prevalence=pos_prev)
        return prevalence


class LasCalQuantifier(AggregativeSoftQuantifier):

    def __init__(self, classifier, val_split=5, verbose=False):
        self.classifier = classifier
        self.lascal = LasCalCalibration(verbose=verbose)
        self.val_split = val_split

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.Ptr = classif_predictions.X
        self.ytr = classif_predictions.y
        return self

    def aggregate(self, Pte):
        Pte_recalib = self.lascal.predict_proba(self.Ptr, self.ytr, Pte)
        prevalence = Pte_recalib.mean(axis=0)
        return prevalence


# -----------------------------------------------------
# Calibrators
# -----------------------------------------------------

class Binning:

    def __init__(self, nbins, mode='isometric'):
        if mode not in ['isometric']:
            raise NotImplementedError(f'{mode=} not implemented!')
        self.nbins = nbins

    def fit(self, scores):
        self.bins = np.linspace(scores.min(), scores.max(), self.nbins + 1)
        self.bins[0] = -np.inf
        self.bins[-1] = np.inf
        return self

    def transform(self, scores):
        return np.digitize(scores, self.bins) - 1

    def fit_transform(self, scores):
        return self.fit(scores).transform(scores)


class EmpiricalCalibrationMap:

    def __init__(self, nbins=10):
        self.nbins = nbins
        self.binning = Binning(nbins, mode='isometric')

    def fit(self, uncal_scores, y):
        if uncal_scores.ndim==2:
            uncal_scores = uncal_scores[:, 1]
        bins = self.binning.fit_transform(uncal_scores)
        self.cal_map = np.asarray([np.mean(y[bins==i]) for i in range(self.nbins)])
        return self

    def calibrate(self, uncal_scores):
        if uncal_scores.ndim == 2:
            uncal_scores = uncal_scores[:, 1]
        bins = self.binning.transform(uncal_scores)
        cal_scores = self.cal_map[bins]
        return np.vstack((1-cal_scores, cal_scores)).T


class QuantificationCalibrationMap:

    def __init__(self, quantifier):
        self.quantifier = quantifier

    def fit(self, X, uncal_scores, y):
        if uncal_scores.ndim==2:
            uncal_scores = uncal_scores[:, 1]
        unique = sorted(np.unique(uncal_scores))
        self.cal_map = {}
        for u in unique:
            sel = uncal_scores==u
            Xsel = X[sel]
            ysel = y[sel]
            prev = self.quantifier.quantify(Xsel, ysel)
            self.cal_map[u] = prev
        return self

    def calibrate(self, uncal_scores):
        if uncal_scores.ndim == 2:
            uncal_scores = uncal_scores[:, 1]
        cal_scores = np.vstack([self.cal_map[score] for score in uncal_scores])
        return cal_scores


class CalibratedClassifierFromMap:

    def __init__(self, h: BaseEstimator, calibration_map: EmpiricalCalibrationMap):
        self.h = h
        self.calibration_map = calibration_map

    def predict_proba(self, X):
        uncal_scores = self.h.predict_proba(X)
        return self.calibration_map.calibrate(uncal_scores)


class OracleCalibrator:

    def __init__(self, h, nbins=10):
        self.nbins = nbins
        self.h = h

    def fit(self, X, y):
        return self

    def calibrate(self, X, y):
        uncal_post = self.h.predict_proba(X)
        calibration_map = EmpiricalCalibrationMap(nbins=self.nbins)
        calibration_map.fit(uncal_post, y)
        return CalibratedClassifierFromMap(self.h, calibration_map)


class OracleCalibratorFromQuantification:

    def __init__(self, h: BaseEstimator, oracle_quantifier: OracleQuantifier):
        self.h = h
        self.oracle_quantifier = oracle_quantifier

    def fit(self, X, y):
        return self

    def calibrate(self, X, y):
        uncal_post = self.h.predict_proba(X)
        calibration_map = QuantificationCalibrationMap(self.oracle_quantifier)
        calibration_map.fit(X, uncal_post, y)
        return CalibratedClassifierFromMap(self.h, calibration_map)


class OracleCalibratorFromCAP:

    def __init__(self, h: BaseEstimator, oracle_cap: 'OracleCAP'):
        self.h = h
        self.oracle_cap = oracle_cap

    def fit(self, X, y):
        return self

    def calibrate(self, X, y):
        oracle_quantifier = OracleQuantifierFromCAP(self.oracle_cap)
        oracle_calibrator = OracleCalibratorFromQuantification(self.h, oracle_quantifier)
        return oracle_calibrator.calibrate(X, y)


# -----------------------------------------------------
# Classifier Accuracy Predictors
# -----------------------------------------------------

class OracleCAP:

    def __init__(self, h):
        self.h = h

    def predict_accuracy(self, X, y):
        y_hat = self.h.predict(X)
        return (y_hat == y).mean()


class OracleCAPfromCalibration:

    def __init__(self, h, oracle_calibrator: OracleCalibrator):
        self.h = h
        self.oracle_calibrator = oracle_calibrator

    def predict_accuracy(self, X, y):
        y_hat = self.h.predict(X)

        n = len(y)

        X_pos = X[y_hat == 1]
        y_pos = y[y_hat == 1]

        X_neg = X[y_hat == 0]
        y_neg = y[y_hat == 0]

        h_pos = self.oracle_calibrator.calibrate(X_pos, y_pos)
        h_neg = self.oracle_calibrator.calibrate(X_neg, y_neg)

        cal_pos = h_pos.predict_proba(X_pos)[:, 1]
        cal_neg = h_neg.predict_proba(X_neg)[:, 0]

        accuracy = (cal_pos.sum() + cal_neg.sum()) / n
        return accuracy


class OracleCAPfromQuantification:

    def __init__(self, h, oracle_quantifier: OracleQuantifier):
        self.h = h
        self.oracle_quantifier = oracle_quantifier

    def predict_accuracy(self, X, y):
        oracle_calibrator = OracleCalibratorFromQuantification(self.h, self.oracle_quantifier)
        oracle_cap = OracleCAPfromCalibration(self.h, oracle_calibrator)
        accuracy = oracle_cap.predict_accuracy(X, y)
        return accuracy

