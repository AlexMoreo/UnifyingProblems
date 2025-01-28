import itertools as IT
from copy import deepcopy
from typing import Callable

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier, BaseQuantifier, DistributionMatchingY, PACC
from quapy.protocol import UPP, AbstractProtocol
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import scipy


import itertools as IT
from abc import ABC, abstractmethod
from typing import List

from quapy.data.base import LabelledCollection
from quapy.protocol import AbstractProtocol
from sklearn.base import BaseEstimator

from model.classifier_calibrators import LasCalCalibration, HellingerDistanceCalibration


# Adapted from https://github.com/lorenzovolpi/QuAcc/blob/devel


class ClassifierAccuracyPrediction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, val: LabelledCollection, posteriors):
        """
        Trains a CAP method.

        :param val: training data
        :return: self
        """
        ...

    @abstractmethod
    def predict(self, X, posteriors, oracle_prev=None) -> float:
        """
        Predicts directly the accuracy using the accuracy function

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        ...

    def batch_predict(self, prot: AbstractProtocol, posteriors, oracle_prevs=None) -> list[float]:
        if oracle_prevs is None:
            estim_accs = [self.predict(Ui.X, posteriors=P) for Ui, P in IT.zip_longest(prot(), posteriors)]
            return estim_accs
        else:
            assert isinstance(oracle_prevs, List), "Invalid oracles"
            estim_accs = [
                self.predict(Ui.X, P, oracle_prev=op) for Ui, P, op in IT.zip_longest(prot(), posteriors, oracle_prevs)
            ]
            return estim_accs


class CAPDirect(ClassifierAccuracyPrediction):
    def __init__(self, acc: Callable):
        super().__init__()
        self.acc = acc

    def true_acc(self, sample: LabelledCollection, posteriors):
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)

    def switch_and_fit(self, acc_fn, data, posteriors):
        self.acc = acc_fn
        return self.fit(data, posteriors)



class ATC(CAPDirect):

    VALID_FUNCTIONS = {"maxconf", "neg_entropy"}

    def __init__(self, acc_fn: Callable, scoring_fn="maxconf"):
        assert scoring_fn in ATC.VALID_FUNCTIONS, f"unknown scoring function, use any from {ATC.VALID_FUNCTIONS}"
        super().__init__(acc_fn)
        self.scoring_fn = scoring_fn

    def get_scores(self, P):
        if self.scoring_fn == "maxconf":
            scores = max_conf(P)
        else:
            scores = neg_entropy(P)
        return scores

    def fit(self, val: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=1)
        true_labels = val.y
        scores = self.get_scores(posteriors)
        _, self.threshold = self.__find_ATC_threshold(scores=scores, labels=(pred_labels == true_labels))
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        scores = self.get_scores(posteriors)
        return self.__get_ATC_acc(self.threshold, scores)

    def __find_ATC_threshold(self, scores, labels):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        sorted_idx = np.argsort(scores)

        sorted_scores = scores[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_scores[i]

        return min_fp_fn, thres

    def __get_ATC_acc(self, thres, scores):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        return np.mean(scores >= thres)


class DoC(CAPDirect):
    def __init__(self, acc_fn: Callable, protocol: AbstractProtocol, prot_posteriors, clip_vals=(0, 1)):
        super().__init__(acc_fn)
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.clip_vals = clip_vals

    def _get_post_stats(self, X, y, posteriors):
        P = posteriors
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        acc = self.acc(y, pred_labels)
        return mc, acc

    def _doc(self, mc1, mc2):
        return mc2.mean() - mc1.mean()

    def train_regression(self, prot_mcs, prot_accs):
        docs = [self._doc(self.val_mc, prot_mc_i) for prot_mc_i in prot_mcs]
        target = [self.val_acc - prot_acc_i for prot_acc_i in prot_accs]
        docs = np.asarray(docs).reshape(-1, 1)
        target = np.asarray(target)
        lin_reg = LinearRegression()
        return lin_reg.fit(docs, target)

    def predict_regression(self, test_mc):
        docs = np.asarray([self._doc(self.val_mc, test_mc)]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return self.val_acc - pred_acc

    def fit(self, val: LabelledCollection, posteriors):
        self.val_mc, self.val_acc = self._get_post_stats(*val.Xy, posteriors)

        prot_stats = [
            self._get_post_stats(*sample.Xy, P) for sample, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        prot_mcs, prot_accs = list(zip(*prot_stats))

        self.reg_model = self.train_regression(prot_mcs, prot_accs)

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        mc = max_conf(posteriors)
        acc_pred = self.predict_regression(mc)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


def get_posteriors_from_h(h, X):
    if hasattr(h, "predict_proba"):
        P = h.predict_proba(X)
    else:
        n_classes = len(h.classes_)
        dec_scores = h.decision_function(X)
        if n_classes == 1:
            dec_scores = np.vstack([-dec_scores, dec_scores]).T
        P = scipy.special.softmax(dec_scores, axis=1)
    return P


def max_conf(P, keepdims=False):
    mc = P.max(axis=1, keepdims=keepdims)
    return mc


def neg_entropy(P, keepdims=False):
    ne = scipy.stats.entropy(P, axis=1)
    if keepdims:
        ne = ne.reshape(-1, 1)
    return ne


def max_inverse_softmax(P, keepdims=False):
    P = smooth(P, epsilon=1e-12, axis=1)
    lgP = np.log(P)
    mis = np.max(lgP - lgP.mean(axis=1, keepdims=True), axis=1, keepdims=keepdims)
    return mis


def smooth(prevalences, epsilon=1e-5, axis=None):
    """
    Smooths a prevalence vector.

    :param prevalences: np.ndarray
    :param epsilon: float, a small quantity (default 1E-5)
    :return: smoothed prevalence vector
    """
    prevalences = prevalences + epsilon
    prevalences /= prevalences.sum(axis=axis, keepdims=axis is not None)
    return prevalences


class LasCal2CAP(CAPDirect):

    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def fit(self, val: LabelledCollection, posteriors):
        X, y = val.Xy
        y_hat = self.classifier.predict(X)

        # Xpos = X[y_hat == 1]
        self.ypos = y[y_hat == 1]
        self.Ppos = posteriors[y_hat == 1]

        Xneg = X[y_hat == 0]
        self.yneg = y[y_hat == 0]
        self.Pneg = posteriors[y_hat == 0]

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        lascal = LasCalCalibration()
        y_hat = self.classifier.predict(X)
        cal_pos = lascal.predict_proba(self.Ppos, self.ypos, posteriors[y_hat==1])
        cal_neg = lascal.predict_proba(self.Pneg, self.yneg, posteriors[y_hat==0])
        n_instances = posteriors.shape[0]
        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances
        return acc_pred


class HDC2CAP(CAPDirect):

    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def fit(self, val: LabelledCollection, posteriors):
        X, y = val.Xy
        y_hat = self.classifier.predict(X)

        Xpos = X[y_hat == 1]
        ypos = y[y_hat == 1]
        Ppos = posteriors[y_hat == 1]

        Xneg = X[y_hat == 0]
        yneg = y[y_hat == 0]
        Pneg = posteriors[y_hat == 0]

        DMpos = DistributionMatchingY(classifier=self.classifier, nbins=10)
        preclassified_pos = LabelledCollection(Ppos, ypos)
        data_pos = LabelledCollection(Xpos, ypos)
        DMpos.aggregation_fit(classif_predictions=preclassified_pos, data=data_pos)
        self.cal_pos = HellingerDistanceCalibration(DMpos)

        DMneg = DistributionMatchingY(classifier=self.classifier, nbins=10)
        preclassified_neg = LabelledCollection(Pneg, yneg)
        data_neg = LabelledCollection(Xneg, yneg)
        DMneg.aggregation_fit(classif_predictions=preclassified_neg, data=data_neg)
        self.cal_neg = HellingerDistanceCalibration(DMneg)

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        y_hat = self.classifier.predict(X)
        cal_pos = self.cal_pos.predict_proba(posteriors[y_hat==1])
        cal_neg = self.cal_neg.predict_proba(posteriors[y_hat==0])
        n_instances = posteriors.shape[0]
        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances
        return acc_pred


class PACC2CAP_(CAPDirect):

    def __init__(self, classifier: BaseEstimator, from_posteriors=True):
        self.classifier = classifier
        self.from_posteriors = from_posteriors

    def fit(self, val: LabelledCollection, posteriors):
        X, y = val.Xy
        y_hat = self.classifier.predict(X)

        Cov_pos = posteriors[y_hat == 1] if self.from_posteriors else X[y_hat == 1]
        # Xpos = X[y_hat == 1]
        ypos = y[y_hat == 1]
        # Ppos = posteriors[y_hat == 1]

        Cov_neg = posteriors[y_hat == 0] if self.from_posteriors else X[y_hat == 0]
        # Xneg = X[y_hat == 0]
        yneg = y[y_hat == 0]
        # Pneg = posteriors[y_hat == 0]

        self.q_pos = PACC()
        self.q_pos.fit(LabelledCollection(Cov_pos, ypos))

        self.q_neg = PACC()
        self.q_neg.fit(LabelledCollection(Cov_neg, yneg))

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        y_hat = self.classifier.predict(X)
        Cov_pos = posteriors[y_hat == 1] if self.from_posteriors else X[y_hat == 1]
        Cov_neg = posteriors[y_hat == 0] if self.from_posteriors else X[y_hat == 0]
        pos_prev = self.q_pos.quantify(Cov_pos)[1]
        neg_prev = self.q_neg.quantify(Cov_neg)[0]
        n_instances = posteriors.shape[0]
        n_pred_pos = y_hat.sum()
        n_pred_neg = n_instances-n_pred_pos
        acc_pred = (pos_prev*n_pred_pos + neg_prev*n_pred_neg) / n_instances
        return acc_pred


class PACC2CAP(CAPDirect):

    def __init__(self, classifier: BaseEstimator): #, from_posteriors=True):
        self.classifier = classifier
        # self.from_posteriors = from_posteriors

    def fit(self, val: LabelledCollection, posteriors):
        X, y = val.Xy
        y_hat = self.classifier.predict(X)

        # Cov_pos = posteriors[y_hat == 1] if self.from_posteriors else X[y_hat == 1]
        Xpos = X[y_hat == 1]
        ypos = y[y_hat == 1]
        Ppos = posteriors[y_hat == 1]

        # Cov_neg = posteriors[y_hat == 0] if self.from_posteriors else X[y_hat == 0]
        Xneg = X[y_hat == 0]
        yneg = y[y_hat == 0]
        Pneg = posteriors[y_hat == 0]

        self.q_pos = PACC(classifier=self.classifier)
        self.q_pos.aggregation_fit(
            LabelledCollection(Ppos, ypos),
            LabelledCollection(Xpos, ypos),
        )

        self.q_neg = PACC(classifier=self.classifier)
        self.q_neg.aggregation_fit(
            LabelledCollection(Pneg, yneg),
            LabelledCollection(Xneg, yneg)
        )

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        y_hat = self.classifier.predict(X)
        Ppos = posteriors[y_hat == 1]
        Pneg = posteriors[y_hat == 0]
        pos_prev = self.q_pos.aggregate(Ppos)[1]
        neg_prev = self.q_neg.aggregate(Pneg)[0]
        n_instances = posteriors.shape[0]
        n_pred_pos = y_hat.sum()
        n_pred_neg = n_instances-n_pred_pos
        acc_pred = (pos_prev*n_pred_pos + neg_prev*n_pred_neg) / n_instances
        return acc_pred