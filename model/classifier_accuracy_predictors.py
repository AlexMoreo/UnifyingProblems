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
from util import posterior_probabilities


# Adapted from https://github.com/lorenzovolpi/QuAcc/blob/devel


class ClassifierAccuracyPrediction(ABC):

    def __init__(self, classifier: BaseEstimator):
        self.h = classifier

    def classify(self, X):
        return self.h.predict(X)

    def posterior_probabilities(self, X):
        return posterior_probabilities(self.h, X)

    @abstractmethod
    def fit(self, X, y):
        """
        Trains a CAP method.

        :param X: training data
        :param y: labels
        :return: self
        """
        ...

    @abstractmethod
    def predict(self, X) -> float:
        """
        Predicts the accuracy of the classifier on X

        :param X: test data
        :return: float, predicted accuracy
        """
        ...


class ATC(ClassifierAccuracyPrediction):

    VALID_FUNCTIONS = {"maxconf", "neg_entropy"}

    def __init__(self, classifier: BaseEstimator, scoring_fn="maxconf"):
        assert scoring_fn in ATC.VALID_FUNCTIONS, f"unknown scoring function, use any from {ATC.VALID_FUNCTIONS}"
        super().__init__(classifier)
        self.scoring_fn = scoring_fn

    def _get_scores(self, P):
        if self.scoring_fn == "maxconf":
            scores = max_conf(P)
        else:
            scores = neg_entropy(P)
        return scores

    def fit(self, X, y):
        posteriors = self.posterior_probabilities(X)
        pred_labels = np.argmax(posteriors, axis=1)
        scores = self._get_scores(posteriors)
        correct_predictions = (pred_labels == y)
        _, self.threshold = self.__find_ATC_threshold(scores=scores, labels=correct_predictions)
        return self

    def predict(self, X):
        posteriors = self.posterior_probabilities(X)
        scores = self._get_scores(posteriors)
        predicted_acc = self.__get_ATC_acc(self.threshold, scores)
        return predicted_acc

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


class DoC(ClassifierAccuracyPrediction):

    def __init__(self, classifier: BaseEstimator, protocol: AbstractProtocol, clip_vals=(0, 1)):
        super().__init__(classifier)
        self.protocol = protocol
        self.clip_vals = clip_vals

    def _get_post_stats(self, posteriors, y):
        P = posteriors
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        accuracy = (y == pred_labels).mean()
        return mc, accuracy

    def _doc(self, mc1, mc2):
        return mc2.mean() - mc1.mean()

    def _train_regression(self, prot_mcs, prot_accs):
        docs = [self._doc(self.val_mc, prot_mc_i) for prot_mc_i in prot_mcs]
        target = [self.val_acc - prot_acc_i for prot_acc_i in prot_accs]
        docs = np.asarray(docs).reshape(-1, 1)
        target = np.asarray(target)
        lin_reg = LinearRegression()
        return lin_reg.fit(docs, target)

    def _predict_regression(self, test_mc):
        docs = np.asarray([self._doc(self.val_mc, test_mc)]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return self.val_acc - pred_acc

    def fit(self, X, y):
        posteriors = self.posterior_probabilities(X)
        self.val_mc, self.val_acc = self._get_post_stats(posteriors, y)

        prot_stats = [
            self._get_post_stats(self.posterior_probabilities(sample.X), sample.y) for sample in self.protocol()
        ]
        prot_mcs, prot_accs = list(zip(*prot_stats))

        self.reg_model = self._train_regression(prot_mcs, prot_accs)

        return self

    def predict(self, X):
        posteriors = self.posterior_probabilities(X)
        mc = max_conf(posteriors)
        acc_pred = self._predict_regression(mc)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


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


class LasCal2CAP(ClassifierAccuracyPrediction):

    def __init__(self, classifier: BaseEstimator):
        super().__init__(classifier)

    def fit(self, X, y):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        # h(x)=1
        self.Ppos = posteriors[y_hat == 1]
        self.ypos = y[y_hat == 1]

        # h(x)=0
        self.Pneg = posteriors[y_hat == 0]
        self.yneg = y[y_hat == 0]

        return self

    def predict(self, X):
        lascal = LasCalCalibration()

        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        cal_pos = lascal.predict_proba(self.Ppos, self.ypos, posteriors[y_hat==1])
        cal_neg = lascal.predict_proba(self.Pneg, self.yneg, posteriors[y_hat==0])

        n_instances = posteriors.shape[0]

        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances

        return acc_pred


class HDC2CAP(ClassifierAccuracyPrediction):

    def __init__(self, classifier: BaseEstimator):
        super().__init__(classifier)

    def fit(self, X, y):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        # h(x)=1
        Xpos = X[y_hat == 1]
        ypos = y[y_hat == 1]
        Ppos = posteriors[y_hat == 1]

        # h(x)=0
        Xneg = X[y_hat == 0]
        yneg = y[y_hat == 0]
        Pneg = posteriors[y_hat == 0]

        # calibrator for predicted positives
        DMpos = DistributionMatchingY(classifier=self.h, nbins=10)
        preclassified_pos = LabelledCollection(Ppos, ypos)
        data_pos = LabelledCollection(Xpos, ypos)
        DMpos.aggregation_fit(classif_predictions=preclassified_pos, data=data_pos)
        self.cal_pos = HellingerDistanceCalibration(DMpos)

        # calibrator for predicted negatives
        DMneg = DistributionMatchingY(classifier=self.h, nbins=10)
        preclassified_neg = LabelledCollection(Pneg, yneg)
        data_neg = LabelledCollection(Xneg, yneg)
        DMneg.aggregation_fit(classif_predictions=preclassified_neg, data=data_neg)
        self.cal_neg = HellingerDistanceCalibration(DMneg)

        return self

    def predict(self, X):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        cal_pos = self.cal_pos.predict_proba(posteriors[y_hat==1])
        cal_neg = self.cal_neg.predict_proba(posteriors[y_hat==0])

        n_instances = posteriors.shape[0]
        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances

        return acc_pred


class PACC2CAP_(ClassifierAccuracyPrediction):

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


class PACC2CAP(ClassifierAccuracyPrediction):

    def __init__(self, classifier: BaseEstimator):
        super().__init__(classifier)

    def fit(self, X, y):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        # h(x)=1
        Xpos = X[y_hat == 1]
        ypos = y[y_hat == 1]
        Ppos = posteriors[y_hat == 1]

        # h(x)=0
        Xneg = X[y_hat == 0]
        yneg = y[y_hat == 0]
        Pneg = posteriors[y_hat == 0]

        # quantifier for predicted positives
        self.q_pos = PACC(classifier=self.h)
        self.q_pos.aggregation_fit(
            LabelledCollection(Ppos, ypos),
            LabelledCollection(Xpos, ypos),
        )

        # quantifier for predicted negatives
        self.q_neg = PACC(classifier=self.h)
        self.q_neg.aggregation_fit(
            LabelledCollection(Pneg, yneg),
            LabelledCollection(Xneg, yneg)
        )

        return self

    def predict(self, X):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        # predicted prevalence of positive instances in predicted positives
        pos_prev = self.q_pos.aggregate(posteriors[y_hat == 1])[1]

        # predicted prevalence of negative instances in predicted negatives
        neg_prev = self.q_neg.aggregate(posteriors[y_hat == 0])[0]

        n_instances = posteriors.shape[0]
        n_pred_pos = y_hat.sum()
        n_pred_neg = n_instances-n_pred_pos

        acc_pred = (pos_prev*n_pred_pos + neg_prev*n_pred_neg) / n_instances

        return acc_pred