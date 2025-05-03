from copy import copy
from typing import Callable, Literal

import numpy as np
import quapy as qp
import quapy.functional as F
import torch
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeSoftQuantifier, PACC, EMQ, PCC, KDEyML
from quapy.method.base import BaseQuantifier
from quapy.protocol import AbstractProtocol
from sklearn.base import BaseEstimator, clone

from lascal import EceLabelShift, get_importance_weights
from model.classifier_accuracy_predictors import ClassifierAccuracyPrediction, ATC, DoC, LEAP
from model.classifier_calibrators import LasCalCalibration, TransCalCalibrator, HeadToTailCalibrator, CpcsCalibrator, \
    CalibratorSimple
from util import posterior_probabilities
from abc import abstractmethod


class Method2Quant(BaseQuantifier):
    @abstractmethod
    def fit(self, data, *args, **kwargs):
        ...

    @abstractmethod
    def quantify(self, instances, *args, **kwargs):
        ...


class CAP2Quant(Method2Quant):
     
    def __init__(self, classifier: BaseEstimator, cap_instance: ClassifierAccuracyPrediction):
        self.h = classifier
        self.cap_pos = copy(cap_instance)
        self.cap_neg = copy(cap_instance)
        
    def fit(self, data: LabelledCollection, *args, **kwargs):
        X, y = data.Xy
        y_pred = self.h.predict(X)

        # h(x)=1
        pred_positives = y_pred==1
        self.cap_pos.fit(X[pred_positives], y[pred_positives])

        # h(x)=1
        pred_negatives = y_pred==0
        self.cap_neg.fit(X[pred_negatives], y[pred_negatives])

        return self

    def quantify(self, X, *args, **kwargs):
        n = X.shape[0]
        y_pred = self.h.predict(X)

        pred_positives = y_pred==1
        n_pos = sum(pred_positives)
        acc_pos = self.cap_pos.predict(X[pred_positives]) if n_pos > 0 else 0


        pred_negatives = y_pred==0
        n_neg = sum(pred_negatives)
        acc_neg = self.cap_neg.predict(X[pred_negatives]) if n_neg > 0 else 0

        estim_pos_prev = acc_pos*(n_pos/n) + (1-acc_neg)*(n_neg/n)
        estim_prev = F.as_binary_prevalence(estim_pos_prev)

        return estim_prev


class ATC2Quant(CAP2Quant):
    def __init__(self, classifier):
        super().__init__(classifier, ATC(classifier=classifier))
    

class LEAP2Quant(CAP2Quant):
    def __init__(self, classifier, quantifier_cls=KDEyML):
        super().__init__(classifier, LEAP(classifier=classifier, q_class=quantifier_cls(classifier=classifier)))


class DoC2Quant(Method2Quant):

    def __init__(self, classifier: BaseEstimator, protocol_constructor: Callable):
        self.h = classifier
        self.protocol_constructor = protocol_constructor

    def fit(self, data: LabelledCollection, *args, **kwargs):
        X, y = data.Xy
        y_pred = self.h.predict(X)

        # h(x)=1
        pred_positives = y_pred==1
        Xpos = X[pred_positives]
        ypos = y[pred_positives]
        pos_prot = self.protocol_constructor(Xpos, ypos, classes=self.h.classes_)
        self.doc_pos = DoC(self.h, protocol=pos_prot).fit(Xpos, ypos)

        # h(x)=1
        pred_negatives = y_pred==0
        Xneg = X[pred_negatives]
        yneg = y[pred_negatives]
        neg_prot = self.protocol_constructor(Xneg, yneg, classes=self.h.classes_)
        self.doc_neg = DoC(self.h, protocol=neg_prot).fit(Xneg, yneg)

        return self

    def quantify(self, X, *args, **kwargs):
        n = X.shape[0]
        y_pred = self.h.predict(X)

        pred_positives = y_pred==1
        n_pos = sum(pred_positives)
        acc_pos = self.doc_pos.predict(X[pred_positives]) if n_pos>0 else 0

        pred_negatives = y_pred==0
        n_neg = sum(pred_negatives)
        acc_neg = self.doc_neg.predict(X[pred_negatives]) if n_neg>0 else 0

        estim_pos_prev = acc_pos * (n_pos / n) + (1 - acc_neg) * (n_neg / n)
        estim_prev = F.as_binary_prevalence(estim_pos_prev)

        return estim_prev


class LasCal2Quant(Method2Quant):
    def __init__(self, classifier, prob2logits=True):
        self.classifier = classifier
        self.lascal = LasCalCalibration(prob2logits)

    def fit(self, data: LabelledCollection, *args, **kwargs):
        X, y = data.Xy
        P = posterior_probabilities(self.classifier, X)
        self.Ptr = P
        self.ytr = y
        return self

    def quantify(self, X, *args, **kwargs):
        P_uncal = posterior_probabilities(self.classifier, X)
        P_calib = self.lascal.calibrate(self.Ptr, self.ytr, P_uncal)
        prev_estim = np.mean(P_calib, axis=0)
        return prev_estim


class EMLasCal2Quant(Method2Quant):
    def __init__(self, classifier, prob2logits=True):
        self.classifier = classifier
        self.lascal = LasCalCalibration(prob2logits)

    def fit(self, data: LabelledCollection, *args, **kwargs):
        self.train_prevalence = data.prevalence()
        X, y = data.Xy
        P = posterior_probabilities(self.classifier, X)
        self.Ptr = P
        self.ytr = y
        return self

    def quantify(self, X, *args, **kwargs):
        P_uncal = posterior_probabilities(self.classifier, X)
        P_lascal = self.lascal.calibrate(self.Ptr, self.ytr, P_uncal)
        priors, posteriors = EMQ.EM(tr_prev=self.train_prevalence, posterior_probabilities=P_lascal)
        return priors


class CalibratorCompound2Quant(Method2Quant):
    def __init__(self, classifier, Ftr, ytr, calibrator_cls, prob2logits=True):
        self.classifier = classifier
        self.calib = calibrator_cls(prob2logits)
        self.Ftr = Ftr
        self.ytr = ytr

    def fit(self, data: LabelledCollection, hidden=None, *args, **kwargs):
        X, y = data.Xy
        P = posterior_probabilities(self.classifier, X)
        
        Fsrc = X if hidden is None else hidden
        self.Fsrc = Fsrc
        self.Psrc = P
        self.ysrc = y
        return self

    def quantify(self, X, hidden=None, *args, **kwargs):
        P_uncal = posterior_probabilities(self.classifier, X)
        
        Ftgr = X if hidden is None else hidden

        P_calib = self.calib.calibrate(
            Ftr=self.Ftr, ytr=self.ytr, Fsrc=self.Fsrc, Zsrc=self.Psrc, ysrc=self.ysrc, Ftgt=Ftgr, Ztgt=P_uncal
        )
        prev_estim = np.mean(P_calib, axis=0)
        return prev_estim


class Transcal2Quant(CalibratorCompound2Quant):
    def __init__(self, classifier, Ftr, ytr, prob2logits=True):
        super().__init__(classifier, Ftr, ytr, calibrator_cls=TransCalCalibrator, prob2logits=prob2logits)


class Cpcs2Quant(CalibratorCompound2Quant):
    def __init__(self, classifier, Ftr, ytr, prob2logits=True):
        super().__init__(classifier, Ftr, ytr, calibrator_cls=CpcsCalibrator, prob2logits=prob2logits)


# class HeadToTail2Quant(CalibratorCompound2Quant):
#     def __init__(self, classifier, Ftr, ytr, prob2logits=True):
#         super().__init__(classifier, Ftr, ytr, calibrator_cls=HeadToTailCalibrator, prob2logits=prob2logits)

class HeadToTail2Quant(Method2Quant):
    def __init__(self, classifier, Ftr, ytr, prob2logits=True, n_components=None):
        self.classifier = classifier
        self.Ftr = Ftr
        self.ytr = ytr
        self.head2tails = HeadToTailCalibrator(prob2logits, n_components=n_components)

    def fit(self, data: LabelledCollection, hidden=None, *args, **kwargs):
        X, y = data.Xy
        P = posterior_probabilities(self.classifier, X)
        Fsrc = X if hidden is None else hidden
        self.head2tails.fit(Ftr=self.Ftr, ytr=self.ytr, Fsrc=Fsrc, Zsrc=P, ysrc=y)
        return self

    def quantify(self, instances, *args, **kwargs):
        P_uncal = posterior_probabilities(self.classifier, instances)
        P_cal = self.head2tails.calibrate(P_uncal)
        prev_estim = P_cal.mean(axis=0)
        return prev_estim




class PACCLasCal(PACC):

    def __init__(self,
                 classifier: BaseEstimator=None,
                 val_split=5,
                 solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
                 method: Literal['inversion', 'invariant-ratio'] = 'inversion',
                 norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
                 n_jobs=None,
                 prob2logits=True
    ):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver
        self.method = method
        self.norm = norm
        self.prob2logits = prob2logits

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.Ptr, self.ytr = classif_predictions.Xy
        self.lascal = LasCalCalibration(prob2logits=self.prob2logits)
        super().aggregation_fit(classif_predictions, data)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        P_uncal = classif_predictions
        P_calib = self.lascal.calibrate(self.Ptr, self.ytr, P_uncal)
        return super().aggregate(P_calib)


class EMQ_BCTS(EMQ):
    def __init__(self, classifier = None):
        super().__init__(classifier, val_split=None, exact_train_prev=False, recalib='bcts', n_jobs=None)

    def fit(self, data, fit_classifier=True, val_split=None):
        try:
            return super().fit(data, fit_classifier, val_split)
        except AssertionError:
            print('Abstention raised an error. Backing up to EMQ without recalibration')
            self.val_split=None
            self.exact_train_prev=True
            self.recalib=None
            return self.fit(data, fit_classifier, val_split)

