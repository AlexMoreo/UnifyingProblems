from typing import Callable, Literal

import numpy as np
import quapy as qp
import quapy.functional as F
import torch
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeSoftQuantifier, PACC, EMQ
from quapy.method.base import BaseQuantifier
from quapy.protocol import AbstractProtocol
from sklearn.base import BaseEstimator

from lascal import EceLabelShift, get_importance_weights
from model.classifier_accuracy_predictors import ATC, DoC
from model.classifier_calibrators import LasCalCalibration
from util import posterior_probabilities


class ATC2Quant(BaseQuantifier):

    def __init__(self, classifier: BaseEstimator):
        self.h = classifier
        self.cap_pos = ATC(classifier=classifier)
        self.cap_neg = ATC(classifier=classifier)

    def fit(self, data: LabelledCollection):
        X, y = data.Xy
        y_pred = self.h.predict(X)

        # h(x)=1
        Xpos = X[y_pred == 1]
        ypos = y[y_pred == 1]
        self.cap_pos.fit(Xpos, ypos)

        # h(x)=1
        Xneg = X[y_pred == 0]
        yneg = y[y_pred == 0]
        self.cap_neg.fit(Xneg, yneg)

        return self

    def quantify(self, X):
        n = X.shape[0]
        y_pred = self.h.predict(X)

        Xpos = X[y_pred == 1]
        acc_pos = self.cap_pos.predict(Xpos)
        n_pos = Xpos.shape[0]

        Xneg = X[y_pred == 0]
        acc_neg = self.cap_neg.predict(Xneg)
        n_neg = Xneg.shape[0]

        estim_pos_prev = acc_pos*(n_pos/n) + (1-acc_neg)*(n_neg/n)
        estim_prev = F.as_binary_prevalence(estim_pos_prev)

        return estim_prev


class DoC2Quant(BaseQuantifier):

    def __init__(self, classifier: BaseEstimator, protocol_constructor: Callable):
        self.h = classifier
        self.protocol_constructor = protocol_constructor

    def fit(self, data: LabelledCollection):
        X, y = data.Xy
        y_pred = self.h.predict(X)

        # h(x)=1
        Xpos = X[y_pred == 1]
        ypos = y[y_pred == 1]
        pos_prot = self.protocol_constructor(Xpos, ypos, classes=self.h.classes_)
        self.doc_pos = DoC(self.h, protocol=pos_prot)
        self.doc_pos.fit(Xpos, ypos)

        # h(x)=1
        Xneg = X[y_pred == 0]
        yneg = y[y_pred == 0]
        neg_prot = self.protocol_constructor(Xneg, yneg, classes=self.h.classes_)
        self.doc_neg = DoC(self.h, protocol=neg_prot)
        self.doc_neg.fit(Xneg, yneg)

        return self

    def quantify(self, X):
        n = X.shape[0]
        y_pred = self.h.predict(X)

        Xpos = X[y_pred == 1]
        acc_pos = self.doc_pos.predict(Xpos)
        n_pos = Xpos.shape[0]

        Xneg = X[y_pred == 0]
        acc_neg = self.doc_neg.predict(Xneg)
        n_neg = Xneg.shape[0]

        estim_pos_prev = acc_pos * (n_pos / n) + (1 - acc_neg) * (n_neg / n)
        estim_prev = F.as_binary_prevalence(estim_pos_prev)

        return estim_prev


class LasCal2Quant(BaseQuantifier):
    def __init__(self, classifier):
        self.classifier = classifier
        self.lascal = LasCalCalibration()

    def fit(self, data: LabelledCollection):
        X, y = data.Xy
        P = posterior_probabilities(self.classifier, X)
        self.Ptr = P
        self.ytr = y
        return self

    def quantify(self, X):
        P_uncal = posterior_probabilities(self.classifier, X)
        P_calib = self.lascal.predict_proba(self.Ptr, self.ytr, P_uncal)
        prev_estim = np.mean(P_calib, axis=0)
        return prev_estim


class PACCLasCal(PACC):

    def __init__(self,
                 classifier: BaseEstimator=None,
                 val_split=5,
                 solver: Literal['minimize', 'exact', 'exact-raise', 'exact-cc'] = 'minimize',
                 method: Literal['inversion', 'invariant-ratio'] = 'inversion',
                 norm: Literal['clip', 'mapsimplex', 'condsoftmax'] = 'clip',
                 n_jobs=None
    ):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver
        self.method = method
        self.norm = norm

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.Ptr, self.ytr = classif_predictions.Xy
        self.lascal = LasCalCalibration()
        super().aggregation_fit(classif_predictions, data)
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        P_uncal = classif_predictions
        P_calib = self.lascal.predict_proba(self.Ptr, self.ytr, P_uncal)
        return super().aggregate(P_calib)


class EMQLasCal(EMQ):

    def __init__(self, classifier: BaseEstimator = None, val_split=None, n_jobs=None):
        self.classifier = qp._get_classifier(classifier)
        self.val_split = val_split
        self.exact_train_prev = True
        self.recalib = None
        self.n_jobs = n_jobs

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.Ptr, self.ytr = classif_predictions.Xy
        # self.lascal = LasCalCalibration()
        return self

    def aggregate(self, classif_predictions: np.ndarray):
        from scipy.special import softmax
        # P_uncal = classif_predictions
        # P_calib = self.lascal.predict_proba(self.Ptr, self.ytr, P_uncal)
        Ptr = torch.from_numpy(self.Ptr)
        ytr = torch.from_numpy(self.ytr)
        P_uncal = torch.from_numpy(classif_predictions)
        temperature = lascal_fn(Ptr, ytr, P_uncal)
        Ptr = softmax(self.Ptr / temperature, axis=1)
        P_calib = softmax(P_uncal / temperature, axis=1)
        fit_data = LabelledCollection(Ptr, self.ytr)
        super().aggregation_fit(fit_data, fit_data)
        return super().aggregate(P_calib)


# local (simplified) copy of lascal.post_hoc_calibration.calibrator:lascal
def lascal_fn(Str, ytr, Ste):
    import torch
    weights_method = "rlls-hard"
    p = 2
    classwise = True

    ece_criterion = EceLabelShift(
        p=p, n_bins=15, adaptive_bins=True, classwise=classwise
    )
    optim_temp = -1
    best_loss = torch.finfo(torch.float).max

    STEPS = 100
    MIN_TEMP = 0.1
    MAX_TEMP = 20.0
    for temp in torch.linspace(MIN_TEMP, MAX_TEMP, steps=STEPS):
        # Prepare source labels
        num_classes = Str.size(1)
        y_source_ohe = np.eye(num_classes)[ytr.numpy()].astype(
            float
        )
        scaled_logits_source = Str / temp
        scaled_logits_target = Ste / temp
        # Get weight
        output = get_importance_weights(
            valid_preds=scaled_logits_source.softmax(-1).numpy(),
            valid_labels=y_source_ohe,
            shifted_test_preds=scaled_logits_target.softmax(-1).numpy(),
            method=weights_method,
        )
        # Measure loss
        loss = ece_criterion(
            logits_source=scaled_logits_source,
            labels_source=ytr,
            logits=scaled_logits_target,
            weights=output["weights"],
        ).mean()
        if loss < best_loss:
            best_loss = loss
            optim_temp = temp

    return optim_temp
