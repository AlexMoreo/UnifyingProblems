from copy import deepcopy
from typing import Callable
import quapy as qp
from quapy.method.aggregative import AggregativeQuantifier, ACC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy

from abc import ABC, abstractmethod

from quapy.data.base import LabelledCollection
from quapy.protocol import AbstractProtocol
from quapy.method.aggregative import EMQ
import quapy.functional as F
from sklearn.base import BaseEstimator

from model.classifier_calibrators import (
    LasCalCalibration,
    DistributionMatchingCalibration,
    CalibratorCompound
)
from util import posterior_probabilities, accuracy, accuracy_from_contingency_table


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
    def predict(self, X, *args, **kwargs) -> float:
        """
        Predicts the accuracy of the classifier on X

        :param X: test data
        :return: float, predicted accuracy
        """
        ...


class NaiveIID(ClassifierAccuracyPrediction):
    # Estimates classifier accuracy on the validation set (i.e., disregarding the shift effect)

    def __init__(self, classifier: BaseEstimator):
        super().__init__(classifier)

    def fit(self, X, y):
        y_hat = self.classify(X)
        self.estim_acc = accuracy(y_true=y, y_pred=y_hat)
        return self

    def predict(self, X, *args, **kwargs) -> float:
        return self.estim_acc


class ATC(ClassifierAccuracyPrediction):
    # Average Threshold Confidence: https://arxiv.org/abs/2201.04234
    # Leveraging Unlabeled Data to Predict Out-of-Distribution Performance
    # by Saurabh Garg, Sivaraman Balakrishnan, Zachary C. Lipton, Behnam Neyshabur, Hanie Sedghi

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

    def predict(self, X, *args, **kwargs):
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
    # Differences of Confidence: https://arxiv.org/abs/2107.03315
    # Predicting with Confidence on Unseen Distributions
    # by Devin Guillory, Vaishaal Shankar, Sayna Ebrahimi, Trevor Darrell, Ludwig Schmidt

    def __init__(self, classifier: BaseEstimator, protocol: AbstractProtocol, clip_vals=(0, 1)):
        super().__init__(classifier)
        self.protocol = protocol
        self.clip_vals = clip_vals

    def _get_post_stats(self, posteriors, y):
        P = posteriors
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        acc = accuracy(y, pred_labels)
        return mc, acc

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

    def predict(self, X, *args, **kwargs):
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
    return -ne


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
    # Adapts LasCal as a CAP method

    def __init__(self, classifier: BaseEstimator, probs2logits=True):
        super().__init__(classifier)
        self.probs2logits = probs2logits

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

    def predict(self, X, *args, **kwargs):
        lascal = LasCalCalibration(prob2logits=self.probs2logits)

        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        cal_pos = lascal.calibrate(self.Ppos, self.ypos, posteriors[y_hat==1])
        cal_neg = lascal.calibrate(self.Pneg, self.yneg, posteriors[y_hat==0])

        n_instances = posteriors.shape[0]

        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances

        return acc_pred


class CalibratorCompound2CAP(ClassifierAccuracyPrediction):
    # Adapts any CalibratorCompound instance as a CAP method

    def __init__(self, classifier: BaseEstimator, calibrator_cls: CalibratorCompound, Ftr, ytr, probs2logits=True):
        super().__init__(classifier)
        self.calibrator_cls = calibrator_cls
        self.Ftr = Ftr
        self.ytr = ytr
        self.probs2logits = probs2logits

    def fit(self, X, y, hidden=None):
        # if hidden is passed, then the stored features are taken from these;
        # otherwise, the stored features are taken from the raw covariates

        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        features = X if hidden is None else hidden

        # h(x)=1
        self.P_src_pos = posteriors[y_hat == 1]
        self.y_src_pos = y[y_hat == 1]
        self.F_src_pos = features[y_hat == 1]

        # h(x)=0
        self.P_src_neg = posteriors[y_hat == 0]
        self.y_src_neg = y[y_hat == 0]
        self.F_src_neg = features[y_hat == 0]

        return self

    def predict(self, X, hidden=None, *args, **kwargs):
        # if hidden is passed, then the calibration is computed on thoese; 
        # otherwise, calibration is computed on the raw covariates
        calibrator = self.calibrator_cls(prob2logits=self.probs2logits)

        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        features = X if hidden is None else hidden

        F_tgt_pos = features[y_hat == 1]
        F_tgt_neg = features[y_hat == 0]
        P_tgt_pos = posteriors[y_hat == 1]
        P_tgt_neg = posteriors[y_hat == 0]

        cal_pos = calibrator.calibrate(
            Ftr=self.Ftr, ytr=self.ytr,
            Fsrc=self.F_src_pos, Zsrc=self.P_src_pos, ysrc=self.y_src_pos,
            Ftgt=F_tgt_pos, Ztgt=P_tgt_pos
        )
        cal_neg = calibrator.calibrate(
            Ftr=self.Ftr, ytr=self.ytr,
            Fsrc=self.F_src_neg, Zsrc=self.P_src_neg, ysrc=self.y_src_neg,
            Ftgt=F_tgt_neg, Ztgt=P_tgt_neg
        )

        n_instances = posteriors.shape[0]

        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances

        return acc_pred


class DMCal2CAP(ClassifierAccuracyPrediction):
    # Adapts DMCal as a CAP method

    def __init__(self, classifier: BaseEstimator):
        super().__init__(classifier)

    def fit(self, X, y):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        # h(x)=1
        ypos = y[y_hat == 1]
        Ppos = posteriors[y_hat == 1]

        # h(x)=0
        yneg = y[y_hat == 0]
        Pneg = posteriors[y_hat == 0]

        # calibrator for predicted positives
        self.cal_pos = DistributionMatchingCalibration(self.h, nbins=10).fit(Ppos, ypos)

        # calibrator for predicted negatives
        self.cal_neg = DistributionMatchingCalibration(self.h, nbins=10).fit(Pneg, yneg)

        return self

    def predict(self, X, *args, **kwargs):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        cal_pos = self.cal_pos.calibrate(posteriors[y_hat==1])
        cal_neg = self.cal_neg.calibrate(posteriors[y_hat==0])

        n_instances = posteriors.shape[0]
        acc_pred = (cal_pos[:,1].sum() + cal_neg[:,0].sum()) / n_instances

        return acc_pred


def check_posteriors(q, posteriors):
    # the recalibration in quapy's EMQ is only invoked if the method "quantify" is called;
    # but not if the method "aggregate" is called. This should be fixed in the new version
    if qp.__version__ in ['0.1.8', '0.1.9'] and isinstance(q, EMQ) and q.recalib is not None:
        if q.calibration_function is not None:
            posteriors = q.calibration_function(posteriors)
    return posteriors


class Quant2CAP(ClassifierAccuracyPrediction):
    # Adapts any quantification method as a CAP method

    def __init__(self, classifier: BaseEstimator, quantifier_class):
        super().__init__(classifier)
        self.quantifier_class = quantifier_class

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
        self.q_pos = self.quantifier_class(classifier=self.h)
        self.q_pos.aggregation_fit(
            LabelledCollection(Ppos, ypos),
            LabelledCollection(Xpos, ypos),
        )

        # quantifier for predicted negatives
        self.q_neg = self.quantifier_class(classifier=self.h)
        self.q_neg.aggregation_fit(
            LabelledCollection(Pneg, yneg),
            LabelledCollection(Xneg, yneg)
        )

        return self

    def predict(self, X, *args, **kwargs):
        y_hat = self.classify(X)
        posteriors = self.posterior_probabilities(X)

        n_instances = posteriors.shape[0]
        n_pred_pos = y_hat.sum()
        n_pred_neg = n_instances - n_pred_pos

        # predicted prevalence of positive instances in predicted positives
        if n_pred_pos>0:
            pos_posteriors = posteriors[y_hat == 1]
            pos_posteriors = check_posteriors(self.q_pos, pos_posteriors)
            pos_prev = self.q_pos.aggregate(pos_posteriors)[1]
        else:
            pos_prev = 0

        # predicted prevalence of negative instances in predicted negatives
        if n_pred_neg>0:
            neg_posteriors = posteriors[y_hat == 0]
            neg_posteriors = check_posteriors(self.q_neg, neg_posteriors)
            neg_prev = self.q_neg.aggregate(neg_posteriors)[0]
        else:
            neg_prev = 0

        acc_pred = (pos_prev*n_pred_pos + neg_prev*n_pred_neg) / n_instances

        return acc_pred


class LEAP(ClassifierAccuracyPrediction):
    # Linear-Equation-based Acuracy Prediction: https://link.springer.com/chapter/10.1007/978-3-031-78980-9_17
    # A Simple Method for Classifier Accuracy Prediction Under Prior Probability Shift
    # by Lorenzo Volpi, Alejandro Moreo, Fabrizio Sebastiani

    def __init__(self,
                 classifier,
                 q_class: AggregativeQuantifier = ACC(),
                 acc_fn: Callable= accuracy_from_contingency_table):
        super().__init__(classifier)
        self.q_class = q_class
        self.acc_fn = acc_fn

    def predict(self, X, *args, **kwargs):
        """
        Predicts the contingency table for the test data

        :param X: test data
        :param posteriors: posterior probabilities of X
        :return: the accuracy function as computed from a contingency table
        """
        posteriors = self.posterior_probabilities(X)
        cont_table = self.predict_ct(X, posteriors)
        return self.acc_fn(cont_table)

    def _prepare_quantifier(self):
        assert isinstance(self.q_class, AggregativeQuantifier), (
            f"quantifier {self.q_class} is not of type aggregative"
        )
        self.q = deepcopy(self.q_class)
        self.q.set_params(classifier=self.h)

    def fit(self, X, y):
        data = LabelledCollection(X,y)
        posteriors = self.posterior_probabilities(X)
        data = self._preprocess_data(data, posteriors)
        self._prepare_quantifier()
        classif_predictions = self.q.classifier_fit_predict(data, fit_classifier=False, predict_on=data)
        self.q.aggregation_fit(classif_predictions, data)
        return self

    def _preprocess_data(self, data: LabelledCollection, posteriors):
        self.classes_ = data.classes_
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]

        # I is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        I = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n * n, n * n))
        b = np.zeros(shape=(n * n))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the
        # PPS assumptions. Example in three classes, a ratio: a/(a+b+c) [test] = ar [a ratio in training]
        # a / (a + b + c) = ar
        # a = (a + b + c) * ar
        # a = a ar + b ar + c ar
        # a - a ar - b ar - c ar = 0
        # a (1-ar) + b (-ar)  + c (-ar) = 0
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(1, n):
            for j in range(1, n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, I[i, :]] = -ratio_ij
                A[eq_no, I[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(1, n):
            A[eq_no, I[:, i]] = 1
            # b[eq_no] = cc_prev_estim[i]
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(1, n):
            A[eq_no, I[i, :]] = 1
            # b[eq_no] = q_prev_estim[i]
            eq_no += 1

        return A, b

    def predict_ct(self, test, posteriors):
        """
        :param test: test instances
        :param posteriors: posterior probabilities of test instances
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = F.prevalence_from_labels(h_label_preds, self.classes_)
        q_prev_estim = self.q.quantify(test)

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * (n - 1) : -(n - 1)] = cc_prev_estim[1:]
        b[-(n - 1) :] = q_prev_estim[1:]

        # try the fast solution (may not be valid)
        x = np.linalg.solve(A, b)

        if any(x < 0) or not np.isclose(x.sum(), 1):

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            x = F.optim_minimize(loss, n_classes=n**2)

        cont_table_test = x.reshape(n, n)
        return cont_table_test