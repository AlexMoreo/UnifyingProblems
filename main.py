import numpy as np
import quapy as qp
from quapy.error import mae
from quapy import functional as F
import scipy.special
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC, AggregativeSoftQuantifier, EMQ
from quapy.method.base import BaseQuantifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
import torch
from scipy.special import softmax

from lascal import Calibrator

dataset=qp.datasets.UCI_BINARY_DATASETS[0]

data = qp.datasets.fetch_UCIBinaryDataset(dataset)
train, test = data.train_test

test = test.sampling(len(test), 0.8, 0.2, random_state=1)

Xtr, ytr = train.Xy
Xte, yte = test.Xy


lr = LogisticRegression()

Ptr = cross_val_predict(lr, Xtr, ytr, cv=3, n_jobs=-1, method='predict_proba')
lr.fit(Xtr, ytr)
Pte = lr.predict_proba(Xte)





lascal_ece = LasCalQuantifier(LogisticRegression(), criterion='ece')
lascal_ece.fit(train)

lascal_cross = LasCalQuantifier(LogisticRegression(), criterion='cross_entropy')
lascal_cross.fit(train)

pacc = PACC()
pacc.fit(train)

emq = EMQ(LogisticRegression())
emq.fit(train)

print('[done]')

lascal_ece_estim_prev = lascal_ece.quantify(test.X)
lascal_cross_estim_prev = lascal_cross.quantify(test.X)
pacc_estim_prev = pacc.quantify(test.X)
emq_estim_prev = emq.quantify(test.X)

true_prev  = test.prevalence()

print('true-prev', F.strprev(true_prev))
print('LasCal(ece)-estim-prev', F.strprev(lascal_ece_estim_prev), mae(true_prev, lascal_ece_estim_prev))
print('LasCal(cross)-estim-prev', F.strprev(lascal_cross_estim_prev), mae(true_prev, lascal_cross_estim_prev))
print('EMQ-estim-prev', F.strprev(emq_estim_prev), mae(true_prev, emq_estim_prev))
print('PACC-estim-prev', F.strprev(pacc_estim_prev), mae(true_prev, pacc_estim_prev))


print(pacc.classifier.decision_function(test.X)[:10])
print(pacc.classifier.predict_proba(test.X)[:10])