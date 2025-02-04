from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

from model.classifier_accuracy_predictors import ATC, DoC, LasCal2CAP, HDC2CAP, PACC2CAP, LEAP, NaiveIID
import quapy as qp
from tqdm import tqdm
import numpy as np

from util import datasets


def accuracy(y_true, y_pred):
    return (y_true==y_pred).mean()


def cap_error(acc_true, acc_estim):
    return abs(acc_true-acc_estim)

DOC_VAL_SAMPLES = 50
REPEATS = 100

"""
Methods:
- Naive [baseline]
- LEAP, ATC, DoC [proper CAP methods]
- Lascal2Cap [a method from calibration]
- PACC2C [a method from quantification]
- HDC2C [a method from calibration that comes from quantification]
"""

naive_ERR = []
leap_ERR = []
atc_ERR = []
doc_ERR = []
lascal2cap_ERR = []
hdc2cap_ERR = []
pacc2cap_ERR = []

for dataset in datasets():

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy

    h = LogisticRegression()
    # Ptr = cross_val_predict(lr, Xtr, ytr, cv=5, n_jobs=-1, method='predict_proba')
    h.fit(Xtr, ytr)

    Xva, yva = val.Xy

    naive = NaiveIID(classifier=h)
    naive.fit(Xva, yva)

    leap = LEAP(classifier=h, q_class=KDEyML(classifier=h))
    leap.fit(Xva, yva)

    atc = ATC(h)
    atc.fit(Xva, yva)

    val_prot = UPP(val, sample_size=len(val), repeats=DOC_VAL_SAMPLES, random_state=0, return_type='labelled_collection')
    doc = DoC(h, protocol=val_prot)
    doc.fit(Xva, yva)

    lascal2cap = LasCal2CAP(classifier=h)
    lascal2cap.fit(Xva, yva)

    hdc2cap = HDC2CAP(classifier=h)
    hdc2cap.fit(Xva, yva)

    pacc2cap = PACC2CAP(classifier=h)
    pacc2cap.fit(Xva, yva)

    naive_err = []
    leap_err = []
    atc_err = []
    doc_err = []
    lascal2cap_err = []
    hdc2cap_err = []
    pacc2cap_err = []

    app = UPP(test, sample_size=len(test), repeats=REPEATS, return_type='labelled_collection')
    for test_shifted in tqdm(app(), total=app.total()):
        Xte, yte = test_shifted.Xy

        y_pred = h.predict(Xte)
        true_acc = accuracy(y_true=yte, y_pred=y_pred)

        naive_acc = naive.predict(Xte)
        leap_acc = leap.predict(Xte)
        atc_acc = atc.predict(Xte)
        doc_acc = doc.predict(Xte)
        lascal2cap_acc = lascal2cap.predict(Xte)
        hdc2cap_acc = hdc2cap.predict(Xte)
        pacc2cap_acc = pacc2cap.predict(Xte)

        naive_err.append(cap_error(acc_true=true_acc, acc_estim=naive_acc))
        leap_err.append(cap_error(acc_true=true_acc, acc_estim=leap_acc))
        atc_err.append(cap_error(acc_true=true_acc, acc_estim=atc_acc))
        doc_err.append(cap_error(acc_true=true_acc, acc_estim=doc_acc))
        lascal2cap_err.append(cap_error(acc_true=true_acc, acc_estim=lascal2cap_acc))
        hdc2cap_err.append(cap_error(acc_true=true_acc, acc_estim=hdc2cap_acc))
        pacc2cap_err.append(cap_error(acc_true=true_acc, acc_estim=pacc2cap_acc))

    print(dataset)
    print(f'Naive {np.mean(naive_err):.4f}')
    print(f'LEAP {np.mean(leap_err):.4f}')
    print(f'ATC {np.mean(atc_err):.4f}')
    print(f'DoC {np.mean(doc_err):.4f}')
    print(f'L2C {np.mean(lascal2cap_err):.4f}')
    print(f'HDC2C {np.mean(hdc2cap_err):.4f}')
    print(f'PACC2C {np.mean(pacc2cap_err):.4f}')

    naive_ERR.append(np.mean(naive_err))
    leap_ERR.append(np.mean(leap_err))
    atc_ERR.append(np.mean(atc_err))
    doc_ERR.append(np.mean(doc_err))
    lascal2cap_ERR.append(np.mean(lascal2cap_err))
    hdc2cap_ERR.append(np.mean(hdc2cap_err))
    pacc2cap_ERR.append(np.mean(pacc2cap_err))


print('End. Averages:')
print(f'Naive {np.mean(naive_ERR):.4f}')
print(f'LEAP {np.mean(leap_ERR):.4f}')
print(f'ATC {np.mean(atc_ERR):.4f}')
print(f'DoC {np.mean(doc_ERR):.4f}')
print(f'L2C {np.mean(lascal2cap_ERR):.4f}')
print(f'HDC2C {np.mean(hdc2cap_ERR):.4f}')
print(f'PACC2C {np.mean(pacc2cap_ERR):.4f}')





