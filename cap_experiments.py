from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

from model.classifier_accuracy_predictors import ATC, DoC, LasCal2CAP, HDC2CAP, PACC2CAP
import quapy as qp
from tqdm import tqdm
import numpy as np


def accuracy(y_true, y_pred):
    return (y_true==y_pred).mean()


def cap_error(acc_true, acc_estim):
    return abs(acc_true-acc_estim)


for dataset in qp.datasets.UCI_BINARY_DATASETS[:10]:

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train, val = train.split_stratified(0.5, random_state=0)

    Xtr, ytr = train.Xy

    h = LogisticRegression()
    # Ptr = cross_val_predict(lr, Xtr, ytr, cv=5, n_jobs=-1, method='predict_proba')
    h.fit(Xtr, ytr)

    Xva, yva = val.Xy


    atc = ATC(h)
    atc.fit(Xva, yva)

    val_prot = UPP(val, sample_size=len(val), repeats=50, random_state=0, return_type='labelled_collection')
    doc = DoC(h, protocol=val_prot)
    doc.fit(Xva, yva)

    lascal2cap = LasCal2CAP(classifier=h)
    lascal2cap.fit(Xva, yva)

    hdc2cap = HDC2CAP(classifier=h)
    hdc2cap.fit(Xva, yva)

    pacc2cap = PACC2CAP(classifier=h)
    pacc2cap.fit(Xva, yva)

    atc_err = []
    doc_err = []
    lascal2cap_err = []
    hdc2cap_err = []
    pacc2cap_err = []

    app = UPP(test, sample_size=len(test), repeats=10, return_type='labelled_collection')
    for test_shifted in tqdm(app(), total=app.total()):
        Xte, yte = test_shifted.Xy

        y_pred = h.predict(Xte)
        true_acc = accuracy(y_true=yte, y_pred=y_pred)

        atc_acc = atc.predict(Xte)
        doc_acc = doc.predict(Xte)
        lascal2cap_acc = lascal2cap.predict(Xte)
        hdc2cap_acc = hdc2cap.predict(Xte)
        pacc2cap_acc = pacc2cap.predict(Xte)

        atc_err.append(cap_error(acc_true=true_acc, acc_estim=atc_acc))
        doc_err.append(cap_error(acc_true=true_acc, acc_estim=doc_acc))
        lascal2cap_err.append(cap_error(acc_true=true_acc, acc_estim=lascal2cap_acc))
        hdc2cap_err.append(cap_error(acc_true=true_acc, acc_estim=hdc2cap_acc))
        pacc2cap_err.append(cap_error(acc_true=true_acc, acc_estim=pacc2cap_acc))

    print(dataset)
    print(f'ATC {np.mean(atc_err):.4f}')
    print(f'DoC {np.mean(doc_err):.4f}')
    print(f'L2C {np.mean(lascal2cap_err):.4f}')
    print(f'HDC2C {np.mean(hdc2cap_err):.4f}')
    print(f'PACC2C {np.mean(pacc2cap_err):.4f}')





