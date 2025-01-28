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

    lr = LogisticRegression()
    # Ptr = cross_val_predict(lr, Xtr, ytr, cv=5, n_jobs=-1, method='predict_proba')
    lr.fit(Xtr, ytr)

    val_posteriors = lr.predict_proba(val.X)

    atc = ATC(acc_fn=accuracy)
    atc.fit(val, val_posteriors)

    val_prot = UPP(val, sample_size=len(val), repeats=50, random_state=0, return_type='labelled_collection')
    val_prot_posteriors = [lr.predict_proba(sample.X) for sample in val_prot()]
    doc = DoC(acc_fn=accuracy, protocol=val_prot, prot_posteriors=val_prot_posteriors)
    doc.fit(val, val_posteriors)

    lascal2cap = LasCal2CAP(classifier=lr)
    lascal2cap.fit(val, val_posteriors)

    hdc2cap = HDC2CAP(classifier=lr)
    hdc2cap.fit(val, val_posteriors)

    pacc2cap = PACC2CAP(classifier=lr)
    pacc2cap.fit(val, val_posteriors)

    atc_err = []
    doc_err = []
    lascal2cap_err = []
    hdc2cap_err = []
    pacc2cap_err = []

    app = UPP(test, sample_size=len(test), repeats=10, return_type='labelled_collection')
    for test_shifted in tqdm(app(), total=app.total()):
        Xte, yte = test_shifted.Xy
        Pte = lr.predict_proba(Xte)

        y_pred = lr.predict(Xte)
        true_acc = accuracy(y_true=yte, y_pred=y_pred)

        atc_acc = atc.predict(Xte, Pte)
        doc_acc = doc.predict(Xte, Pte)
        lascal2cap_acc = lascal2cap.predict(Xte, Pte)
        hdc2cap_acc = hdc2cap.predict(Xte, Pte)
        pacc2cap_acc = pacc2cap.predict(Xte, Pte)

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





