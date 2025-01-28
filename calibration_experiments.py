from quapy.data import LabelledCollection
from quapy.method.aggregative import DistributionMatchingY
from quapy.protocol import ArtificialPrevalenceProtocol, UPP
from sklearn.calibration import CalibratedClassifierCV

from lascal import Ece, EceLabelShift
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import torch
import numpy as np
import quapy as qp
from model.classifier_calibrators import LasCalCalibration, HellingerDistanceCalibration, HeadToTailCalibrator, CpcsCalibrator
from tqdm import tqdm


def cal_error(Pte, yte):
    expected_cal_error = Ece(adaptive_bins=True, n_bins=15, p=2, classwise=False)

    logits = prob2logits(Pte)
    ece = expected_cal_error(
        logits=logits,
        labels=torch.from_numpy(yte),
    ).item()

    return ece * 100


def prob2logits(P, asnumpy=False):
    logits = torch.log(torch.from_numpy(P))
    if asnumpy:
        logits = logits.numpy()
    return logits


uncal_ECE=[]
head2tail_ECE =[]
lascal_ECE =[]
lascal_logit_ECE=[]
hdc_ECE = []
for dataset in qp.datasets.UCI_BINARY_DATASETS[:1]:

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test

    Xtr, ytr = train.Xy

    lr = LogisticRegression()
    Ptr = cross_val_predict(lr, Xtr, ytr, cv=5, n_jobs=-1, method='predict_proba')
    lr.fit(Xtr, ytr)

    cpcs = CpcsCalibrator()
    cpcs.fit(Ptr, ytr)

    dm = DistributionMatchingY(classifier=lr, nbins=10)
    preclassified = LabelledCollection(Ptr, ytr)
    dm.aggregation_fit(classif_predictions=preclassified, data=train)
    hdc = HellingerDistanceCalibration(dm)

    uncal_ece=[]
    head2tail_ece = []
    lascal_ece=[]
    lascal_logit_ece=[]
    hdc_ece = []

    app = UPP(test, sample_size=len(test), repeats=50, return_type='labelled_collection')
    for test_shifted in tqdm(app(), total=app.total()):
        Xte, yte = test_shifted.Xy
        Pte = lr.predict_proba(Xte)

        ece = cal_error(Pte, yte)
        uncal_ece.append(ece)

        head2tail = HeadToTailCalibrator(verbose=True)
        Pte_cal = head2tail.predict_proba(Ptr, ytr, Xtr, Ptr, ytr, Pte)
        # Pte_cal = cpcs.predict_proba(Pte)
        ece_cal = cal_error(Pte_cal, yte)
        head2tail_ece.append(ece_cal)

        lasCal = LasCalCalibration()
        Pte_cal = lasCal.predict_proba(Ptr, ytr, Pte)
        ece_cal = cal_error(Pte_cal, yte)
        lascal_ece.append(ece_cal)

        Pte_hdc = hdc.predict_proba(Pte)
        ece_hdc = cal_error(Pte_hdc, yte)
        hdc_ece.append(ece_hdc)

        Str = prob2logits(Ptr, asnumpy=True)
        Ste = prob2logits(Pte, asnumpy=True)
        Pte_cal = lasCal.predict_proba(Str, ytr, Ste)
        ece_cal = cal_error(Pte_cal, yte)
        lascal_logit_ece.append(ece_cal)

    uncal_ece = np.mean(uncal_ece)
    head2tail_ece = np.mean(head2tail_ece)
    lascal_ece = np.mean(lascal_ece)
    lascal_logit_ece = np.mean(lascal_logit_ece)
    hdc_ece = np.mean(hdc_ece)

    uncal_ECE.append(uncal_ece)
    head2tail_ECE.append(head2tail_ece)
    lascal_ECE.append(lascal_ece)
    lascal_logit_ECE.append(lascal_ece)
    hdc_ECE.append(hdc_ece)

    print()
    print(dataset)
    print(f'uncalibrated ECE = {uncal_ece:.4f}')
    print(f'Head2Tail ECE = {head2tail_ece:.4f}')
    print(f'LasCal ECE = {lascal_ece:.4f}')
    print(f'LasCal(logit) ECE = {lascal_logit_ece:.4f}')
    print(f'HDC ECE = {hdc_ece:.4f}')


print('*'*80)
print('End:')
print(f'uncalibrated ECE = {np.mean(uncal_ECE):.4f}')
print(f'Head2Tail ECE = {np.mean(head2tail_ECE):.4f}')
print(f'LasCal ECE = {np.mean(lascal_ECE):.4f}')
print(f'LasCal(logit) ECE = {np.mean(lascal_logit_ECE):.4f}')
print(f'HDC ECE = {np.mean(hdc_ECE):.4f}')