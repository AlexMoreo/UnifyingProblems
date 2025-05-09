from pycalib.visualisations import plot_binary_reliability_diagram_gaps, plot_calibration_map
from sklearn.linear_model import LogisticRegression

from calibration_experiments_label_shift import calibrate
from commons import uci_datasets
from model.classifier_calibrators import *
import quapy as qp
import itertools
from util import makepath


def dataset_iter():
    for dataset in uci_datasets():
        train, test = qp.datasets.fetch_UCIBinaryDataset(dataset).train_test
        train, val = train.split_stratified(0.5, random_state=0)
        test = test.sampling(250, 0.5, 0.5)
        yield dataset, (train, val, test)

def classifier_iter():
    yield 'lr', LogisticRegression()
    # yield 'nb', GaussianNB()
    # yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    # yield 'mlp', MLPClassifier()

def calibrator_iter(classifier):
    yield 'Platt', PlattScaling().fit(Pva, yva)
    yield 'LasCal-P', LasCalCalibration(prob2logits=False)
    yield 'EM-BCTS', EMQ_BCTS_Calibrator().fit(Pva, yva)
    yield f'DMCal', DistributionMatchingCalibration(classifier, nbins=8).fit(Pva, yva)

for dataset_arg, classif_arg in itertools.product(dataset_iter(), classifier_iter()):
    dataset, (train, val, test) = dataset_arg
    cls_name, h = classif_arg

    Xtr, ytr = train.Xy
    h.fit(Xtr, ytr)

    Xva, yva = val.Xy
    Pva = h.predict_proba(Xva)

    Xte, yte = test.Xy
    Pte = h.predict_proba(Xte)

    for cal_name, calibrator in calibrator_iter(h):
        Pte_cal = calibrate(calibrator, Xtr, ytr, Xva, Pva, yva, Xte, Pte)

        fig, ax = plot_binary_reliability_diagram_gaps(
            y_true=yte,
            p_pred=Pte_cal,
            show_histogram=True,
        )
        outpath = f'figures/calibration/label_shift/{cls_name}_{dataset}_{cal_name}.pdf'
        makepath(outpath)
        fig.set_size_inches(10, 10)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")

