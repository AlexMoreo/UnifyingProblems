from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC, EMQ, AggregativeQuantifier, CC, PCC, KDEyML
from quapy.method.base import BaseQuantifier
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation
from quapy.protocol import NaturalPrevalenceProtocol
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
from os.path import join
import pathlib
from util import PrecomputedClassifier
import quapy as qp
from tqdm import tqdm
import numpy as np
from model.quantifiers import *
from dataclasses import dataclass, asdict

from commons import *


result_dir = f'results/quantification/covariate_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)
qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

@dataclass
class ResultRow:
    dataset: str  # '{source}->{target}'
    source: str
    target: str
    id: int
    shift: float
    method: str
    classifier: str
    ae: float
    rae: float

def new_natural_protocol(X, y, classes):
    lc = LabelledCollection(X, y, classes=classes)
    app = NaturallPrevalenceProtocol(
        lc,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return app


def quantifiers(classifier, setup: Setup, x_val_idx:np.ndarray):
    # quantification methods
    yield 'Naive', MaximumLikelihoodPrevalenceEstimation()
    yield 'CC', CC(classifier)
    yield 'PCC', PCC(classifier)
    yield 'PACC', PACC(classifier)
    yield 'EMQ', EMQ(classifier)
    yield 'KDEy', KDEyML(classifier)

    # CAP methods
    #yield 'ATC-q', ATC2Quant(classifier)
    #yield 'DoC-q', DoC2Quant(classifier, protocol_constructor=new_natural_protocol)
    #yield 'LEAP-q', LEAP2Quant(classifier)

    # Calibration methods
    #yield 'LasCal-q', LasCal2Quant(classifier, prob2logits=True)
    ## yield 'LasCal-q-P', LasCal2Quant(classifier, prob2logits=False)
    #yield 'TransCal-q', Transcal2Quant(classifier, Xtr, ytr, prob2logits=True)
    ## yield 'TransCal-q-P', Transcal2Quant(classifier, Xtr, ytr, prob2logits=False)
    ## yield 'Head2Tail-q', Head2Tail2Quant(classifier, Xtr, ytr, prob2logits=True)
    #yield 'Head2Tail-q-P', Head2Tail2Quant(classifier, Xtr, ytr, prob2logits=False)
    ## yield 'PACC(LasCal)', PACCLasCal(classifier)
    ## yield 'EMQ(LasCal)', EMQLasCal(classifier)


def fit_quantifier(quant, x_val_idx, val_labels):
    valid_lc = LabelledCollection(x_val_idx, val_labels, classes=[0,1])
    if isinstance(quant, AggregativeQuantifier):
        quant.fit(valid_lc, fit_classifier=False)
    elif isinstance(quant, BaseQuantifier):
        quant.fit(valid_lc)
    else:
        raise ValueError(f'{quant}: unrecognized object')


all_results = []
methods_order = []

pbar = tqdm(iterate_datasets_covariate_shift(), total=total_setups_covariate_shift)
for setup in pbar:
    description = f'[{setup.model}]::{setup.source}->{setup.target}'
    
    h = PrecomputedClassifier()
    x_val_idx = h.feed(X=setup.valid.hidden, P=setup.valid.posteriors, L=setup.valid.logits)
    
    for name, quant in quantifiers(h, setup, x_val_idx):
        if name not in methods_order:
            methods_order.append(name)

        result_method_setup_path = join(result_dir, f'{name}_{setup.model}_{setup.source}__{setup.target}.csv')
        if os.path.exists(result_method_setup_path):
            report = pd.read_csv(result_method_setup_path)
        else:
            method_setup_results = []
            err_ave = []
            for idx, (test_sample, shift) in  enumerate(yield_random_samples(setup.in_test, setup.out_test, repeats=REPEATS, samplesize=SAMPLE_SIZE)):
                x_test_idx = h.feed(X=test_sample.hidden, P=test_sample.posteriors, L=test_sample.logits)

                fit_quantifier(quant, x_val_idx, setup.valid.labels)
                
                prev_estim = quant.quantify(x_test_idx)
                prev_true = test_sample.prevalence
                ae = qp.error.ae(prev_true, prev_estim)
                rae = qp.error.rae(prev_true, prev_estim)

                err_ave.append(ae)
                pbar.set_description(description + f' {name} ({idx}/{REPEATS}) AE-ave={np.mean(err_ave):.5f}')

                result = ResultRow(
                    dataset=f'{setup.source}->{setup.target}',
                    source=setup.source,
                    target=setup.target,
                    id=idx,
                    shift=shift,
                    method=name,
                    classifier=setup.model,
                    ae=ae,
                    rae=rae)
                method_setup_results.append(asdict(result))

        all_results.append(result)

df = pd.concat(all_results)
pivot = df.pivot_table(index='dataset', columns='method', values='ae')
print(df)
print(pivot)
print(pivot.mean(axis=0))


from new_table import LatexTable

table = LatexTable.from_dataframe(df, method='method', benchmark='dataset', value='ae')
table.name = 'quantification_covariate_shift'
table.reorder_methods(methods_order)
table.format.configuration.show_std=False
table.latexPDF('./tables/quantification_covariate_shift.pdf')