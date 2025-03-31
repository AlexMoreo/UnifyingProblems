from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC, EMQ, AggregativeQuantifier, CC, PCC, KDEyML, ACC
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
    app = NaturalPrevalenceProtocol(
        lc,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return app





def quantifiers(classifier, setup: Setup, x_val_idx:np.ndarray):
    # quantification methods
    #yield 'Naive', MaximumLikelihoodPrevalenceEstimation()
    yield 'CC', CC(classifier)
    yield 'PCC', PCC(classifier)
    yield 'PACC', PACC(classifier)
    yield 'EMQ', EMQ(classifier)
    yield 'EMQ-BCTS', EMQ_BCTS(classifier)
    yield 'KDEy', KDEyML(classifier)

    # CAP methods
    yield 'ATC-q', ATC2Quant(classifier)
    yield 'DoC-q', DoC2Quant(classifier, protocol_constructor=new_natural_protocol)
    yield 'LEAP-q', LEAP2Quant(classifier)
    yield 'LEAP-PCC-q', LEAP2Quant(classifier, quantifier_cls=PCC)

    # Calibration methods
    Ftr = setup.train.hidden
    ytr = setup.train.labels
    yield 'LasCal-q-P', LasCal2Quant(classifier, prob2logits=False)
    yield 'TransCal-q-P', Transcal2Quant(classifier, Ftr=Ftr, ytr=ytr, prob2logits=False)
    yield 'Cpcs-q-P', Cpcs2Quant(classifier, Ftr=Ftr, ytr=ytr, prob2logits=False)
    # yield 'Head2Tail-q-S', HeadToTail2Quant(classifier, Ftr, ytr, prob2logits=True)
    yield 'Head2Tail-q-P', HeadToTail2Quant(classifier, Ftr, ytr, prob2logits=False, n_components=50)
    # yield 'Head2Tail-q-S', HeadToTail2Quant(classifier, Ftr, ytr, prob2logits=True, n_components=50)
    ## yield 'PACC(LasCal)', PACCLasCal(classifier)
    ## yield 'EMQ(LasCal)', EMQLasCal(classifier)


def fit_quantifier(quant, x_val_idx, val_labels, val_hidden):
    valid_lc = LabelledCollection(x_val_idx, val_labels, classes=[0,1])
    if isinstance(quant, AggregativeQuantifier):
        quant.fit(valid_lc, fit_classifier=False)
    elif isinstance(quant, Method2Quant):
        quant.fit(valid_lc, hidden=val_hidden)
    elif isinstance(quant, BaseQuantifier):
        quant.fit(valid_lc)
    else:
        raise ValueError(f'{quant}: unrecognized object')


def quantify(quant, x_test_idx, test_hidden):
    if isinstance(quant, Method2Quant):
        prev_estim = quant.quantify(x_test_idx, hidden=test_hidden)
    else:
        prev_estim = quant.quantify(x_test_idx)
    return prev_estim


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

                fit_quantifier(quant, x_val_idx, setup.valid.labels, val_hidden=setup.valid.hidden)
                
                prev_estim = quantify(quant, x_test_idx, test_hidden=test_sample.hidden)
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

            report=pd.DataFrame(method_setup_results)
            report.to_csv(result_method_setup_path, index=False)
        all_results.append(report)

df = pd.concat(all_results)
pivot = df.pivot_table(index='dataset', columns='method', values='ae')
print(df)
print(pivot)
print(pivot.mean(axis=0))


from new_table import LatexTable

tables = []
for classifier_name in models:
    df_h = df[df['classifier']==classifier_name]
    table = LatexTable.from_dataframe(df_h, method='method', benchmark='dataset', value='ae')
    table.name = f'quantification_cv_AE_{classifier_name}'
    table.reorder_methods(methods_order)
    table.format.configuration.show_std=False
    table.format.configuration.side_columns = True
    tables.append(table)
LatexTable.LatexPDF('./tables/quantification_covariate_shift.pdf', tables=tables)