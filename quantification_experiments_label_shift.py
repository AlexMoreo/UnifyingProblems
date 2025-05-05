from itertools import product

from quapy.method.aggregative import AggregativeQuantifier, CC
import os
from os.path import join
import pandas as pd
from quapy.method.aggregative import AggregativeQuantifier, CC
from quapy.protocol import UniformPrevalenceProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from commons import REPEATS, SAMPLE_SIZE, EXPERIMENT_FOLDER, uci_datasets, new_artif_prev_protocol
from model.quantifiers import *

result_dir = f'results/quantification/label_shift/{EXPERIMENT_FOLDER}'
os.makedirs(result_dir, exist_ok=True)

datasets_selected = uci_datasets(top_length_k=10)


def quantifiers(classifier, Xtr, ytr):
    # quantification methods
    yield 'CC', CC(classifier)
    yield 'PCC', PCC(classifier)
    yield 'PACC', PACC(classifier)
    yield 'EMQ', EMQ(classifier)
    yield 'EMQ-BCTS', EMQ_BCTS(classifier)
    yield 'KDEy', KDEyML(classifier)

    # CAP methods
    yield 'ATC-q', ATC2Quant(classifier)
    yield 'DoC-q', DoC2Quant(classifier, protocol_constructor=new_artif_prev_protocol)
    yield 'LEAP-q', LEAP2Quant(classifier)

    # Calibration methods
    yield 'Cpcs-q-P', Cpcs2Quant(classifier, Xtr, ytr, prob2logits=False)
    yield 'TransCal-q-P', Transcal2Quant(classifier, Xtr, ytr, prob2logits=False)
    yield 'LasCal-q-P', LasCal2Quant(classifier, prob2logits=False)
    yield 'Head2Tail-q-P', HeadToTail2Quant(classifier, Xtr, ytr, prob2logits=False, n_components=50)

    # yield 'PACC(LasCal)', PACC_LasCal(classifier, prob2logits=True)
    yield 'EMQ(LasCal)2-S', EMQ_LasCal(classifier, prob2logits=True)


def fit_quantifier(quant, train, val):
    if isinstance(quant, AggregativeQuantifier):
        quant.fit(train, fit_classifier=False, val_split=val)
    elif isinstance(quant, BaseQuantifier):
        quant.fit(val)
    else:
        raise ValueError(f'{quant}: unrecognized object')


def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    yield 'mlp', MLPClassifier()


print('Datasets:', datasets_selected)
print('Repeats:', REPEATS)

all_results = []
methods_order = []

n_classifiers = len([c for _,c in classifiers()])
pbar = tqdm(product(datasets_selected, classifiers()), total=len(datasets_selected)*n_classifiers)
for dataset, (cls_name, h) in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test
    train_prev = train.prevalence()

    train, val = train.split_stratified(0.5, random_state=0)
    app = UniformPrevalenceProtocol(test, sample_size=SAMPLE_SIZE, repeats=REPEATS, random_state=0)
    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE

    # h = LogisticRegression()
    h.fit(*train.Xy)

    for name, quant in quantifiers(h, *train.Xy):
        if name not in methods_order:
            methods_order.append(name)

        result_method_dataset_path = join(result_dir, f'{name}_{dataset}_{cls_name}.csv')
        if os.path.exists(result_method_dataset_path):
            report = pd.read_csv(result_method_dataset_path)
        else:
            fit_quantifier(quant, train, val)
            report = qp.evaluation.evaluation_report(quant, protocol=app, error_metrics=['ae', 'rae'])
            true_prevs = np.vstack(report['true-prev'])
            report['id'] = np.arange(REPEATS)
            report['shift'] = qp.error.ae(true_prevs, np.tile(train_prev, (REPEATS, 1)))
            report['method'] = name
            report['dataset'] = dataset
            report['classifier'] = cls_name
            report.to_csv(result_method_dataset_path, index=False)

        all_results.append(report)

df = pd.concat(all_results)
df.pop('true-prev')
df.pop('estim-prev')
pivot = df.pivot_table(index='dataset', columns='method', values='ae')
print(df)
print(pivot)
print(pivot.mean(axis=0))


from new_table import LatexTable

table = LatexTable.from_dataframe(df, method='method', benchmark='dataset', value='ae')
table.name = 'quantification_pps'
table.reorder_methods(methods_order)
table.format.configuration.show_std=False
table.latexPDF('./tables/quantification_label_shift.pdf')

