import numpy as np
from itertools import product
import quapy as qp
import pandas as pd

from commons import uci_datasets, classifiers
from tqdm import tqdm

results = []

datasets_selected = uci_datasets(top_length_k=10)

replace_classifier = {
    'lr': 'Logistic Regression',
    'nb': 'Naïve Bayes',
    'knn': 'k Nearest Neighbor',
    'mlp': 'Multi-layer Perceptron'
}

n_classifiers = len([c for _,c in classifiers()])
pbar = tqdm(product(datasets_selected, classifiers()), total=len(datasets_selected)*n_classifiers)
for dataset, (cls_name, h) in pbar:
    pbar.set_description(f'running: {dataset}')

    data = qp.datasets.fetch_UCIBinaryDataset(dataset)
    train, test = data.train_test

    train, val = train.split_stratified(0.5, random_state=0)

    h.fit(*train.Xy)
    pred_labels = h.predict(test.X)
    true_labels = test.labels
    accuracy = np.mean(pred_labels==true_labels)

    results.append({
        'classifier': replace_classifier[cls_name],
        'dataset': dataset,
        'acc': accuracy
    })

df = pd.DataFrame(results)

pv = df.pivot_table(index='dataset', columns='classifier', values='acc')
print(pv)

path_out = './tables/tables/sklearn_acc.tex'
latex = pv.to_latex(float_format="%.3f")

with open(path_out, 'wt') as foo:
    foo.write(latex)