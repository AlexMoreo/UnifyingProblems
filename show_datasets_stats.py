from commons import uci_datasets
import quapy as qp
import numpy as np

sizes = []
feats = []
prevs = []
datasets = uci_datasets()
for dataset in datasets:
    data = qp.datasets.fetch_UCIBinaryLabelledCollection(dataset)
    sizes.append(len(data))
    feats.append(data.X.shape[1])
    prevs.append(data.prevalence()[1])
prevs=np.asarray(prevs)
balance=np.abs(0.5-prevs)

print(f'size min={min(sizes)} in {datasets[np.argmin(sizes)]} ; max={max(sizes)} in {datasets[np.argmax(sizes)]}')
print(f'feats min={min(feats)} {datasets[np.argmin(feats)]} ; max={max(feats)} in {datasets[np.argmax(feats)]}')
print(f'prev min={min(prevs)} {datasets[np.argmin(prevs)]} ; max={max(prevs)} in {datasets[np.argmax(prevs)]} ; '
      f'balanced min={min(balance)} with prev {prevs[np.argmin(balance)]} for {datasets[np.argmin(balance)]}')

for dataset in uci_datasets():
    print(f'\\texttt{{{dataset}}}, ')