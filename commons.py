from dataclasses import dataclass
import numpy as np
import quapy as qp
import torch
from os.path import join

from quapy.data import LabelledCollection
from quapy.protocol import NaturalPrevalenceProtocol, UniformPrevalenceProtocol
from scipy.special import softmax
import quapy.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


REPEATS = 100
SAMPLE_SIZE = 250

EXPERIMENT_FOLDER = f'repeats_{REPEATS}_samplesize_{SAMPLE_SIZE}'
NEURAL_PRETRAINED = './neural_training/embeds/'

sentiment_datasets = ['imdb', 'rt', 'yelp']
models = ['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base']

total_setups_covariate_shift = len(models)*(len(sentiment_datasets)*(len(sentiment_datasets)-1))

@dataclass
class Dataset:
    hidden: np.ndarray
    logits: np.ndarray
    posteriors: np.ndarray
    labels: np.ndarray
    prevalence: np.ndarray

@dataclass
class Setup:
    model: str
    source: str
    target: str
    train: Dataset
    valid: Dataset
    in_test: Dataset
    out_test: Dataset


def iterate_datasets_covariate_shift(neural_models_path=None):
    if neural_models_path is None:
        neural_models_path = NEURAL_PRETRAINED

    def load_dataset(path, domain, splitname, reduce=None, random_seed=0):
        hidden = torch.load(join(path, f'{domain}.{splitname}.hidden_states.pt')).numpy()
        logits = torch.load(join(path, f'{domain}.{splitname}.logits.pt')).numpy()
        labels = torch.load(join(path, f'{domain}.{splitname}.labels.pt')).numpy()
        if reduce is not None and isinstance(reduce,int) and reduce<len(labels):
            np.random.seed(random_seed)
            sel_idx = np.random.choice(reduce, size=reduce, replace=False)
            hidden = hidden[sel_idx]
            logits = logits[sel_idx]
            labels = labels[sel_idx]
        posteriors = softmax(logits, axis=1)
        prevalence = F.prevalence_from_labels(labels, classes=[0,1])
        return Dataset(hidden=hidden, logits=logits, posteriors=posteriors, prevalence=prevalence, labels=labels)

    for source in sentiment_datasets:
        for model in models:
            path = f'{neural_models_path}/{source}/{model}'

            train = load_dataset(path, 'source', 'train', reduce=5000)
            valid = load_dataset(path, 'source', 'validation')
            in_test  = load_dataset(path, 'source', 'test')

            for target in sentiment_datasets:
                if target==source: continue
                out_test = load_dataset(path, f'target_{target}', 'test')
                yield Setup(model=model, source=source, target=target, train=train, valid=valid, in_test=in_test, out_test=out_test)


def yield_random_samples(in_test: Dataset, out_test: Dataset, repeats, samplesize):
    np.random.seed(0)
    indexes = []
    in_test_length = len(in_test.labels)
    out_test_length = len(out_test.labels)
    from_out_test = np.round(np.linspace(0, samplesize, repeats)).astype(int)
    for out_size in from_out_test:
        from_in = np.random.choice(in_test_length, size=samplesize-out_size, replace=True)
        from_out = np.random.choice(out_test_length, size=out_size, replace=True)
        indexes.append((from_in,from_out,))
    for index_in, index_out in indexes:
        sample_hidden = np.vstack([in_test.hidden[index_in], out_test.hidden[index_out]])
        sample_logits = np.vstack([in_test.logits[index_in], out_test.logits[index_out]])
        sample_labels = np.concatenate([in_test.labels[index_in], out_test.labels[index_out]])
        sample_posteriors = np.vstack([in_test.posteriors[index_in], out_test.posteriors[index_out]])
        sample_prevalence = F.prevalence_from_labels(sample_labels, classes=[0,1])
        shift = len(index_out)/samplesize
        yield Dataset(hidden=sample_hidden, logits=sample_logits, labels=sample_labels, posteriors=sample_posteriors, prevalence=sample_prevalence), shift


def uci_datasets(top_length_k=10):
    datasets_selected, _ = list(zip(*sorted([
        (dataset, len(qp.datasets.fetch_UCIBinaryLabelledCollection(dataset)))
        for dataset in qp.datasets.UCI_BINARY_DATASETS
    ], key=lambda x:x[1])[-top_length_k:]))
    return list(datasets_selected)


def classifiers():
    yield 'lr', LogisticRegression()
    yield 'nb', GaussianNB()
    yield 'knn', KNeighborsClassifier(n_neighbors=10, weights='uniform')
    yield 'mlp', MLPClassifier()


def new_artif_prev_protocol(X, y, classes=[0, 1]):
    lc = LabelledCollection(X, y, classes=classes)
    protocol = UniformPrevalenceProtocol(
        lc,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return protocol

def new_natur_prev_protocol(X, y, classes=[0, 1]):
    lc = LabelledCollection(X, y, classes=classes)
    protocol = NaturalPrevalenceProtocol(
        lc,
        sample_size=SAMPLE_SIZE,
        repeats=REPEATS,
        return_type='labelled_collection',
        random_state=0
    )
    return protocol
