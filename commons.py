from dataclasses import dataclass
import numpy as np
import torch
from os.path import join
from scipy.special import softmax
import quapy.functional as F


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


def iterate_datasets__depr():

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
            path = f'./neural_training/embeds/{source}/{model}'

            train = load_dataset(path, 'source', 'train', reduce=5000)
            valid = load_dataset(path, 'source', 'validation')

            for target in sentiment_datasets:
                if target == source:
                    target_prefix = 'source'
                else:
                    target_prefix = f'target_{target}'

                test = load_dataset(path, target_prefix, 'test')

                yield Setup(model=model, source=source, target=target, train=train, valid=valid, test=test)


def iterate_datasets_covariate_shift():

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
            path = f'{NEURAL_PRETRAINED}/{source}/{model}'

            train = load_dataset(path, 'source', 'train', reduce=5000)
            valid = load_dataset(path, 'source', 'validation')
            in_test  = load_dataset(path, 'source', 'test')

            for target in sentiment_datasets:
                if target==source: continue
                out_test = load_dataset(path, f'target_{target}', 'test')
                yield Setup(model=model, source=source, target=target, train=train, valid=valid, in_test=in_test, out_test=out_test)


def yield_random_samples__depr(test: Dataset, repeats, samplesize):
    np.random.seed(0)
    indexes = []
    test_length = len(test.labels)
    for _ in range(repeats):
        indexes.append(np.random.choice(test_length, size=samplesize, replace=True))
    for index in indexes:
        sample_hidden = test.hidden[index]
        sample_logits = test.logits[index]
        sample_labels = test.labels[index]
        sample_posteriors = test.posteriors[index]
        sample_prevalence = F.prevalence_from_labels(sample_labels, classes=[0,1])
        yield Dataset(hidden=sample_hidden, logits=sample_logits, labels=sample_labels, posteriors=sample_posteriors, prevalence=sample_prevalence)


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
