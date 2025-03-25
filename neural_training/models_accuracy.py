from commons import *
import numpy as np
import pandas as pd
from util import cal_error

def get_accuracy(dataset: Dataset):
    predictions = dataset.logits.argmax(axis=1)
    return np.mean(predictions==dataset.labels)

accs = []

for setup in iterate_datasets_covariate_shift(neural_models_path='./embeds'):
    print(f'{setup.model}: {setup.source} {setup.target}')
    # print(f'\tvalid-acc = {get_accuracy(setup.valid)*100:.2f}%')
    print(f'\tsource acc = {get_accuracy(setup.in_test) * 100:.2f}%')
    print(f'\ttarget acc = {get_accuracy(setup.out_test) * 100:.2f}%')
    accs.append({
        'model': setup.model,
        'source': setup.source,
        'target': setup.source,
        'accuracy': get_accuracy(setup.in_test),
        'ECE': cal_error(setup.in_test.logits, setup.in_test.labels, arelogits=True)
    })
    accs.append({
        'model': setup.model,
        'source': setup.source,
        'target': setup.target,
        'accuracy': get_accuracy(setup.out_test),
        'ECE': cal_error(setup.out_test.logits, setup.out_test.labels, arelogits=True)
    })

df = pd.DataFrame(accs)
print(df.pivot_table(index=['model', 'source'], columns='target', values='accuracy'))
print(df.pivot_table(index=['model', 'source'], columns='target', values='ECE'))