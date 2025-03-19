import numpy as np

import commons
from glob import glob
from os.path import join
import pandas as pd
import sys
from pathlib import Path
from new_table import LatexTable

import util

task = 'calibration'
dataset_shift='covariate_shift'
folder = commons.EXPERIMENT_FOLDER
results_path = join('./results', task, dataset_shift, folder)

baselines = ['Uncal', 'Platt', 'Isotonic']
reference = ['TransCal', 'CPCS', 'LasCal']
contenders = ['HDcal8-sm-mono', 'PACC-cal(log)', 'PACC-cal(iso)', 'Bin6-PCC5', 'Bin2-DoC6']

classifiers = ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']
datasets = ['imdb', 'rt', 'yelp']
setups = []
for source in datasets:
    for target in datasets:
        if source==target: continue
        setups.append(f'{source}->{target}')

all_methods = baselines + reference + contenders

all_dfs = []
for result_file in glob(join(results_path, '*.csv')):
    methodname, classifier, *_ = Path(result_file).name.split('_')
    if methodname in all_methods:
        df = pd.read_csv(result_file, index_col=None)
        assert len(df) == 100, 'unexpected number of rows'
        all_dfs.append(df)

df = pd.concat(all_dfs)
print(f'read {len(df)} rows')

counts, reject_H0 = util.count_successes(df, baselines=baselines, value='ece', expected_repetitions=100)
for method in contenders:
    print(method)
    for i in range(1,len(reference)+1):
        print(f'\t>{i}: {counts[method][i]*100:.2f}% : significance {reject_H0[method][i]}')



error = 'ece'

tables = {}
for i, classifier in enumerate(classifiers):
    df_block = df[df['classifier']==classifier]
    table = LatexTable.from_dataframe(df_block, method='method', benchmark='dataset', value=error)
    table.reorder_methods(all_methods)
    table.reorder_benchmarks(setups)
    tables[classifier] = table


n_setups = len(setups)
n_classifiers = len(classifiers)
n_methods = len(all_methods)
n_rows = 2 + n_setups*n_classifiers
n_cols = 2 + n_methods
str_table = np.full(shape=(n_rows, n_cols), dtype=object, fill_value='')
str_table[1,2:] = tables[classifiers[0]].methods

for i, classifier in enumerate(classifiers):
    table_arr = tables[classifier].as_str_array()
    str_table[2+i*n_setups:2+(i+1)*n_setups, 1:] = table_arr[1:, :]

print(str_table)
