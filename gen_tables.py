import numpy as np

import commons
from glob import glob
from os.path import join
import pandas as pd
import sys
from pathlib import Path
import util
from result_table.src.format import Format
from result_table.src.new_table import LatexTable, Configuration
from tools import tabular2pdf
from os.path import join

task = 'calibration'
dataset_shift='covariate_shift'
# dataset_shift = 'label_shift'
folder = commons.EXPERIMENT_FOLDER
results_path = join('./results', task, dataset_shift, folder)

baselines = ['Uncal', 'Platt', 'Isotonic']
reference = ['TransCal-S', 'CPCS-S', 'LasCal-S']
contenders = ['EM', 'HDcal8-sm-mono', 'PACC-cal(log)', 'PACC-cal(iso)', 'Bin6-EM5', 'Bin6-KDEy5', 'Bin2-ATC6', 'Bin2-DoC6', 'Bin2-LEAP6']
#['HDcal8-sm-mono', 'PACC-cal(log)', 'PACC-cal(iso)', 'Bin2-DoC6']

classifiers = {
    'covariate_shift': ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'],
    'label_shift': ['lr', 'nb', 'mlp']
}[dataset_shift]

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
n_reference = len(reference)
n_rows = 2 + n_setups*n_classifiers
n_cols = 2 + n_methods

config = Configuration()
config.style = 'rules'
myformat = Format(config)
style = myformat.get_latex_style(n_cols)

def escape_latex_underscore(s):
    import re
    s = re.sub(r'(?<!\\)_', r'\_', s)
    s = s.replace('->', '$\\rightarrow$')
    return s

def prepare_strings(table_arr):
    for j in range(1, table_arr.shape[1]):
        table_arr[0, j] = escape_latex_underscore(table_arr[0, j])
    for i in range(1, table_arr.shape[0]):
        table_arr[i, 0] = escape_latex_underscore(table_arr[i, 0])

replace_classifier={
    'bert-base-uncased': 'BERT',
    'distilbert-base-uncased': 'DistilBERT',
    'roberta-base': 'RoBERTa'
}
replace_method={
    'HDcal8-sm-mono': 'DM-cal',
}

lines = []
lines.append('\\begin{tabular}{' + style['column_format'] + '} '+style['TOPLINE'])
for i, classifier in enumerate(classifiers):
    table_arr = tables[classifier].as_str_array()
    prepare_strings(table_arr)
    if i==0:
        for j in range(1, table_arr.shape[1]):
            table_arr[0, j] = replace_method.get(table_arr[0, j],table_arr[0, j])
        if config.side_columns:
            for j in range(1, table_arr.shape[1]):
                table_arr[0, j] = '\\begin{sideways}' + table_arr[0, j] + '\;\end{sideways}'
        header = '\multicolumn{2}{c|}{} & ' + ' & '.join(table_arr[0,1:]) + style['ENDL'] + style['MIDLINE']
        lines.append(header)
    for j, row in enumerate(table_arr[1:]):
        endl = style['HLINE']
        if j == table_arr.shape[0] - 2:
            if i < (len(classifiers)-1):
                endl = style['MIDLINE']
            else:
                endl = style['BOTTOMLINE']

        first_col = ' & ' if j!=0 else (
            '\multirow{'+
                str(table_arr.shape[0]-1)+
            '}{*}{\\begin{sideways}' +
                replace_classifier.get(classifier, classifier) +
            '\;\end{sideways}} & '
        )
        line = first_col + ' & '.join(row) + style['ENDL'] + endl
        lines.append(line)

    if i==n_classifiers-1:
        for j in range(1, n_reference+1):
            first_col = ' & ' if j != 1 else ('\multirow{' + str(n_reference) + '}{*}{\\begin{sideways}Wins\;\end{sideways}} & ')
            second_col = f' $\Pr(M\succ {j}R)$ &'
            parts = []
            for k, method in enumerate(all_methods):
                if method in counts:
                    count = counts[method][j] * 100
                    stat  = reject_H0[method][j]
                    dag = '\dag ' if stat else ''
                    parts.append(f'${dag}{count:.2f}\%$')
                else:
                    parts.append('---')
            endl = style['HLINE'] if j < n_reference else style['BOTTOMLINE']
            line = first_col + second_col + ' & '.join(parts) + style['ENDL'] + endl
            lines.append(line)


lines.append('\\end{tabular}')

tables_path = './fulltables'
filename = f'{task}_{dataset_shift}'
tabular_path = join(tables_path, 'tabular', f'{filename}.tex')
util.save_text(tabular_path, '\n'.join(lines))
tabular2pdf(tabular_path, join(tables_path, f'{filename}.pdf'), landscape=True, resizebox=True)
