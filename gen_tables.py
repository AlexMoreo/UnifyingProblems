import itertools
import os

import numpy as np
from critdd import Diagram

import commons
from glob import glob
from os.path import join
import pandas as pd
import sys
from pathlib import Path
import util
from result_table.src.format import Format
from result_table.src.new_table import LatexTable, Configuration
from result_table.src.tools import *
from tools import tabular2pdf
from os.path import join


tasks = ['classifier_accuracy_prediction', 'calibration', 'quantification']
dataset_shifts=['label_shift', 'covariate_shift']
os.makedirs('cddiagrams', exist_ok=True)


def tikz2pdf(tikz_path, pdf_path, landscape=False):
    parent = Path(pdf_path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)

    doc_path = pdf_path.replace('.pdf', '.tex')

    tickz_path_rel = os.path.relpath(tikz_path, parent)
    tickz_str = '\input{'+tickz_path_rel+'} \n'
    tex_document(doc_path, [tickz_str], landscape=landscape, add_package=['tikz', 'pgfplots'])
    latex2pdf(pdf_path)


for task, dataset_shift in itertools.product(tasks, dataset_shifts):

    error = {
        'calibration': 'ece',
        'classifier_accuracy_prediction': 'err',
        'quantification': 'ae'
    }[task]

    # dataset_shift = 'label_shift'
    folder = commons.EXPERIMENT_FOLDER
    results_path = join('./results', task, dataset_shift, folder)

    baselines = {
        'calibration': ['Platt'],#['Uncal', 'Platt', 'Isotonic'],
        'classifier_accuracy_prediction': ['Naive'],
        'quantification': ['CC', 'PCC']
    }[task]

    reference = {
        'calibration': ['TransCal', 'CPCS', 'LasCal'],
        'classifier_accuracy_prediction': ['ATC', 'DoC', 'LEAP'],
        'quantification': ['PACC', 'EMQ', 'KDEy']
    }[task]

    # patch...
    if task=='calibration' and dataset_shift=='label_shift':
        reference = [f'{M}-S' for M in reference]

    contenders = {
        'calibration': ['EM', 'HDcal8-sm-mono', 'PACC-cal(clip)', 'Bin6-PACC5', 'Bin6-EM5', 'Bin6-KDEy5', 'Bin2-ATC6', 'Bin2-DoC6', 'Bin2-LEAP6'],
        'classifier_accuracy_prediction': ['TransCal-a-S', 'Cpcs-a-S', 'LasCal-a-P', 'PACC-a', 'KDEy-a', 'EMQ-a'] + (['PCC-a', 'EMQ-BCTS-a'] if dataset_shift=='covariate_shift' else []),
        'quantification': ['ATC-q', 'DoC-q', 'LEAP-q', 'LasCal-q-P', 'TransCal-q-P', 'Cpcs-q-P']
    }[task]

    classifiers = {
        'covariate_shift': ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'],
        'label_shift': ['lr', 'nb', 'mlp'] if task!='quantification' else ['']
    }[dataset_shift]

    if dataset_shift=='covariate_shift':
        domains = ['imdb', 'rt', 'yelp']
        datasets = []
        for source in domains:
            for target in domains:
                if source==target: continue
                datasets.append(f'{source}->{target}')
    else:
        datasets = commons.uci_datasets()

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

    counts, reject_H0 = util.count_successes(df, baselines=reference, value=error, expected_repetitions=100)
    ranks, cd_df = util.get_ranks(df, value=error, expected_repetitions=100)

    # get_critical_difference_diagram(cd_df, )
    cd_df = cd_df.pivot_table(
        index="dataset",
        columns="method",
        values="score"
    )

    diagram = Diagram(
        cd_df.to_numpy(),
        treatment_names=cd_df.columns,
        maximize_outcome=True
    )
    # inspect average ranks and groups of statistically indistinguishable treatments
    # diagram.average_ranks  # the average rank of each treatment
    # diagram.get_groups(alpha=.01, adjustment="holm")

    # export the diagram to a file
    diagram.to_file(
        f'cddiagrams/diagrams/{task}-{dataset_shift}.tex',
        alpha=.05,
        adjustment="holm",
        reverse_x=False,
        axis_options={"title": f"{task} {dataset_shift}".replace('_','\_')},
    )
    tikz2pdf(f'cddiagrams/diagrams/{task}-{dataset_shift}.tex', f'cddiagrams/{task}-{dataset_shift}_doc.pdf')




    for method in contenders:
        print(method)
        for i in range(1,len(reference)+1):
            print(f'\t>{i}: {counts[method][i]*100:.2f}% : significance {reject_H0[method][i]}')

    tables = {}
    for i, classifier in enumerate(classifiers):
        if classifier=='':
            df_block = df
        else:
            df_block = df[df['classifier']==classifier]
        table = LatexTable.from_dataframe(df_block, method='method', benchmark='dataset', value=error)
        table.reorder_methods(all_methods)
        table.reorder_benchmarks(datasets)
        table.format.configuration.show_std=False
        tables[classifier] = table


    n_setups = len(datasets)
    n_classifiers = len(classifiers)
    n_methods = len(all_methods)
    n_baselines = len(baselines)
    n_reference = len(reference)
    n_contenders = len(contenders)
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
        'roberta-base': 'RoBERTa',
        'lr': 'Logistic Regression',
        'nb': 'Naïve Bayes',
        'mlp': 'Multi-layer Perceptron'
    }
    replace_method={
        'HDcal8-sm-mono': 'DM-cal',
    }

    # column_format = style['column_format']
    column_format = 'cc|'+'c'*n_baselines+'|'+'c'*n_reference+'|'+'c'*n_contenders
    lines = []
    lines.append('\\begin{tabular}{' + column_format + '} '+style['TOPLINE'])
    lines.append('\multicolumn{2}{c}{} & \multicolumn{'+str(n_baselines)+'}{c|}{Baselines} & \multicolumn{'+str(n_reference)+'}{c|}{Reference} & \multicolumn{'+str(n_contenders)+'}{c}{Contenders}' + style['ENDL'])
    for i, classifier in enumerate(classifiers):
        table_arr = tables[classifier].as_str_array()
        prepare_strings(table_arr)
        if i==0:
            for j in range(1, table_arr.shape[1]):
                table_arr[0, j] = replace_method.get(table_arr[0, j],table_arr[0, j])
            if config.side_columns:
                for j in range(1, table_arr.shape[1]):
                    table_arr[0, j] = '\\begin{sideways}' + table_arr[0, j] + '\;\end{sideways}'
            header = '\multicolumn{2}{c}{} & ' + ' & '.join(table_arr[0,1:]) + style['ENDL'] + style['MIDLINE']
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
            # add the method vs. reference comparisons
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

            # add the average ranks
            method_ranks = np.asarray([ranks[method] for method in all_methods])
            order = np.argsort(method_ranks)
            top_3_ranks = set(order[:3])
            parts = [f'{m_rank:.2f}' for m_rank in method_ranks]
            parts = [f'\\textbf{{{m_rank}}}' if pos in top_3_ranks else m_rank for pos, m_rank in enumerate(parts)]
            line = '  & Ave Rank &' + ' & '.join(parts) + style['ENDL'] + style['BOTTOMLINE']
            lines.append(line)


    lines.append('\\end{tabular}')

    tables_path = './fulltables'
    filename = f'{task}_{dataset_shift}'
    tabular_path = join(tables_path, 'tabular', f'{filename}.tex')
    util.save_text(tabular_path, '\n'.join(lines))
    tabular2pdf(tabular_path, join(tables_path, f'{filename}.pdf'), landscape=True, resizebox=True)
