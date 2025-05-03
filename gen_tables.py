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
# tasks = ['classifier_accuracy_prediction']
dataset_shifts=['label_shift', 'covariate_shift']
# dataset_shifts=['label_shift']


replace_method = {
    'HDcal8-sm-mono': 'DMCal',
    'ATC-q': r'ATC$_{\alpha 2 \rho}$',
    'DoC-q': r'DoC$_{\alpha 2 \rho}$',
    'LEAP-q': r'LEAP$_{\alpha 2 \rho}$',
    'LEAP-PCC-q': r'LEAP$_{\alpha 2 \rho}^{\text{PCC}}$',
    'LEAP-PCC': r'LEAP$^{\text{PCC}}$',
    'Cpcs-q-P': r'CPCS$_{\zeta 2 \rho}$',
    'CPCS-P': r'CPCS',
    'TransCal-q-P': r'TransCal$_{\zeta 2 \rho}$',
    'TransCal-S': r'TransCal',
    'LasCal-q-P': r'LasCal$_{\zeta 2 \rho}$',
    'LasCal-P': r'LasCal',
    'Head2Tail-q-P': r'Head2Tail$_{\zeta 2 \rho}$',
    'Head2Tail-P': r'Head2Tail',
    'Cpcs-a-S': r'CPCS$_{\zeta 2 \alpha}$',
    'TransCal-a-S': r'TransCal$_{\zeta 2 \alpha}$',
    'LasCal-a-P': r'LasCal$_{\zeta 2 \alpha}$',
    'PACC-a': r'PACC$_{\rho 2 \alpha}$',
    'PCC-a': r'PCC$_{\rho 2 \alpha}$',
    'EMQ-a': r'EMQ$_{\rho 2 \alpha}$',
    'EMQ-BCTS-a': r'EMQ$_{\rho 2 \alpha}^{\text{BCTS}}$',
    'KDEy-a': r'KDEy$_{\rho 2 \alpha}$',
    'HDc-a-sm-mono': r'DMCal$_{\zeta 2 \alpha}$',
    'EM': r'EMQ',
    'EM-BCTS': r'EMQ$^{\text{BCTS}}$',
    'EM-TransCal': r'EMQ$^{\text{TransCal}}$',
    'EMLasCal': r'EMQ$^{\text{LasCal}}$',
    'PACC-cal(soft)': r'PacCal$^{\sigma}$',
    'Bin6-PCC5': r'$\text{PCC}^{5\text{B}}_{\rho 2\zeta}$', #r'PCC$_{\zeta}^{5\text{B}}$',
    'Bin6-PACC5': r'PACC$_{\rho 2\zeta}^{5\text{B}}$',
    'Bin6-EM5': r'EMQ$_{\rho 2\zeta}^{5\text{B}}$',
    'Bin6-KDEy5': r'KDEy$_{\rho 2\zeta}^{5\text{B}}$',
    'Bin2-ATC6': r'ATC$_{\alpha 2\zeta}^{6\text{B}}$',
    'Bin2-DoC6': r'DoC$_{\alpha 2\zeta}^{6\text{B}}$',
    'Bin2-LEAP6': r'LEAP$_{\alpha 2\zeta}^{6\text{B}}$',
    'Bin2-LEAP-PCC-6': r'LEAP$_{\alpha 2\zeta}^{\text{PCC}-6\text{B}}$',
}


def get_error_name(task, dataset_shift):
    error = {
        'calibration': 'ece',
        'classifier_accuracy_prediction': 'err',
        'quantification': 'ae'
    }[task]
    return error


def get_baselines(task, dataset_shift):
    baselines = {
        'calibration': ['Platt'],
        'classifier_accuracy_prediction': ['Naive'],
        'quantification': ['CC']
    }[task]
    return baselines


def get_reference(task, dataset_shift):
    reference = {
        'calibration': ['Head2Tail', 'CPCS', 'TransCal', 'LasCal'],
        'classifier_accuracy_prediction': ['ATC', 'DoC', 'LEAP'],
        'quantification': ['PACC', 'EMQ', 'KDEy']
    }[task]

    # add specific methods
    if task=='calibration' and dataset_shift=='label_shift':
        reference = ['Head2Tail-P', 'CPCS-P', 'TransCal-S', 'LasCal-P']
    if task == 'quantification' and dataset_shift=='covariate_shift':
        reference = ['PCC'] + reference
    if task == 'classifier_accuracy_prediction' and dataset_shift=='covariate_shift':
        reference += ['LEAP-PCC']
    return reference


def get_contenders(task, dataset_shift):
    if task == 'calibration':
        if dataset_shift == 'label_shift':
            contenders = ['Bin6-PACC5', 'Bin6-EM5', 'Bin6-KDEy5', 'Bin2-ATC6', 'Bin2-DoC6', 'Bin2-LEAP6', 'EM', 'EM-BCTS', 'EMLasCal', 'PACC-cal(soft)', 'HDcal8-sm-mono']
        if dataset_shift=='covariate_shift':
            contenders = ['Bin6-PCC5', 'Bin6-PACC5', 'Bin6-EM5', 'Bin6-KDEy5', 'Bin2-ATC6', 'Bin2-DoC6', 'Bin2-LEAP6', 'Bin2-LEAP-PCC-6', 'EM', 'EM-BCTS', 'EM-TransCal', 'PACC-cal(soft)', 'HDcal8-sm-mono'] #, 'HDcal8-sm-mono', 'PACC-cal(clip)', 'Bin6-PACC5', 'Bin6-EM5', 'Bin6-KDEy5', 'Bin2-ATC6', 'Bin2-DoC6', 'Bin2-LEAP6']
    if task == 'classifier_accuracy_prediction':
        if dataset_shift=='label_shift':
            contenders = ['PACC-a', 'EMQ-a', 'KDEy-a', 'Cpcs-a-S', 'TransCal-a-S', 'LasCal-a-P', 'HDc-a-sm-mono']
        if dataset_shift=='covariate_shift':
            contenders = ['PCC-a', 'PACC-a', 'EMQ-a', 'EMQ-BCTS-a', 'KDEy-a', 'Cpcs-a-S', 'TransCal-a-S', 'LasCal-a-P', 'HDc-a-sm-mono']
    if task == 'quantification':
        if dataset_shift == 'label_shift':
            contenders = ['ATC-q', 'DoC-q', 'LEAP-q', 'Cpcs-q-P', 'TransCal-q-P', 'LasCal-q-P'] #, 'Head2Tail-q-P']
        if dataset_shift == 'covariate_shift':
            contenders = ['ATC-q', 'DoC-q', 'LEAP-q', 'LEAP-PCC-q', 'Cpcs-q-P', 'TransCal-q-P', 'LasCal-q-P']

    return contenders

def get_classifiers(task, dataset_shift):
    classifiers = {
        'covariate_shift': ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base'],
        'label_shift': ['lr', 'nb', 'knn', 'mlp'] if task!='quantification' else ['']
    }[dataset_shift]
    return classifiers


def get_datasets(task, dataset_shift):
    if dataset_shift=='covariate_shift':
        domains = ['imdb', 'rt', 'yelp']
        datasets = []
        for source in domains:
            for target in domains:
                if source==target: continue
                datasets.append(f'{source}->{target}')
    else:
        datasets = commons.uci_datasets()
    return datasets


def load_results(results_path, all_methods):
    all_dfs = []
    for result_file in glob(join(results_path, '*.csv')):
        methodname, classifier, *_ = Path(result_file).name.split('_')
        if methodname in all_methods:
            df = pd.read_csv(result_file, index_col=None)
            assert len(df) == 100, 'unexpected number of rows'
            all_dfs.append(df)

    df = pd.concat(all_dfs)
    print(f'read {len(df)} rows')
    return df


def tikz2pdf(tikz_path, pdf_path, landscape=False):
    parent = Path(pdf_path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)

    doc_path = pdf_path.replace('.pdf', '.tex')

    tickz_path_rel = os.path.relpath(tikz_path, parent)
    tickz_str = r'\input{'+tickz_path_rel+'} \n'
    tex_document(doc_path, [tickz_str], landscape=landscape, add_package=['tikz', 'pgfplots'])
    latex2pdf(pdf_path)


def critical_difference_diagram(cd_df, task, dataset_shift, diagrams_folder):
    filename = f'{task}-{dataset_shift}'
    tex_path = join(diagrams_folder, f'diagrams/{filename}.tex')
    pdf_path = join(diagrams_folder, f'{filename}.pdf')
    util.makepath(tex_path)

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

    # export the diagram to a file
    diagram.to_file(
        tex_path,
        alpha=.05,
        adjustment="holm",
        reverse_x=False,
        axis_options={"title": f"{task} {dataset_shift}".replace('_', r'\_')},
    )
    tikz2pdf(tex_path, pdf_path)


def gen_table(tables_folder):
    tables = {}
    for i, classifier in enumerate(classifiers):
        if classifier == '':
            df_block = df
        else:
            df_block = df[df['classifier'] == classifier]
        table = LatexTable.from_dataframe(df_block, method='method', benchmark='dataset', value=error)
        table.reorder_methods(all_methods)
        table.reorder_benchmarks(datasets)
        table.format.configuration.show_std = False
        tables[classifier] = table

    n_setups = len(datasets)
    n_classifiers = len(classifiers)
    n_methods = len(all_methods)
    n_baselines = len(baselines)
    n_reference = len(reference)
    n_contenders = len(contenders)
    # n_rows = 2 + n_setups * n_classifiers
    n_cols = 2 + n_methods

    config = Configuration()
    config.style = 'rules'
    config.side_columns = True
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

    replace_classifier = {
        'bert-base-uncased': 'BERT',
        'distilbert-base-uncased': 'DistilBERT',
        'roberta-base': 'RoBERTa',
        'lr': 'Logistic Regression',
        'nb': 'Naïve Bayes',
        'knn': 'k Nearest Neighbor',
        'mlp': 'Multi-layer Perceptron'
    }

    # column_format = style['column_format']
    column_format = 'cc|' + 'c' * n_baselines + '|' + 'c' * n_reference + '|' + 'c' * n_contenders
    lines = []
    lines.append(r'\begin{tabular}{' + column_format + '} ' + style['TOPLINE'])
    lines.append(r'\multicolumn{2}{c}{} & \multicolumn{' + str(n_baselines) + r'}{c|}{Baselines} & \multicolumn{' + str(
        n_reference) + r'}{c|}{Reference} & \multicolumn{' + str(n_contenders) + r'}{c}{Adapted Methods}' + style['ENDL'])
    for i, classifier in enumerate(classifiers):
        table_arr = tables[classifier].as_str_array()
        prepare_strings(table_arr)
        if i == 0:
            for j in range(1, table_arr.shape[1]):
                table_arr[0, j] = replace_method.get(table_arr[0, j], table_arr[0, j])
            if config.side_columns:
                for j in range(1, table_arr.shape[1]):
                    table_arr[0, j] = r'\begin{sideways}' + table_arr[0, j] + r'\;\end{sideways}'
            header = r'\multicolumn{2}{c}{} & ' + ' & '.join(table_arr[0, 1:]) + style['ENDL'] + style['MIDLINE']
            lines.append(header)
        for j, row in enumerate(table_arr[1:]):
            endl = style['HLINE']
            if j == table_arr.shape[0] - 2:
                if i < (len(classifiers) - 1):
                    endl = style['MIDLINE']
                else:
                    endl = style['BOTTOMLINE']

            first_col = ' & ' if j != 0 else (
                    r'\multirow{' +
                    str(table_arr.shape[0] - 1) +
                    r'}{*}{\begin{sideways}' +
                    replace_classifier.get(classifier, classifier) +
                    r'\;\end{sideways}} & '
            )
            line = first_col + ' & '.join(row) + style['ENDL'] + endl
            lines.append(line)

        if i == n_classifiers - 1:
            # add the method vs. reference comparisons
            for j in range(1, n_reference + 1):
                first_col = ' & ' if j != 1 else (
                            r'\multirow{' + str(n_reference) + r'}{*}{\begin{sideways}Wins\;\end{sideways}} & ')
                second_col = r' $\Pr(M\succ {' + str(j) +r'}R)$ &'
                parts = []
                for k, method in enumerate(all_methods):
                    if method in counts:
                        count = counts[method][j] * 100
                        stat = reject_H0[method][j]
                        dag = r'\dag ' if stat else ''
                        parts.append(f'${dag}{count:.2f}'+r'\%$')
                    else:
                        parts.append('---')
                endl = style['HLINE'] if j < n_reference else style['BOTTOMLINE']
                line = first_col + second_col + ' & '.join(parts) + style['ENDL'] + endl
                lines.append(line)

            # add the average ranks
            method_ranks = np.asarray([ranks[method] for method in all_methods])
            order = np.argsort(method_ranks)
            top_n_ranks = set(order[:n_reference])
            parts = [f'{m_rank:.2f}' for m_rank in method_ranks]
            parts = [f'\\textbf{{{m_rank}}}' if pos in top_n_ranks else m_rank for pos, m_rank in enumerate(parts)]
            line = '  & Ave Rank &' + ' & '.join(parts) + style['ENDL'] + style['BOTTOMLINE']
            lines.append(line)

    lines.append('\\end{tabular}')

    filename = f'{task}_{dataset_shift}'
    tabular_path = join(tables_folder, 'tabular', f'{filename}.tex')
    util.save_text(tabular_path, '\n'.join(lines))
    tabular2pdf(tabular_path, join(tables_folder, f'{filename}.pdf'), landscape=False, resizebox=True)

if __name__ == '__main__':
    folder = commons.EXPERIMENT_FOLDER
    for task, dataset_shift in itertools.product(tasks, dataset_shifts):
        results_path = join('./results', task, dataset_shift, folder)

        error = get_error_name(task, dataset_shift)

        baselines = get_baselines(task, dataset_shift)
        reference = get_reference(task, dataset_shift)
        contenders = get_contenders(task, dataset_shift)

        classifiers = get_classifiers(task, dataset_shift)
        datasets = get_datasets(task, dataset_shift)

        all_methods = baselines + reference + contenders

        df = load_results(results_path, all_methods)

        print(task, dataset_shift)
        counts, reject_H0 = util.count_successes(df, baselines=reference, value=error, expected_repetitions=100)
        ranks, cd_df = util.get_ranks(df, value=error, expected_repetitions=100)

        cd_df['method'] = cd_df['method'].replace(replace_method)
        critical_difference_diagram(cd_df, task, dataset_shift, diagrams_folder='./cddiagrams/')
        gen_table(tables_folder='./fulltables/')

