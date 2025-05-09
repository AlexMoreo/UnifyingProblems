from itertools import product
import re
from typing import List
from dataclasses import asdict
import warnings
import pandas as pd
import numpy as np
from collections import Counter
import tools
from util import *
from format import Configuration, Format
from format import FormatModifier
import os
from os.path import join
from pathlib import Path
# import tools

class LatexTable:

    def __init__(self, name='mytable', configuration: Configuration=None):
        self.name = name
        self.benchmarks = np.empty(0)
        self.methods = np.empty(0)
        self.table = None
        self._cached_means = None
        if configuration is None:
            configuration = Configuration()
        self.format = Format(configuration)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, method: str, benchmark: str, value: str, name='mytable', configuration: Configuration=None):
        instance = cls(name=name, configuration=configuration)
        grouped = dataframe.groupby([benchmark, method])[value].apply(list)
        for (m, b), values in grouped.items():
            instance.add(m, b, values)
        return instance

    @property
    def n_methods(self):
        return len(self.methods)

    @property
    def n_benchmarks(self):
        return len(self.benchmarks)

    def add(self, benchmark, method, v):
        assert isinstance(benchmark, str), 'the benchmark name is not a str'
        assert isinstance(method, str),    'the method name is not a str'
        self.clear_cache()

        def _new_empty_arr(rows=1, cols=1):
            # np.full does not work, nor list comprehension
            arr = np.empty(shape=(rows, cols), dtype=object)
            for i, j in product(range(rows), range(cols)):
                arr[i,j]=list()
            return arr

        first_addition = self.table is None

        if benchmark not in self.benchmarks:
            self.benchmarks = np.append(self.benchmarks, benchmark)
            if not first_addition:
                new_row = _new_empty_arr(cols=self.n_methods)
                self.table = np.vstack([self.table, new_row])
        bench_idx = np.where(self.benchmarks==benchmark)[0][0]

        if method not in self.methods:
            self.methods = np.append(self.methods, method)
            if not first_addition:
                new_col = _new_empty_arr(rows=self.n_benchmarks)
                self.table = np.hstack([self.table, new_col])
        meth_idx = np.where(self.methods==method)[0][0]

        if first_addition:
            self.table = _new_empty_arr()

        def check_type(v):
            return isinstance(v, int) or isinstance(v, float)

        if hasattr(v, "__len__"):
            assert all(check_type(vi) for vi in v), 'wrong data type; use int or float'
            self.table[bench_idx, meth_idx].extend(v)
        else:
            assert check_type(v), 'wrong data type; use int or float'
            self.table[bench_idx, meth_idx].append(v)

    def clear_cache(self):
        self._cached_means = None

    def means(self):
        if self._cached_means is None:
            means = np.zeros(shape=(self.n_benchmarks, self.n_methods), dtype=float)
            for row, col, vals in self.iter():
                means[row, col] = np.mean(vals) if len(vals)>0 else np.nan
            self._cached_means = means
        return self._cached_means

    def iter(self, selection_mask=None):
        if selection_mask is None:
            for i in range(self.n_benchmarks):
                for j in range(self.n_methods):
                    yield i, j, self.table[i, j]
        else:
            assert isinstance(selection_mask, np.ndarray) and selection_mask.dtype==bool, 'wrong selection mask format'
            for (i, j) in np.argwhere(selection_mask):
                yield i, j, self.table[i, j]

    @property
    def shape(self):
        return tuple((self.n_benchmarks, self.n_methods))

    def empties(self):
        empties = np.full_like(self.table, fill_value=False)
        for i, j, v in self.iter():
            empties[i,j] = len(v)==0
        return empties

    def reorder_benchmarks(self, order=None):
        if order is None:
            order = sorted(self.benchmarks)
        self.reorder(benchmarks_order=order)

    def reorder_methods(self, order=None):
        if order is None:
            order = sorted(self.methods)
        self.reorder(methods_order=order)

    def reorder(self, benchmarks_order=None, methods_order=None):
        """
        :param benchmarks_order:
        :param methods_order:
        :return:
        """
        if benchmarks_order is None and methods_order is None:
            raise ValueError('both order arguments are None')

        def prepare_order(names_order, names_reference):
            if names_order is None:
                names_order = names_reference
            if sorted(names_order)!=sorted(names_reference):
                raise ValueError(f'order list {names_order} does not match all elements from {names_reference}')
            names_order = np.asarray(names_order)
            names_reindex = [np.where(names_reference == r)[0][0] for r in names_order]
            return names_order, names_reindex

        benchmarks_order, benchmarks_reindex = prepare_order(benchmarks_order, self.benchmarks)
        methods_order, methods_reindex = prepare_order(methods_order, self.methods)

        self.table = self.table[np.ix_(benchmarks_reindex, methods_reindex)]
        self.benchmarks = benchmarks_order
        self.methods = methods_order

        self.clear_cache()

    def __str__(self):
        df = pd.DataFrame(
            {
                'benchmark': self.benchmarks[i],
                'method': self.methods[j],
                'value': np.mean(values) if len(values)>0 else np.nan
            } for i, j, values in self.iter()
        )
        return str(df.pivot_table(index='benchmark', columns='method', values='value'))

    def add_format_modifier(self, format_modifier:FormatModifier):
        self.format.user_format_modifiers.append(format_modifier)

    def as_str_array(self):
        str_array = np.full(shape=(self.n_benchmarks + 1, self.n_methods + 1), dtype=object, fill_value='')
        str_array[0, 1:] = self.methods
        str_array[1:, 0] = self.benchmarks
        format_table = self.format.format(self)
        for i, j, values in self.iter():
            str_array[i + 1, j + 1] = format_table[i, j].to_str(values)
        return str_array

    def print_tmp(self):
        print(self.as_str_array())

    def tex_tabular(self):
        conf = self.format.configuration
        n_rows, n_columns = (self.n_benchmarks+1, self.n_methods+1)
        if conf.transpose:
            n_rows, n_columns = n_columns, n_rows

        warnings.filterwarnings("ignore", message="character scape warning")
        style = self.format.get_latex_style(n_columns)

        str_array = self.as_str_array()
        if conf.transpose:
            str_array = str_array.T
        str_array[0,0] = style['CORNER']

        # protect "_" in methods and benchmarks
        def escape_latex_underscore(s):
            return re.sub(r'(?<!\\)_', r'\_', s)

        for j in range(1,str_array.shape[1]):
            str_array[0,j] = escape_latex_underscore(str_array[0,j])
        for i in range(1,str_array.shape[0]):
            str_array[i,0] = escape_latex_underscore(str_array[i,0])

        if conf.side_columns:
            for j in range(1, str_array.shape[1]):
                str_array[0, j] = '\\begin{sideways}'+str_array[0, j]+'\;\end{sideways}'

        lines = []

        # begin tabular and column format
        lines.append('\\begin{tabular}{' + style['column_format'] + '} '+style['TOPLINE'])
        lines.append(' & '.join(str_array[0]) + style['ENDL'] + style['MIDLINE'])
        for row_elements in str_array[1:-1]:
            lines.append(' & '.join(row_elements) + style['ENDL'] + style['HLINE'])
        lines.append(' & '.join(str_array[-1]) + style['ENDL'] + style['BOTTOMLINE'])
        lines.append('\\end{tabular}')

        tabular_tex = '\n'.join(lines)

        return tabular_tex


    @classmethod
    def LatexPDF(cls, pdf_path: str, tables:List['LatexTable'], tabular_dir: str = 'tables', delete_tex=True, *args, **kwargs):

        assert pdf_path.endswith('.pdf'), f'{pdf_path=} does not seem a valid name for a pdf file'

        pdf_folder = Path(pdf_path).parent
        if pdf_folder:
            os.makedirs(pdf_folder, exist_ok=True)

        table_names = rename_file_names([table.name for table in tables])

        tables_str = []
        for tabular, tabular_name in zip(tables, table_names):
        
            tabular_str = tabular.tex_tabular()

            resizebox = tabular.format.configuration.resizebox
            if resizebox is None:
                resizebox = (tabular.n_methods >= 8) or (tabular.n_benchmarks >= 8)
            
            tabular_rel_path = join(tabular_dir, tabular_name+'.tex')
            save_text(text=tabular_str, path=join(pdf_folder, tabular_rel_path))

            table_str = tools.tex_table(tabular_rel_path, resizebox=resizebox)
            tables_str.append(table_str)

        doc_path  = pdf_path.replace('.pdf', '.tex')
        doc_str = tools.tex_document(doc_path, tables_str)
        tools.latex2pdf(pdf_path, delete_tex=delete_tex)


    def latexPDF(self, pdf_path, tabular_dir='tables', *args, **kwargs):
        return LatexTable.LatexPDF(pdf_path, tables=[self], tabular_dir=tabular_dir, *args, **kwargs)


def rename_file_names(names):
    """
    Renames the element in names so that there are no duplicates.
    E.g.: names=[table, my_table, table]
    returns [table_0, my_table, table_1]
    :param names: the file names
    :return: file names (in the same order) renamed so that there are no duplicates
    """
    if len(set(names)) == len(names):
        # all different, nothing to do
        return names
    counts = Counter(names)
    deduplicated_names = []
    for i, name in enumerate(names):
        repeated = counts[name]>1
        if repeated:
            new_name = f'{name}_{names[:i].count(name)}'
            if new_name in deduplicated_names:
                raise ValueError('error while renaming tables; please indicate unique table names')
            deduplicated_names.append(new_name)
        else:
            deduplicated_names.append(name)
    return deduplicated_names


def save_text(text, path=None):
    parent = Path(path).parent
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'wt') as foo:
        foo.write(text)

if __name__ == '__main__':
    from comparison_group import *
    table = LatexTable()
    np.random.seed(0)
    table.add('yeast', 'pacc', np.random.normal(loc=-7, scale=0.1, size=50))
    table.add('yeast', 'baseline', np.random.normal(loc=0.05, scale=0.5, size=50))
    table.add('abalone', 'kde', np.random.normal(loc=1.0, scale=1, size=50))
    table.add('abalone', 'pacc', np.random.normal(loc=1.5, scale=1, size=50))
    table.add('abalone', 'baseline', np.random.normal(loc=1.0, scale=0.5, size=50))
    table.add('wine', 'kde', np.random.normal(loc=0.0, scale=1, size=50))
    table.add('abalone', 'baseline2', np.random.normal(loc=-1.0, scale=0.5, size=50))
    table.add('wine', 'pacc', np.random.normal(loc=0.65, scale=1, size=50))
    table.add('wine', 'baseline', np.random.normal(loc=0.75, scale=0.5, size=50))
    table.add('wine', 'baseline2', np.random.normal(loc=1.75, scale=0.5, size=50))
    table.add('yeast', 'kde', np.random.normal(loc=-7, scale=0.1, size=50))
    table.add('yeast', 'baseline2', np.random.normal(loc=0.05, scale=0.5, size=50))

    # table.add_format_best_in_bold()
    # table.add_format_best_baseline_underlined(baselines=['baseline', 'baseline2'])
    # table.add_format_background_color()
    # table.add_format_statistical_significance()
    # table.add_format_relative_variation(reference_method='baseline')

    # print(table.tex_table())

    table.reorder(benchmarks_order=['abalone', 'wine', 'yeast'])
    table.reorder_methods()
    table.latexPDF('../examples/new_table3/mypdf.pdf')

    #table.latexPDF('../examples/new_table/main.pdf')
    # tabular_path = '../examples/new_table3/tabular.tex'
    # pdf_path = '../examples/new_table3/mypdf.pdf'
    # table.tex_table(tabular_path)
    # tools.tabular2pdf(tabular_path, pdf_path)
    
    
    
    


