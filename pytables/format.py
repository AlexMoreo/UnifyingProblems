from abc import ABC, abstractmethod
from idlelib.configdialog import changes
from re import template
from typing import Callable

import numpy as np

from comparison_group import *

from dataclasses import dataclass, replace, asdict


@dataclass
class Configuration:
    mean_prec: int = 3
    std_prec: int = 3
    # meanrank_prec: int = 1
    # meanrank_std_prec: int = 1
    show_std: bool = True
    # small_std: bool = True
    # rel_reduction_wrt: str = None
    # rel_increment_wrt: str = None
    # rel_val_prec: int = 2
    # remove_zero: bool = False
    with_color: bool = True
    maxtone: int = 40
    style: str = 'minimal'
    lower_is_better: bool = True
    stat_test: str = 'wilcoxon' # set to None to avoid
    stat_alpha: float = 0.95
    stat_mark: str = 'bold' # use "bold" or "dag"
    # color_mode: str = 'local'
    # with_mean: bool = True
    # with_rank_mean: bool = True
    # only_full_mean: bool = True
    best_color: str = 'green'
    worst_color: str = 'red'
    transpose: bool = False
    side_columns: bool = False
    resizebox: bool = None  # None is automatic
    best_in_bold: bool = True



def safe_mean(values):
    return np.mean(values) if len(values)>0 else np.nan


def safe_std(values):
    return np.std(values) if len(values)>0 else np.nan

# -------------------------------------------------
# Cell format
# -------------------------------------------------

@dataclass
class CellFormat(ABC):
    mean_prec: int = 3
    std_prec: int = 3
    show_std: bool = False
    bold_mean: bool = False
    underline_mean: bool = False
    italic_mean: bool = False
    dag: bool = False
    color_str: str = ''
    with_color: bool = False
    relative_variation: str = ''

    def mean_str(self, values):
        field_mean = '{mean:.'+str(self.mean_prec)+'f}'
        if self.bold_mean:
            field_mean = '\\textbf{{'+field_mean+'}}'
        if self.underline_mean:
            field_mean = '\\underline{{'+field_mean+'}}'
        if self.italic_mean:
            field_mean = '\\textit{{' + field_mean + '}}'
        mean_val = safe_mean(values)
        return field_mean.format(mean=mean_val)

    def std_str(self, values):
        field_std = ''
        if self.show_std:
            field_std = '\pm {std:.' + str(self.std_prec) + 'f}'
        std_val = safe_std(values)
        return field_std.format(std=std_val)

    def dag_str(self):
        dag = ''  # '^{{\phantom{{\dag}}}}'
        if self.dag:
            dag = '^\dag'
        return dag

    def relative_variation_str(self):
        rel_var_str = ''
        rel_var = self.relative_variation # a float value
        if rel_var!='' and rel_var is not None:
            sign = '+' if rel_var>=0 else '-'
            rel_var_str = f'\;({sign}{abs(rel_var) * 100:.2f}\%)'
        return rel_var_str

    def backgroundcolor_str(self):
        field_color = ''
        if self.with_color:
            field_color = self.color_str
        return field_color

    def to_str(self, values)->str:
        # compose the template
        field_mean = self.mean_str(values)
        field_std = self.std_str(values)
        field_dag = self.dag_str()
        field_rel_var = self.relative_variation_str()
        field_color = self.backgroundcolor_str()

        template = '${dag}{mean}{std}{rel_var}${color}'
        str_cell = template.format(dag=field_dag, mean=field_mean, std=field_std, rel_var=field_rel_var, color=field_color)

        return str_cell

    def clone(self) ->'CellFormat':
        return replace(self)  # returns a copy, with no changes


class Format:
    STYLES = ['minimal', 'rules', 'full']

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.user_format_modifiers = []

    def new_cell(self):
        cell_config = {k:v for k,v in asdict(self.configuration).items() if k in CellFormat.__annotations__}
        return CellFormat(**cell_config)

    def format(self, content_table: 'LatexTable'):
        # define a fresh format_table (an array of cells)
        nrows, ncols = content_table.shape
        format_table = np.asarray([[self.new_cell() for _ in range(ncols)] for _ in range(nrows)])
        
        conf = self.configuration
        
        # add the predefined format modifiers as indicated in the configuration file
        format_modifiers = []
        if conf.best_in_bold:
            format_modifiers.extend(self._get_format_best_in_bold(content_table))
        if conf.stat_test is not None:
            format_modifiers.extend(self._get_format_statistical_significance(content_table))
        if conf.with_color:
            format_modifiers.extend(self._get_format_background_color(content_table))
        
        # add the user-defined modifiers if any        
        all_format_modifiers = format_modifiers + self.user_format_modifiers
        
        # run all modifications
        for format_modifier in all_format_modifiers:
            format_modifier.modify(content_table, format_table)
            
        return format_table
    
    def _get_format_best_in_bold(self, content_table: 'LatexTable'):
        format_modifiers = []
        best_selector = MinComparisonMask if self.configuration.lower_is_better else MaxComparisonMask
        for benchmark in content_table.benchmarks:
            format_modifiers.append(
                FormatModifierBoldMean(
                    comparison=best_selector(SelectByName(benchmarks=benchmark))
                )
            )
        return format_modifiers

    def _get_format_background_color(self, content_table: 'LatexTable'):
        format_modifiers = []        
        conf = self.configuration
        
        if conf.lower_is_better:
            low_color, high_color = conf.best_color, conf.worst_color
        else:
            low_color, high_color = conf.worst_color, conf.best_color

        for benchmark in content_table.benchmarks:
            format_modifiers.append(
                FormatModifierColor(
                    comparison=ColourGroup(
                        selection_mask=SelectByName(benchmarks=benchmark)
                    ),
                    low_color=low_color,
                    high_color=high_color,
                    intensity=conf.maxtone
                )
            )

        return format_modifiers

    def _get_format_statistical_significance(self, content_table: 'LatexTable'):
        format_modifiers = []
        conf = self.configuration
        
        best_selection = MinComparisonMask if conf.lower_is_better else MaxComparisonMask
        
        if conf.stat_mark == 'dag':
            formatMarker = FormatModifierAddDagMean 
        elif conf.stat_mark == 'bold':
            formatMarker = FormatModifierBoldMean
        else:
            raise ValueError(f'unrecognized value {conf.stat_mark=}; valid values are "dag" and "bold"')
        
        for benchmark in content_table.benchmarks:
            format_modifiers.append(
                formatMarker(
                    comparison=SelectNotSignificantlyDifferentThan(
                        reference_selector=best_selection(SelectByName(benchmarks=benchmark)),
                        input_selector=SelectByName(benchmarks=benchmark),
                        stat_test=conf.stat_test,
                        alpha=conf.stat_alpha
                    )
                )
            )

        return format_modifiers

    def get_latex_style(self, n_columns):
        conf = self.configuration
        style_cmd={}
        hline = ' \hline'
        style_cmd['ENDL']=' \\\\'
        if conf.style == 'full':
            style_cmd['HLINE'] = hline
            style_cmd['TOPLINE'] = '\cline{2-'+str(n_columns)+'}'
            style_cmd['MIDLINE'] = hline
            style_cmd['BOTTOMLINE'] = hline
            style_cmd['CORNER'] = '\multicolumn{1}{c|}{}'
            style_cmd['column_format'] = '|c' * (n_columns) + '|'
        elif conf.style=='rules':
            style_cmd['HLINE'] = ''
            style_cmd['TOPLINE'] = '\\toprule'
            style_cmd['MIDLINE'] = '\\midrule'
            style_cmd['BOTTOMLINE'] = '\\bottomrule'
            style_cmd['CORNER'] = '\multicolumn{1}{c}{}'
            style_cmd['column_format'] = 'c' * (n_columns) + ''
        elif conf.style=='minimal':
            style_cmd['HLINE'] = ' '
            style_cmd['TOPLINE'] = '\cline{2-' + str(n_columns) + '}'
            style_cmd['MIDLINE'] = hline
            style_cmd['BOTTOMLINE'] = hline
            style_cmd['CORNER'] = '\multicolumn{1}{c|}{}'
            style_cmd['column_format'] = '|c' * (n_columns) + '|'
        else:
            raise ValueError(f'unknown stlye={conf.style}; valid ones are {Format.STYLES}')

        return style_cmd

# -------------------------------------------------
# Cell format Modifiers
# -------------------------------------------------

class FormatModifier(ABC):
    def __init__(self, comparison: 'ComparisonGroup'):
        self.comparison = comparison

    def modify(self, content_table: 'LatexTable', format_matrix:np.ndarray[CellFormat]):
        assert content_table.shape == format_matrix.shape, 'wrong shape'
        if isinstance(self.comparison, SelectionMask):
            to_modify_mask = self.comparison.run(content_table)
            for i, j in np.argwhere(to_modify_mask):
                self.format_change(format_matrix[i,j])
        elif isinstance(self.comparison, OperationGroup):
            to_modify_mask, values = self.comparison.run(content_table)
            for i, j in np.argwhere(to_modify_mask):
                self.format_change(format_matrix[i, j], values[i,j])

    @abstractmethod
    def format_change(self, format_cell:CellFormat, format_value=None):
        ...


class FormatModifierUnderlineMean(FormatModifier):
    def format_change(self, format_cell:CellFormat, format_value=None):
        format_cell.underline_mean = True


class FormatModifierItalizeMean(FormatModifier):
    def format_change(self, format_cell:CellFormat, format_value=None):
        format_cell.italic_mean = True


class FormatModifierBoldMean(FormatModifier):
    def format_change(self, format_cell:CellFormat, format_value=None):
        format_cell.bold_mean = True


class FormatModifierAddDagMean(FormatModifier):
    def format_change(self, format_cell:CellFormat, format_value=None):
        format_cell.dag = True


class FormatModifierColor(FormatModifier):
    def __init__(self, comparison: 'ComparisonGroup', low_color='green', high_color='red', intensity=50):
        super().__init__(comparison)
        self.low_color = low_color
        self.high_color = high_color
        self.intensity = intensity

    def format_change(self, format_cell:CellFormat, format_value=None):
        format_cell.with_color = True
        color = self.high_color if format_value>0 else self.low_color
        intensity = int(abs(format_value)*self.intensity)
        format_cell.color_str = '\cellcolor{'+color+'!'+str(intensity)+'}'


class FormatModifierRelativeValue(FormatModifier):
    def format_change(self, format_cell:CellFormat, format_value=None):
        sign = '+' if format_value>=0 else '-'
        format_cell.relative_variation = f'({sign}{abs(format_value)*100:.2f}\%)'


