import operator

from scipy.stats import wilcoxon, ttest_ind_from_stats
import numpy as np
from abc import ABC, abstractmethod
from util import *



class SelectionMask(ABC):

    NAN_POLICIES = ['ignore', 'raise']

    def __init__(self, nan_policy='ignore'):
        assert nan_policy in SelectionMask.NAN_POLICIES, 'unknown nan policy'
        self.nan_policy = nan_policy

    @abstractmethod
    def run(self, content_table: 'ContentTable')->np.ndarray:
        ...


class SelectNan(SelectionMask):
    def __init__(self, nan_policy='ignore'):
        super().__init__(nan_policy)

    def run(self, content_table: 'ContentTable') -> np.ndarray:
        nans = np.isnan(content_table.means())
        if nans.any() and self.nan_policy == 'raise':
            raise ValueError('NaN found in content table; probably there are some empty cells')
        return nans


class SelectByName(SelectionMask):
    def __init__(self, benchmarks=None, methods=None, nan_policy='ignore'):
        super().__init__(nan_policy)
        self.benchmarks = benchmarks
        self.methods = methods

    def run(self, content_table: 'ContentTable') -> np.ndarray:
        def _select_in_dimension(selection_names, reference_names):
            if selection_names is None:
                selection_names = reference_names
            if isinstance(selection_names, str):
                selection_names = [selection_names]
            return np.isin(reference_names, selection_names)

        selected_benchmakrs = _select_in_dimension(self.benchmarks, content_table.benchmarks)
        selected_methods = _select_in_dimension(self.methods, content_table.methods)

        sel_mask = np.outer(selected_benchmakrs, selected_methods)

        nans = SelectNan(self.nan_policy).run(content_table)

        return sel_mask & ~nans


def blank_mask_like(content_table: 'ContentTable'):
    return np.full(shape=content_table.shape, fill_value=False, dtype=bool)


class NumpyOpComparisonMask(SelectionMask):
    def __init__(self, input_selector: 'SelectionMask', numpy_operation=None):
        super().__init__()
        assert numpy_operation is not None, 'numpy operation not understood'
        self.input_selector = input_selector

        if callable(numpy_operation):
            self.numpy_operation = numpy_operation
        elif isinstance(numpy_operation, str):
            try:
                self.numpy_operation = getattr(np, numpy_operation)
            except AttributeError:
                raise ValueError(f'unknown numpy operation {numpy_operation}')
        else:
            raise ValueError(f'unexpected type for numpy_operation')


    def run(self, content_table: 'ContentTable')->np.ndarray:
        selection_mask = self.input_selector.run(content_table)
        means = content_table.means()
        np_val = self.numpy_operation(means[selection_mask])
        return means == np_val


class MaxComparisonMask(NumpyOpComparisonMask):
    def __init__(self, input_selector: 'SelectionMask'):
        super().__init__(input_selector, numpy_operation=np.max)


class MinComparisonMask(NumpyOpComparisonMask):
    def __init__(self, input_selector: 'SelectionMask'):
        super().__init__(input_selector, numpy_operation=np.min)


class CompareMagnitudeMask(SelectionMask):

    valid_comparisons = ['gt', 'lt']

    # identifies all the cells in input_selector which have a mean value greater or smaller than
    # all mean values indicated by the reference_selector
    def __init__(self, reference_selector: 'SelectionMask', input_selector: 'SelectionMask', comp='gt'):
        super().__init__()
        assert comp in self.valid_comparisons, f'unknown comparison'
        self.reference_selector = reference_selector
        self.input_selector = input_selector
        self.comp = comp

    def run(self, content_table: 'ContentTable') -> np.ndarray:
        if self.comp == 'gt':
            OperationMask = MaxComparisonMask
            comparison = operator.gt
        elif self.comp == 'lt':
            OperationMask = MinComparisonMask
            comparison = operator.lt
        else:
            raise ValueError(f'unknwon comparison mode, valid ones are {self.valid_comparisons}')

        means = content_table.means()

        top_ref_position = OperationMask(self.reference_selector).run(content_table)
        assert top_ref_position.sum()==1, 'more than 1 reference values selected'
        top_ref_mean = means[top_ref_position]

        selection_mask = self.input_selector.run(content_table)
        comp_outcome = blank_mask_like(content_table)
        for i, j, values in content_table.iter(selection_mask):
            mean_ij = safe_mean(values)
            comp_outcome[i, j] = comparison(mean_ij, top_ref_mean)
        return comp_outcome


class SelectGreaterThan(CompareMagnitudeMask):
    # identifies all the cells in selection_mask which have a mean value greater than
    # all mean values indicated by the reference_mask
    def __init__(self,
                 reference_selector: 'SelectionMask',
                 input_selector: 'SelectionMask',
                 nan_policy='ignore'):
        super().__init__(reference_selector, input_selector, comp='gt')


class SelectSmallerThan(CompareMagnitudeMask):
    # identifies all the cells in selection_mask which have a mean value smaller than
    # all mean values indicated by the reference_mask
    def __init__(self,
                 reference_selector: 'SelectionMask',
                 input_selector: 'SelectionMask',
                 nan_policy='ignore'):
        super().__init__(reference_selector, input_selector, comp='lt')

            
class SelectSignificantlyDifferentThan(SelectionMask):
    valid_tests = ['wilcoxon', 'ttest']

    # identifies all the cells in selection_mask which have a mean value statistically significantly different than
    # the (only) one of the reference_mask
    def __init__(self,
                 reference_selector: 'SelectionMask',
                 input_selector: 'SelectionMask',
                 stat_test='wilcoxon',
                 alpha=0.95):
        assert stat_test in self.valid_tests, f'unknown test; valid ones are {self.valid_tests}'
        self.reference_selector = reference_selector
        self.input_selector = input_selector
        self.stat_test = stat_test
        self.alpha = alpha
        super().__init__()

    def check_significant_differences(self, ref_vals, vals):
        pval = None
        if self.stat_test=='wilcoxon':
            pval = wilcoxon(ref_vals, vals).pvalue
        elif self.stat_test=='ttest':
            _, pval = ttest_ind_from_stats(
                ref_vals.mean(), ref_vals.std(), len(ref_vals),
                vals.mean(), vals.std(), len(vals)
            )
        else:
            raise NotImplementedError(f'unknown test of statistical significance "{self.stat_test}"')
        return pval < (1-self.alpha)

    def run(self, content_table: 'ContentTable')->np.ndarray:
        ref_selection = self.reference_selector.run(content_table)
        assert ref_selection.sum()==1, 'more than one reference cells selected'
        ref_values = content_table.table[ref_selection][0]

        selection_mask = self.input_selector.run(content_table)
        selection_mask = selection_mask & ~ref_selection  # do not compare against itself
        significant_differences = blank_mask_like(content_table)
        for i, j, values in content_table.iter(selection_mask):
            significant_differences[i,j] = self.check_significant_differences(ref_values, values)
        return significant_differences


class SelectNotSignificantlyDifferentThan(SelectSignificantlyDifferentThan):
    def run(self, content_table: 'ContentTable') ->np.ndarray:
        ref_selection = self.reference_selector.run(content_table)
        # assert ref_selection.sum()==1, 'more than one reference cells selected'
        # if ref_selection.sum()>1:
        ref_values = content_table.table[ref_selection][0]

        selection_mask = self.input_selector.run(content_table)
        selection_mask = selection_mask & ~ref_selection  # do not compare against itself
        significant_differences = blank_mask_like(content_table)
        for i, j, values in content_table.iter(selection_mask):
            significant_differences[i,j] = not self.check_significant_differences(ref_values, values)
        return significant_differences


class SelectSignificantlyGreaterThan(SelectionMask):
    # identifies all the cells in selection_mask which have a mean value statistically significantly greater than
    # all the cells in the reference_mask
    def __init__(self,
                 reference_selector: 'SelectionMask',
                 input_selector: 'SelectionMask',
                 stat_test='wilcoxon',
                 pval=0.05):
        self.reference_selector = reference_selector
        self.input_selector = input_selector
        self.stat_test = stat_test
        self.pval = pval
        super().__init__()

    def run(self, content_table: 'ContentTable')->np.ndarray:
        max_ref_mask = MaxComparisonMask(self.reference_selector)
        greater = SelectGreaterThan(max_ref_mask, self.input_selector)
        greater_and_different = SelectSignificantlyDifferentThan(max_ref_mask, greater, self.stat_test, self.pval)
        return greater_and_different.run(content_table)


class SelectNotSignificantlyGreaterThan(SelectSignificantlyGreaterThan):
    def run(self, content_table: 'ContentTable') -> np.ndarray:
        max_ref_mask = MaxComparisonMask(self.reference_selector)
        greater = SelectGreaterThan(max_ref_mask, self.input_selector)
        greater_and_notdifferent = SelectNotSignificantlyDifferentThan(max_ref_mask, greater, self.stat_test, self.pval)
        return greater_and_notdifferent.run(content_table)


class SelectSignificantlySmallerThan(SelectionMask):
    # identifies all the cells in selection_mask which have a mean value statistically significantly smaller than
    # all the cells in the reference_mask
    def __init__(self,
                 reference_selector: 'SelectionMask',
                 input_selector: 'SelectionMask',
                 stat_test='wilcoxon',
                 pval=0.05):
        self.reference_selector = reference_selector
        self.input_selector = input_selector
        self.stat_test = stat_test
        self.pval = pval
        super().__init__()

    def run(self, content_table: 'ContentTable')->np.ndarray:
        max_ref_mask = MinComparisonMask(self.reference_selector)
        smaller = SelectSmallerThan(max_ref_mask, self.input_selector)
        smaller_and_different = SelectSignificantlyDifferentThan(max_ref_mask, smaller, self.stat_test, self.pval)
        return smaller_and_different.run(content_table)


class SelectNotSignificantlySmallerThan(SelectSignificantlySmallerThan):
    def run(self, content_table: 'ContentTable') ->np.ndarray:
        max_ref_mask = MinComparisonMask(self.reference_selector)
        smaller = SelectSmallerThan(max_ref_mask, self.input_selector)
        smaller_and_notdifferent = SelectNotSignificantlyDifferentThan(max_ref_mask, smaller, self.stat_test, self.pval)
        return smaller_and_notdifferent.run(content_table)


class OperationGroup(ABC):
    def __init__(self, nan_policy='ignore'):
        assert nan_policy in SelectionMask.NAN_POLICIES, 'unknown nan policy'
        self.nan_policy = nan_policy

    @abstractmethod
    def run(self, content_table: 'ContentTable')->[np.ndarray, np.ndarray]:
        """
        Returns a mask of selected values and an array of float values
        :param content_table:
        :return: a selection mask (boolean ndarray) and a matrix of float values, the
            outputs of the operation
        """
        ...

class RelativeVariationGroup(OperationGroup):
    def __init__(self, reference_mask: SelectionMask, selection_mask: SelectionMask):
        super().__init__()
        self.reference_mask = reference_mask
        self.selection_mask = selection_mask

    def run(self, content_table: 'ContentTable') ->[np.ndarray, np.ndarray]:
        """
        Returns a mask of the selected items for which the relative variation is to be indicated,
        and a matrix of floats with the relative variations (i.e., (val-ref_val)/ref_val)
        :param content_table:
        :return:
        """
        ref_mask = self.reference_mask.run(content_table)
        assert ref_mask.sum()==1, 'more than one value have been selected; could not compare the relative variation'
        selected_mask = self.selection_mask.run(content_table)

        # remove the reference position from the selection mask, to avoid adding a trivial "(+0.00%)"
        selected_mask = selected_mask & ~ref_mask

        means = content_table.means()
        ref_val = means[ref_mask][0]
        vals = means[selected_mask]
        relative_vals = (vals-ref_val)/ref_val

        relative_vals_arr = np.zeros(shape=content_table.shape, dtype=float)
        relative_vals_arr[selected_mask] = relative_vals

        return selected_mask, relative_vals_arr


class ColourGroup(OperationGroup):

    def __init__(self, selection_mask: SelectionMask):
        super().__init__()
        self.selection_mask = selection_mask

    def run(self, content_table: 'ContentTable') ->[np.ndarray, np.ndarray]:
        """
        Returns a mask of the selected items to colour and a matrix with the intensity values, ranging from
        -1 (the smallest value) to +1 (the highest value) and all other values interpolated between those two
        :param content_table:
        :return:
        """
        min_pos = MinComparisonMask(self.selection_mask).run(content_table)
        max_pos = MaxComparisonMask(self.selection_mask).run(content_table)
        means = content_table.means()
        min_val = means[min_pos][0]
        max_val = means[max_pos][0]

        selected_mask = self.selection_mask.run(content_table)
        selected_means = means[selected_mask]
        colours = np.interp(selected_means, [min_val, max_val], [-1,+1])
        colour_map = np.full(shape=content_table.shape, fill_value=0., dtype=float)
        colour_map[selected_mask] = colours
        return selected_mask, colour_map