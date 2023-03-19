"""general_tools.py"""

from itertools import zip_longest
import pandas as pd


def apply_column_filter(df: pd.DataFrame, filter_map: dict)->pd.DataFrame:
    """
    Returns a DataFrame after applying a columnwise filter using a dict
    :param df: DataFrame
    :param filter_map: dict
        keys: names of the columns in the DataFrame that will be filtered
        values: str or list of str that are included
    :return: DataFrame
    """
    result_df = df.copy()
    for key, value in filter_map.items():
        if not isinstance(value, list):
            value = [value]
        result_df = result_df[result_df[key].str.lower().isin([v.lower() for v in value.copy()])]
    return result_df


def _grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def list_grouper(iterable: {list, tuple}, n: int, fill_value=None):
    """
    Returns a list of lists where each sub-list is of size 'n'
    :param iterable: list or tuple
    :param n: length of each sub-list
    :param fill_value: value to be populated as an element into a sub-list that does not have 'n' elements
    :return:
    """
    g = list(_grouper(iterable, n, fill_value))
    try:
        g[-1] = [e for e in g[-1] if e is not None]
        return [list(tup) for tup in g]
    except IndexError:
        return [[]]
