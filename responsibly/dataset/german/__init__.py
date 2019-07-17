import json

import numpy as np
import pandas as pd
from pkg_resources import resource_filename, resource_stream

from responsibly.dataset.core import Dataset


__all__ = ['GermanDataset']

GERMAN_PATH = resource_filename(__name__,
                                'german.data')

VALUES_MAPS = json.loads(resource_stream(__name__,
                                         'values_maps.json')
                         .read()
                         .decode())

COLUMN_NAMES = ['status', 'duration', 'credit_history', 'purpose',
                'credit_amount', 'savings', 'present_employment',
                'installment_rate', 'status_sex', 'other_debtors',
                'present_residence_since', 'property', 'age',
                'installment_plans', 'housing',
                'number_of_existing_credits', 'job',
                'number_of_people_liable_for', 'telephone',
                'foreign_worker', 'credit']


class GermanDataset(Dataset):
    """German Credit Dataset.

    See :class:`~responsibly.dataset.Dataset` for a description of
    the arguments and attributes.

    References:
        - https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
        - Kamiran, F., & Calders, T. (2009, February).
          Classifying without discriminating.
          In 2009 2nd International Conference on Computer, Control
          and Communication (pp. 1-6). IEEE.
          http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.182.6067&rep=rep1&type=pdf


    Extra
        This dataset requires use of a cost matrix (see below)

        ::

               1 2
               ----
            1 | 0 1
              |----
            2 | 5 0

        (1 = Good, 2 = Bad)

        The rows represent the actual classification
        and the columns the predicted classification.
        It is worse to class a customer as good when they are bad (5),
        than it is to class a customer as bad when they are good (1).

    """

    def __init__(self):
        super().__init__(target='credit',
                         sensitive_attributes=['age_factor'])
        self.cost_matrix = [[0, 1], [5, 0]]

    def _load_data(self):
        return pd.read_csv(GERMAN_PATH, sep=' ', names=COLUMN_NAMES,
                           header=None, index_col=False)

    def _preprocess(self):
        """Perform the same preprocessing as the dataset doc file."""
        self.df['credit'] = self.df['credit'].astype(str)

        for col, translation in VALUES_MAPS.items():
            self.df[col] = self.df[col].map(translation)

        new_column_names = COLUMN_NAMES[:]

        self.df['status'], self.df['sex'] = (self.df['status_sex']
                                             .str
                                             .split(' : ')
                                             .str)
        self.df = self.df.drop('status_sex', axis=1)

        status_sex_index = new_column_names.index('status_sex')
        new_column_names[status_sex_index:status_sex_index + 1] = \
            ['status', 'sex']

        self.df['age_factor'] = pd.cut(self.df['age'],
                                       [19, 25, 76],
                                       right=False)
        age_factor_index = new_column_names.index('age') + 1
        new_column_names.insert(age_factor_index, 'age_factor')

        self.df = self.df[new_column_names]

    def _validate(self):
        # pylint: disable=line-too-long
        super()._validate()

        assert len(self.df) == 1000, 'the number of rows should be 1000,'\
                                     ' but it is {}.'.format(len(self.df))
        assert len(self.df.columns) == 23, 'the number of columns should be 23,'\
                                           ' but it is {}.'.format(len(self.df.columns))
        assert not self.df.isnull().any().any(), 'there are null values.'
        assert self.df['age_factor'].nunique() == 2,\
            'age_factor should have only 2 unique values,'\
            ' but it is{}'.format(self.df['age_factor'].nunique())
