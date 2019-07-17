__all__ = ['AdultDataset']

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from responsibly.dataset.core import Dataset


ADULT_TRAIN_PATH = resource_filename(__name__,
                                     'adult.data')
ADULT_TEST_PATH = resource_filename(__name__,
                                    'adult.test')

COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain',
                'capital_loss', 'hours_per_week', 'native_country',
                'income_per_year']


class AdultDataset(Dataset):
    """Adult Dataset.

    See :class:`~responsibly.dataset.Dataset` for a description of
    the arguments and attributes.

    References:
        https://archive.ics.uci.edu/ml/datasets/adult

    """

    def __init__(self):
        super().__init__(target='income_per_year',
                         sensitive_attributes=['sex', 'race'])

    def _load_data(self):
        train_df = pd.read_csv(ADULT_TRAIN_PATH, names=COLUMN_NAMES,
                               skipinitialspace=True,
                               header=None, index_col=False,
                               na_values='?')
        test_df = pd.read_csv(ADULT_TEST_PATH, names=COLUMN_NAMES,
                              skipinitialspace=True,
                              header=0, index_col=False,
                              na_values='?')

        train_df['dataset'] = 'train'
        test_df['dataset'] = 'test'

        return pd.concat([train_df, test_df], ignore_index=True)

    def _preprocess(self):
        """Perform the same preprocessing as the dataset doc file."""
        self.df = self.df.dropna()
        self.df = self.df.drop(['fnlwgt'], axis=1)
        self.df['income_per_year'] = (self.df['income_per_year']
                                      .str
                                      .replace('.', ''))

    def _validate(self):
        # pylint: disable=line-too-long
        super()._validate()

        assert len(self.df) == 45222, 'the number of rows should be 45222,'\
                                      ' but it is {}.'.format(len(self.df))
        assert len(self.df.columns) == 15, 'the number of columns should be 15,'\
                                           ' but it is {}.'.format(len(self.df.columns))

        train_df = self.df[self.df['dataset'] == 'train']
        test_df = self.df[self.df['dataset'] == 'test']
        assert len(train_df) == 30162, 'the number of train rows should be 30162,'\
                                       ' but it is {}.'.format(len(train_df))
        assert len(test_df) == 15060, 'the number of train rows should be 15060,'\
                                      ' but it is {}.'.format(len(test_df))
