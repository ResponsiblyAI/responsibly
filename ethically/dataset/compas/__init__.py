__all__ = ['COMPASDataset']

import numpy as np
import pandas as pd
from pkg_resources import resource_filename

from ethically.dataset.core import Dataset


COMPAS_PATH = resource_filename(__name__,
                                'compas-scores-two-years.csv')


class COMPASDataset(Dataset):
    """ProPublica Recidivism/COMPAS Dataset.

    See :class:`~ethically.dataset.Dataset` for a description of
    the arguments and attributes.

    References:
        https://github.com/propublica/compas-analysis

    """

    def __init__(self):
        super().__init__(target='is_recid',
<<<<<<< HEAD
                         sensitive_attributes=['race', 'sex'],
                         prediction=['y_pred', 'score_factor',
                                     'score_text'])
=======
                         sensitive_attributes=['race'])
>>>>>>> dev

    def _load_data(self):
        return pd.read_csv(COMPAS_PATH)

    def _preprocess(self):
        """Perform the same preprocessing as the original analysis.

        https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        """

        self.df = self.df[(self.df['days_b_screening_arrest'] <= 30)
                          & (self.df['days_b_screening_arrest'] >= -30)
                          & (self.df['is_recid'] != -1)
                          & (self.df['c_charge_degree'] != 'O')
                          & (self.df['score_text'] != 'N/A')]

        self.df['c_jail_out'] = pd.to_datetime(self.df['c_jail_out'])
        self.df['c_jail_in'] = pd.to_datetime(self.df['c_jail_in'])
        self.df['length_of_stay'] = (self.df['c_jail_out']
                                     - self.df['c_jail_in'])

        self.df['score_factor'] = np.where(self.df['score_text']
                                           != 'Low',
                                           'HighScore', 'LowScore')
<<<<<<< HEAD
        self.df['y_pred'] = (self.df['score_factor'] == 'HighScore')
=======
>>>>>>> dev

    def _validate(self):
        # pylint: disable=line-too-long
        super()._validate()

        assert len(self.df) == 6172, 'the number of rows should be 6172,'\
                                     ' but it is {}.'.format(len(self.df))
<<<<<<< HEAD
        assert len(self.df.columns) == 56, 'the number of columns should be 56,'\
=======
        assert len(self.df.columns) == 55, 'the number of columns should be 55,'\
>>>>>>> dev
                                           ' but it is {}.'.format(len(self.df.columns))
