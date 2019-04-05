"""
Collection of common benchmark datasets from fairness research.

Each dataset object contains a `pandas.DataFrame` as `df` attribute
that holds the actual data.
The dataset object will take care of loading, preprocessing
and validating the data.
The preprocessing is done by standard practices that are associated with
this data set: from its manual (e.g., README)
or as other did in the literature.

See :class:`ethically.dataset.Dataset`
for additional attribute and complete documentation.

Currently these are the available datasets:
    - ProPublica recidivism/COMPAS dataset,
      see: :class:`~ethically.dataset.COMPASDataset`

    - Adult dataset, see: :class:`~ethically.dataset.AdultDataset`

    - German credit dataset, see: :class:`~ethically.dataset.GermanDataset`

Usage
-----
.. code:: python

    >>> from ethically.dataset import COMPASDataset
    >>> compas_ds = COMPASDataset()
    >>> print(compas_ds)
    <ProPublica Recidivism/COMPAS Dataset. 6172 rows, 56 columns in
    which {race, sex} are sensitive attributes>
    >>> type(compas_ds.df)
    <class 'pandas.core.frame.DataFrame'>
    >>> compas_ds.df['race'].value_counts()
    African-American    3175
    Caucasian           2103
    Hispanic             509
    Other                343
    Asian                 31
    Native American       11
    Name: race, dtype: int64
"""

from ethically.dataset.adult import AdultDataset
from ethically.dataset.compas import COMPASDataset
from ethically.dataset.core import Dataset
from ethically.dataset.fico import build_FICO_dataset
from ethically.dataset.german import GermanDataset
