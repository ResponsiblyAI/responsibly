import abc


class Dataset(abc.ABC):
    """Base class for datasets.

    Attributes
        - `df` - :class:`pandas.DataFrame` that holds the actual data.

        - `target` - Column name of the variable to predict
                    (ground truth)

        - `sensitive_attributes` - Column name of the
                                sensitive attributes

        - `prediction` - Columns name of the
                        prediction (optional)

    """

    @abc.abstractmethod
    def __init__(self, target, sensitive_attributes, prediction=None):
        """Load, preprocess and validate the dataset.

        :param target: Column name of the variable
                    to predict (ground truth)
        :param sensitive_attributes: Column name of the
                                    sensitive attributes
        :param prediction: Columns name of the
                           prediction (optional)
        :type target: str
        :type sensitive_attributes: list
        :type prediction: str
        """

        self.df = self._load_data()

        self._preprocess()

        self._name = self.__doc__.splitlines()[0]

        self.target = target
        self.sensitive_attributes = sensitive_attributes
        self.prediction = prediction

        self._validate()

    def __str__(self):
        return ('<{} {} rows, {} columns'
                ' in which {{{}}} are sensitive attributes>'
                .format(self._name,
                        len(self.df),
                        len(self.df.columns),
                        ', '.join(self.sensitive_attributes)))

    @abc.abstractmethod
    def _load_data(self):
        pass

    @abc.abstractmethod
    def _preprocess(self):
        pass

    @abc.abstractmethod
    def _validate(self):
        # pylint: disable=line-too-long

        assert self.target in self.df.columns,\
            ('the target label \'{}\' should be in the columns'
             .format(self.target))

        assert all(attr in self.df.columns
                   for attr in self.sensitive_attributes),\
            ('the sensitive attributes {{{}}} should be in the columns'
             .format(','.join(attr for attr in self.sensitive_attributes
                              if attr not in self.df.columns)))

        # assert all(attr in SENSITIVE_ATTRIBUTES
        #           for attr in self.sensitive_attributes),\
        # ('the sensitive attributes {} can be only from {}.'  # noqa
        #  .format(self.sensitive_attributes, SENSITIVE_ATTRIBUTES))
