"""
Metrics and debiasing for bias (such as gender and race) in word embedding.

.. important::
    The following paper suggests that the current methods
    have an only superficial effect on the bias in word embeddings:

    Gonen, H., & Goldberg, Y. (2019).
    `Lipstick on a Pig:
    Debiasing Methods Cover up Systematic Gender Biases
    in Word Embeddings But do not Remove Them <https://arxiv.org/abs/1903.03862>`_.
    arXiv preprint arXiv:1903.03862.

.. important::
   The following paper criticize using most_similar` function
   from `gensim <https://radimrehurek.com/gensim/>`_ in the context
   of word embedding bias and the `generating analogies process:

   Nissim, M., van Noord, R., van der Goot, R. (2019).
   `Fair is Better than Sensational: Man is to Doctor
   as Woman is to Doctor <https://arxiv.org/abs/1905.09866>`_.

   Therefore, in *ethically* there is an implementation of
   :func:`~ethically.we.utils.most_similar` with the argument
   `unrestricted` that doesn't filter the results.
   Similar argument exist for
   :meth:`~ethically.we.bias.BiasWordEmbedding.generate_analogies`.

Currently, three methods are supported:

1. Bolukbasi et al. (2016) bias measure and debiasing
   - :mod:`ethically.we.bias`

2. WEAT measure
   - :mod:`ethically.we.weat`

3. Gonen et al. (2019) clustering as classification
   of biased neutral words
   - :meth:`ethically.we.bias.BiasWordEmbedding.plot_most_biased_clustering`

Besides, some of the standard benchmarks for
word embeddings are also available, primarily to check
the impact of debiasing performance.

"""

from .bias import BiasWordEmbedding, GenderBiasWE
from .data import load_w2v_small
from .utils import most_similar
from .weat import (
    calc_all_weat, calc_single_weat, calc_weat_pleasant_unpleasant_attribute,
)
