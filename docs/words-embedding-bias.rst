Words Embedding Bias
====================

Bolukbasi et al. Debiasing
--------------------------

Auiding and adjusting the gender bias in words embedding
based on Bolukbasi Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. `Man is to computer programmer as woman is to homemaker? debiasing word embeddings <https://arxiv.org/abs/1607.06520>`_. NIPS 2016.

Usage
^^^^^^^
.. code:: python

   >>> from ethically.we import GenderBiasWE
   >>> from gensim import downloader
   >>> w2v_model = downloader.load('word2vec-google-news-300')
   >>> w2v_gender_bias_we = GenderBiasWE(w2v_model)
   >>> w2v_gender_bias_we.calc_direct_bias()
   0.07307904249481942
   >>> w2v_gender_bias_we.debias()
   >>> w2v_gender_bias_we.calc_direct_bias()
   1.7964246601064155e-09

Full reference: :class:`~ethically.we.bias.GenderBiasWE`

Credits
^^^^^^^
Data and part of the code from:
https://github.com/tolga-b/debiaswe
(MIT License)
