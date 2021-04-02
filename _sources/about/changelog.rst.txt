Revision History
================

0.1.3 (2021/04/02)
------------------

- Fix new pagacke dependencies

- Switch from Travis CI to Github Actions

0.1.2 (2020/09/15)
------------------

- Fix Travis CI issues with pipenv

- Fix bugs with word embedding bias

0.1.1 (2019/08/04)
------------------

- Fix a dependencies issue with ``smart_open``

- Change URLs to https

0.1.0 (2019/07/31)
------------------

- Rename the project to ``responsibly`` from ``ethically``

- Word embedding bias

  - Improve functionality of ``BiasWordEmbedding``

- Threshold fairness interventions

  - Fix bugs with ROCs handling
  - Improve API and add functionality (``plot_thresholds``)

0.0.5 (2019/06/14)
------------------

- Word embedding bias

  - Fix bug in computing WEAT

  - Computing and plotting factual property
    association to projections on a bias direction,
    similar to WEFAT


0.0.4 (2019/06/03)
------------------

- Word embedding bias

  - Unrestricted ``most_similar``

  - Unrestricted ``generate_analogies``

  - Running specific experiments with ``calc_all_weat``

  - Plotting clustering by classification
    of biased neutral words


0.0.3 (2019/04/10)
------------------

- Fairness in Classification

  - Three demographic fairness criteria

    - Independence
    - Separation
    - Sufficiency

  - Equalized odds post-processing algorithmic interventions
  - Complete two notebook demos (FICO and COMPAS)

- Word embedding bias

  - Measuring bias with WEAT method

- Documentation improvements

- Fixing security issues with dependencies


0.0.2 (2018/09/01)
------------------

- Word embedding bias

  - Generating analogies along the bias direction
  - Standard evaluations of word embedding (word pairs and analogies)
  - Plotting indirect bias
  - Scatter plot of bias direction projections between two word embedding
  - Improved verbose mode


0.0.1 (2018/08/17)
------------------

-  Gender debiasing for word embedding based on Bolukbasi et al.
