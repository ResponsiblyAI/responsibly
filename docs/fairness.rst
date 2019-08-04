Classification Fairness
=======================

.. automodule:: responsibly.fairness

The `demos section <demos.html>`_ contains two examples of measuring
the fairness of a classifier and applying intervention to adjust it:

1. `COMPAS by ProPublica <notebooks/demo-compas-analysis.html>`_

2. `FICO credit score <notebooks/demo-fico-analysis.html>`_


Metrics
-------

.. automodule:: responsibly.fairness.metrics


Independence
^^^^^^^^^^^^

.. autofunction:: responsibly.fairness.metrics.independence_binary

.. autofunction:: responsibly.fairness.metrics.separation_binary

Separation
^^^^^^^^^^

.. autofunction:: responsibly.fairness.metrics.separation_binary

.. autofunction:: responsibly.fairness.metrics.separation_score

ROC
~~~

The separation criterion has strong relation to the ROC,
therefore these functions can generate ROC and ROC-AUC per
sensitive attribute values:

.. autofunction:: responsibly.fairness.metrics.roc_auc_score_by_attr

.. autofunction:: responsibly.fairness.metrics.roc_curve_by_attr

Plotting
~~~~~~~~

.. autofunction:: responsibly.fairness.metrics.plot_roc_by_attr

.. autofunction:: responsibly.fairness.metrics.plot_roc_curves

Sufficiency
^^^^^^^^^^^
.. autofunction:: responsibly.fairness.metrics.sufficiency_binary

.. autofunction:: responsibly.fairness.metrics.sufficiency_score

Report
^^^^^^
.. autofunction:: responsibly.fairness.metrics.report_binary

A Dictionary of criteria
^^^^^^^^^^^^^^^^^^^^^^^^

https://fairmlbook.org/demographic.html#a-dictionary-of-criteria


Algorithmic Interventions
-------------------------

.. automodule:: responsibly.fairness.interventions


Threshold - Post-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: responsibly.fairness.interventions.threshold
    :members:

