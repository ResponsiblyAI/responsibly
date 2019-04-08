Classification Fairness
=======================

.. automodule:: ethically.fairness

In the `demos section <demos.html>`_ contains two examples of measuring
the fairness of a classifier and applying intervention to adjust it:

1. `COMPAS by ProPublica <notebooks/demo-compas-analysis.html>`_

2. `FICO credit score <notebooks/demo-fico-analysis.html>`_


Metrics
-------

.. automodule:: ethically.fairness.metrics


Independence
^^^^^^^^^^^^

.. autofunction:: ethically.fairness.metrics.independence_binary

.. autofunction:: ethically.fairness.metrics.separation_binary

Separation
^^^^^^^^^^

.. autofunction:: ethically.fairness.metrics.separation_binary

.. autofunction:: ethically.fairness.metrics.separation_score

ROC
~~~

The separation criterion has strong relation to the ROC,
therefore these functions can generate ROC and ROC-AUC per
sensitive attribute values:

.. autofunction:: ethically.fairness.metrics.roc_auc_score_by_attr

.. autofunction:: ethically.fairness.metrics.roc_curve_by_attr

Plotting
~~~~~~~~

.. autofunction:: ethically.fairness.metrics.plot_roc_by_attr

.. autofunction:: ethically.fairness.metrics.plot_roc_curves

Sufficiency
^^^^^^^^^^^
.. autofunction:: ethically.fairness.metrics.sufficiency_binary

.. autofunction:: ethically.fairness.metrics.sufficiency_score

Report
^^^^^^
.. autofunction:: ethically.fairness.metrics.report_binary

A Dictionary of criteria
^^^^^^^^^^^^^^^^^^^^^^^^

https://fairmlbook.org/demographic.html#a-dictionary-of-criteria



Interventions
-------------

.. automodule:: ethically.fairness.interventions


Threshold - Post-processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: ethically.fairness.interventions.threshold
    :members:

