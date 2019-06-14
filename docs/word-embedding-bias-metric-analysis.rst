Analysis of Word Embedding Bias Metrics
=======================================

.. note::

  This page is still work-in-progress.

There are two common ways to measure bias in word embedding intrinsically,
one is given by Tolga et al. work, and the second is called WEAT.
Both of the two approaches use the same building block:
cosine similarity between two word vectors,
but it seems that they capture bias differently.
For example, after a gender debiasing of Word2Vec model
using Tolga's methods, the gender-bias which is measured with WEAT score
is not eliminated. We might hypothesize that WEAT score measures bias
in a more profound sense.

In this page, we aim to bridge the gap between the two measures.
We will formulate the WEAT score using Tolga's terminology,
and observe its power.

We assume that you are familiar with these two papers:

  - Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V.,
    & Kalai, A. T. (2016).
    `Man is to computer programmer as woman is to homemaker?
    debiasing word embeddings <https://arxiv.org/abs/1607.06520>`_.
    in Advances in neural information processing systems
    (pp. 4349-4357).

  - Caliskan, A., Bryson, J. J., & Narayanan, A. (2017).
    `Semantics derived automatically
    from language corpora contain human-like biases
    <http://opus.bath.ac.uk/55288/>`_.
    Science, 356(6334), 183-186.

Let's start with the definition of the WEAT score.
Note that we will use "word", "vector" and "word vector" interchangeably.

Let :math:`X` and :math:`Y` be two sets of target words of equal size,
and :math:`A`and :math:`B` two sets of attribute words of equal size.
Let :math:`cos(\vec a, \vec b)` donate the cosine
of the angle between vector :math:`\vec a` and :math:`\vec b`.
We will assume the word embedding is normalized,
i.e., all its vectors have a norm equal to one.
Therefore, the cosine similarity between two word vectors
is the same as the inner product of these vectors
:math:`\langle\vec a, \vec b\rangle`.

The WEAT test statistic is

.. math::

  s(X, Y, A, B)
  = \sum\limits_{\vec x \in X}{s(\vec x, A, B)} - \sum\limits_{\vec y \in X}{s(\vec y, A, B)}

where

.. math::

  s(w, A, B)
  = mean_{\vec a \in A}cos(\vec w, \vec a) - mean_{\vec b \in B}cos(\vec w, \vec b)

Let :math:`N = |A| = |B|`. Then we can rewrite :math:`s(w, A, B)`:

.. math::
  :nowrap:

  \begin{eqnarray}

  s(w, A, B) & = & mean_{\vec a \in A}cos(\vec w, \vec a) - mean_{\vec b \in B}cos(\vec w, \vec b) \\
             & = & mean_{\vec a \in A}\langle\vec w, \vec a\rangle - mean_{\vec b \in B}\langle\vec w, \vec b\rangle \\
             & = & \frac{1}{N} \sum\limits_{\vec a \in A} \langle\vec w, \vec a\rangle - \frac{1}{N} \sum\limits_{\vec b \in b} \langle\vec w, \vec b\rangle
  \end{eqnarray}

Using the linearity of the inner product:

.. math::
  :nowrap:

  \begin{eqnarray}

  & = & \frac{1}{N} \langle\vec w, \sum\limits_{\vec a \in A} \vec a\rangle - \frac{1}{N} \langle\vec w, \sum\limits_{\vec b \in b}  \vec b\rangle \\
  & = & \frac{1}{N} \langle\vec w, \sum\limits_{\vec a \in A} \vec a - \sum\limits_{\vec b \in b}  \vec b\rangle
  
  \end{eqnarray}

Let's define:

.. math::

  \vec d_{AB} = \sum\limits_{\vec a \in A} \vec a - \sum\limits_{\vec b \in b}  \vec b

And then:

.. math::

  s(w, A, B) = \frac{1}{N} \langle\vec w, \vec d_{AB}\rangle

So :math:`s(w, A, B)` measures the association between
a word :math:`\vec w` and a direction :math:`\vec d_{AB}`
which is defined by two sets of words :math:`A` and :math:`B`.
This is a key point, we formulated the low-level part of WEAT
using the notion of a direction in a word embedding.

Tolga's paper suggests three ways to come up with a direction
in a word embedding between two concepts:

1. Have two words, one for each end, :math:`\vec a` and :math:`\vec b`,
   and substruct them to get :math:`\vec d = \vec a - \vec b`.
   Then, normalize :math:`\vec d`.
   For example, :math:`\overrightarrow{she} - \overrightarrow{he}`.

2. Have two sets of words, one for each end,
   :math:`\vec A` and :math:`\vec B`,
   calculate the normalized sum of each group,
   then subtract the sums and normalized again.
   Up to a factor, this is precisely :math:`d_{AB}`!
   Nevertheless, this factor might be matter,
   as it changes for every check in the p-value calculation
   using the permutation test.
   This will be examined experimentally in the future.

3. The last method has a stronger assumption,
   it requires having a set of pairs of words,
   one from the concept :math:`A` and the other from the concept :math:`B`.
   For example, she-he and mother-father.
   We won't describe the method here.
   Note that this is the method that Tolga's paper use
   to define the gender direction for debiasing.

The first method is basically the same as the second method,
when :math:`A` and :math:`B` contain each only one word vector.

Now, let's move forward to rewrite the WEAT score itself:

.. math::
  :nowrap:

  \begin{eqnarray}

  s(X, Y, A, B) & = & \sum\limits_{\vec x \in X}{s(\vec x, A, B)} - \sum\limits_{\vec y \in X}{s(\vec y, A, B)} \\
                & = & \frac{1}{N}\sum\limits_{\vec x \in X}\langle\vec x, \vec d_{AB}\rangle - \frac{1}{N}\sum\limits_{\vec y \in Y}\langle\vec y, \vec d_{AB}\rangle \\
                & = & \frac{1}{N}\langle\sum\limits_{\vec x \in X} \vec x, \vec d_{AB}\rangle - \frac{1}{N}\langle\sum\limits_{\vec y \in Y} \vec y, \vec d_{AB}\rangle \\
                & = & \frac{1}{N}\langle\sum\limits_{\vec x \in X} \vec x - \sum\limits_{\vec y \in Y} \vec y, \vec d_{AB}\rangle \\
                & = & \frac{1}{N}\langle\vec d_{XY}, \vec d_{AB}\rangle

  \end{eqnarray}

This formulation allows us to see what the WEAT score is really about:
measuring the association between two directions.
Each direction is defined by two concepts ends,
such as Female-Male, Science-Art, Pleasent-Unpleasant.
It explains why WEAT seems like a more deeper measure of bias,
In the WEAT score, the direction is defined by two sets of words,
one for each end. As mentioned above, Tolga's paper
suggests two more methods for specifying the direction.

Note that the WEAT score is scaled only with the size of
:math:`A` and :math:`B`,
because :math:`s(X, Y, A, B)` only sums over :math:`X` and :math:`Y`
and doesn't use the mean, in contrast to :math:`s(\vec w, A, B)`.
Besides, even though the perspective of association between
two directions may help us to understand better what WEAT score measure,
the original formulation matters to compute the p-value.

Tolga's direct bias works a bit different. Given a biad direction
:math:`\vec d`
and a set of neutral words :math:`W`, then:

.. math::

  DirectBias(\vec d, W) = \frac{1}{|W|}\sum\limits_{\vec w \in W} |\langle \vec d, \vec w \rangle|

The bias direction :math:`\vec d` can be defined with
one of the three methods described above,
including the WEAT flavored one as :math:`\vec d_{AB}`
with two word sets :math:`A` and :math:`B`.
The direct bias definition lacks the second direction,
and it is indeed easier to debias, as it requires removing the
:math:`\vec d` part from all the neutral words in the vocabulary.

In Tolga's papar there is another metric - indirect bias - that takes
two words (:math:`\vec v, \vec u`) and the (bias) direction (:math:`\vec d`),
and measures the shared proportion of the two word projections
on the bias direction:

.. math::

  IndirectBias(\vec d, \vec v, \vec w) = \frac{\langle \vec d, \vec v \rangle \langle \vec d, \vec w \rangle}{\langle \vec v, \vec w \rangle}

Therefore, we can formalize the WEAT score as a measure
of association between two concept directions in a word embedding.
Practically, the WEAT score uses two sets of words to define a direction,
while in Tolga's paper, there are an additional two more methods.
