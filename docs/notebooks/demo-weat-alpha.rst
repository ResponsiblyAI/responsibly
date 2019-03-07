
Demo - Word Embedding Association Test (WEAT) [Alpha Version]
=============================================================

Based on: Caliskan, A., Bryson, J. J., & Narayanan, A. (2017).
`Semantics derived automatically from language corpora contain
human-like
biases <http://www.cs.bath.ac.uk/~jjb/ftp/CaliskanEtAl-authors-full.pdf>`__.
Science, 356(6334), 183-186.

WEAT in ``ethically`` is in a alpha version, and therefore it is not yet
in the PyPI release. In order to use this coude, you should install
``ethically`` from the ``dev`` branch by:

``pip install --upgrade git+https://github.com/EthicallyAI/ethically.git@dev``

Imports
-------

.. code:: ipython3

    from ethically.we import calc_all_weat
    from ethically.we.data import load_w2v_small

For unzipping, converting and loading Glove and Word2Vec full models:

.. code:: ipython3

    import os
    import gzip
    import shutil
    from urllib.request import urlretrieve
    from zipfile import ZipFile
    
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

Word2Vec - Only Lowercase and Most Frequent Words
-------------------------------------------------

.. code:: ipython3

    model_w2v_small = load_w2v_small()

.. code:: ipython3

    calc_all_weat(model_w2v_small, filter_by='model', with_original_finding=True,
                  with_pvalue=True, pvalue_kwargs={'method': 'approximate'})




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Target words</th>
          <th>Attrib. words</th>
          <th>Nt</th>
          <th>Na</th>
          <th>s</th>
          <th>d</th>
          <th>p</th>
          <th>original_N</th>
          <th>original_d</th>
          <th>original_p</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Flowers vs. Insects</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>2x2</td>
          <td>24x2</td>
          <td>0.0949031</td>
          <td>1.23443</td>
          <td>1.6e-01</td>
          <td>32</td>
          <td>1.35</td>
          <td>1e-8</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Instruments vs. Weapons</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>16x2</td>
          <td>24x2</td>
          <td>2.11433</td>
          <td>1.58925</td>
          <td>0</td>
          <td>32</td>
          <td>1.66</td>
          <td>1e-10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>6x2</td>
          <td>24x2</td>
          <td>0.287312</td>
          <td>1.10003</td>
          <td>2.6e-02</td>
          <td>26</td>
          <td>1.17</td>
          <td>1e-5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>18x2</td>
          <td>24x2</td>
          <td>0.952434</td>
          <td>1.31962</td>
          <td>0</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>4</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>18x2</td>
          <td>8x2</td>
          <td>0.538377</td>
          <td>0.732444</td>
          <td>1.8e-02</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>5</th>
          <td>Male names vs. Female names</td>
          <td>Career vs. Family</td>
          <td>1x2</td>
          <td>8x2</td>
          <td>0.247673</td>
          <td>2</td>
          <td>0</td>
          <td>39k</td>
          <td>0.72</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Math vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>7x2</td>
          <td>8x2</td>
          <td>0.184416</td>
          <td>0.718851</td>
          <td>1.0e-01</td>
          <td>28k</td>
          <td>0.82</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Science vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>6x2</td>
          <td>8x2</td>
          <td>0.370207</td>
          <td>1.35016</td>
          <td>7.0e-03</td>
          <td>91</td>
          <td>1.47</td>
          <td>1e-24</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Mental disease vs. Physical disease</td>
          <td>Temporary vs. Permanent</td>
          <td>6x2</td>
          <td>5x2</td>
          <td>0.590304</td>
          <td>1.22442</td>
          <td>2.0e-02</td>
          <td>135</td>
          <td>1.01</td>
          <td>1e-3</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Young people’s names vs. Old people’s names</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>0x2</td>
          <td>7x2</td>
          <td></td>
          <td></td>
          <td></td>
          <td>43k</td>
          <td>1.42</td>
          <td>&lt; 1e-2</td>
        </tr>
      </tbody>
    </table>
    </div>



For the two next sections, we need the full Glove and Word2Vec words embedding, as used in the original paper. Note that it might take a while to download, extract and load these models.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Glove - Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
-------------------------------------------------------------------------------------

Download the Glove model:
http://nlp.stanford.edu/data/glove.840B.300d.zip

.. code:: ipython3

    if not os.path.exists('glove.840B.300d.w2v.txt'):
        if not os.path.exists('glove.840B.300d.txt'):
            assert os.path.exists('glove.840B.300d.zip')
            print('Unzipping...')
            with ZipFile('glove.840B.300d.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
        print('Converting to Word2Vec format...')
        glove2word2vec('glove.840B.300d.txt', 'glove.840B.300d.w2v.txt');

.. code:: ipython3

    glove_model = KeyedVectors.load_word2vec_format('glove.840B.300d.w2v.txt')

.. code:: ipython3

    calc_all_weat(glove_model, filter_by='data', with_original_finding=True,
                  with_pvalue=True, pvalue_kwargs={'method': 'approximate'})




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Target words</th>
          <th>Attrib. words</th>
          <th>Nt</th>
          <th>Na</th>
          <th>s</th>
          <th>d</th>
          <th>p</th>
          <th>original_N</th>
          <th>original_d</th>
          <th>original_p</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Flowers vs. Insects</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>25x2</td>
          <td>25x2</td>
          <td>3.87</td>
          <td>1.50</td>
          <td>0</td>
          <td>32</td>
          <td>1.35</td>
          <td>1e-8</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Instruments vs. Weapons</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>25x2</td>
          <td>25x2</td>
          <td>3.85</td>
          <td>1.52</td>
          <td>0</td>
          <td>32</td>
          <td>1.66</td>
          <td>1e-10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>32x2</td>
          <td>25x2</td>
          <td>2.92</td>
          <td>1.43</td>
          <td>0</td>
          <td>26</td>
          <td>1.17</td>
          <td>1e-5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>16x2</td>
          <td>25x2</td>
          <td>1.30</td>
          <td>1.53</td>
          <td>0</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>4</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>16x2</td>
          <td>8x2</td>
          <td>1.11</td>
          <td>1.25</td>
          <td>0</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>5</th>
          <td>Male names vs. Female names</td>
          <td>Career vs. Family</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>1.80</td>
          <td>1.87</td>
          <td>0</td>
          <td>39k</td>
          <td>0.72</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Math vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>0.23</td>
          <td>1.05</td>
          <td>2.2e-02</td>
          <td>28k</td>
          <td>0.82</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Science vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>0.40</td>
          <td>1.27</td>
          <td>2.0e-03</td>
          <td>91</td>
          <td>1.47</td>
          <td>1e-24</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Mental disease vs. Physical disease</td>
          <td>Temporary vs. Permanent</td>
          <td>6x2</td>
          <td>7x2</td>
          <td>0.90</td>
          <td>1.63</td>
          <td>0</td>
          <td>135</td>
          <td>1.01</td>
          <td>1e-3</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Young people’s names vs. Old people’s names</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>0.59</td>
          <td>1.45</td>
          <td>1.0e-03</td>
          <td>43k</td>
          <td>1.42</td>
          <td>&lt; 1e-2</td>
        </tr>
      </tbody>
    </table>
    </div>



Results from the paper: |image0|

.. |image0| image:: weat_glove.png

Word2Vec - Google News dataset (100B tokens, 3M vocab, cased, 300d vectors, 1.65GB download)
--------------------------------------------------------------------------------------------

Download the Word2Vec model: https://code.google.com/archive/p/word2vec/

.. code:: ipython3

    if not os.path.exists('GoogleNews-vectors-negative300.bin'):
        assert os.path.exists('GoogleNews-vectors-negative300.bin.gz')
        print('Unzipping...')
        with gzip.open('GoogleNews-vectors-negative300.bin.gz', 'r') as f_gz:
            with open('GoogleNews-vectors-negative300.bin', 'wb') as f_bin:
                shutil.copyfileobj(f_gz, f_bin)

.. code:: ipython3

    w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                  binary=True)

.. code:: ipython3

    calc_all_weat(w2v_model, filter_by='model', with_original_finding=True,
                  with_pvalue=True, pvalue_kwargs={'method': 'approximate'})




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Target words</th>
          <th>Attrib. words</th>
          <th>Nt</th>
          <th>Na</th>
          <th>s</th>
          <th>d</th>
          <th>p</th>
          <th>original_N</th>
          <th>original_d</th>
          <th>original_p</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Flowers vs. Insects</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>25x2</td>
          <td>25x2</td>
          <td>3.23</td>
          <td>1.55</td>
          <td>0</td>
          <td>32</td>
          <td>1.35</td>
          <td>1e-8</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Instruments vs. Weapons</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>24x2</td>
          <td>25x2</td>
          <td>3.82</td>
          <td>1.66</td>
          <td>0</td>
          <td>32</td>
          <td>1.66</td>
          <td>1e-10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>47x2</td>
          <td>25x2</td>
          <td>1.14</td>
          <td>0.55</td>
          <td>3.0e-03</td>
          <td>26</td>
          <td>1.17</td>
          <td>1e-5</td>
        </tr>
        <tr>
          <th>3</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>18x2</td>
          <td>25x2</td>
          <td>0.86</td>
          <td>1.27</td>
          <td>0</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>4</th>
          <td>European American names vs. African American n...</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>18x2</td>
          <td>8x2</td>
          <td>0.50</td>
          <td>0.71</td>
          <td>2.0e-02</td>
          <td></td>
          <td></td>
          <td></td>
        </tr>
        <tr>
          <th>5</th>
          <td>Male names vs. Female names</td>
          <td>Career vs. Family</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>2.04</td>
          <td>1.93</td>
          <td>0</td>
          <td>39k</td>
          <td>0.72</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>6</th>
          <td>Math vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>0.27</td>
          <td>0.91</td>
          <td>3.7e-02</td>
          <td>28k</td>
          <td>0.82</td>
          <td>&lt; 1e-2</td>
        </tr>
        <tr>
          <th>7</th>
          <td>Science vs. Arts</td>
          <td>Male terms vs. Female terms</td>
          <td>8x2</td>
          <td>8x2</td>
          <td>0.44</td>
          <td>1.27</td>
          <td>2.0e-03</td>
          <td>91</td>
          <td>1.47</td>
          <td>1e-24</td>
        </tr>
        <tr>
          <th>8</th>
          <td>Mental disease vs. Physical disease</td>
          <td>Temporary vs. Permanent</td>
          <td>6x2</td>
          <td>6x2</td>
          <td>0.68</td>
          <td>1.45</td>
          <td>7.0e-03</td>
          <td>135</td>
          <td>1.01</td>
          <td>1e-3</td>
        </tr>
        <tr>
          <th>9</th>
          <td>Young people’s names vs. Old people’s names</td>
          <td>Pleasant vs. Unpleasant</td>
          <td>8x2</td>
          <td>7x2</td>
          <td>-0.14</td>
          <td>-0.38</td>
          <td>7.5e-01</td>
          <td>43k</td>
          <td>1.42</td>
          <td>&lt; 1e-2</td>
        </tr>
      </tbody>
    </table>
    </div>



Results from the paper: |image0|

.. |image0| image:: weat_w2v.png

Calculate WEAT on pleasant-unpleasant attributes with chosen targets (Experimental)
-----------------------------------------------------------------------------------

.. code:: ipython3

    from ethically.we import calc_weat_pleasant_unpleasant_attribute

.. code:: ipython3

    targets = {'first_target': {'name': 'Citizen',
                                'words': ['citizen', 'citizenship', 'nationality', 'native', 'national', 'countryman', 
                                          'inhabitant', 'resident']},
              'second_target': {'name': 'Immigrant',
                                'words': ['immigrant', 'immigration', 'foreigner', 'nonnative', 'noncitizen',
                                          'relocatee', 'newcomer']}}
    calc_weat_pleasant_unpleasant_attribute(w2v_model, **targets)




.. parsed-literal::

    {'Attrib. words': 'Pleasant vs. Unpleasant',
     'Na': '25x2',
     'Nt': '6x2',
     'Target words': 'Citizen vs. Immigrant',
     'd': 0.71920586,
     'p': 0.135,
     's': 0.23210221529006958}


