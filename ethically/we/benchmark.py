"""
Evaluate words embeedings by standard benchmarks.

Reference: https://github.com/kudkudak/word-embeddings-benchmarks


Word Pairs Tasks

1. The WordSimilarity-353 Test Collection
   http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

2. Rubenstein, H., and Goodenough, J. 1965. Contextual correlates of synonymy
   https://www.seas.upenn.edu/~hansens/conceptSim/

3. Stanford Rare Word (RW) Similarity Dataset
   https://nlp.stanford.edu/~lmthang/morphoNLM/

4. The Word Relatedness Mturk-771 Test Collection
   http://www2.mta.ac.il/~gideon/datasets/mturk_771.html

5. The MEN Test Collection
   http://clic.cimec.unitn.it/~elia.bruni/MEN.html

6. SimLex-999
   https://fh295.github.io/simlex.html

7. TR9856
   https://www.research.ibm.com/haifa/dept/vst/files/IBM_Debater_(R)_TR9856.v2.zip


Analogies Tasks

1. Google Analogies (subset of WordRep)
   https://code.google.com/archive/p/word2vec/source

2. MSR - Syntactic Analogies
   http://research.microsoft.com/en-us/projects/rnn/

"""

import os
import warnings

import pandas as pd
from pkg_resources import resource_filename


with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=FutureWarning)


WORD_PAIRS_TASKS = {'WS353': 'wordsim353.tsv',
                    'RG65': 'RG_word.tsv',
                    'RW': 'rw.tsv',
                    'Mturk': 'MTURK-771.tsv',
                    'MEN': 'MEN_dataset_natural_form_full.tsv',
                    'SimLex999': 'SimLex-999.tsv',
                    'TR9856': 'TermRelatednessResults.tsv'}

ANALOGIES_TASKS = {'MSR-syntax': 'msr-syntax.txt',
                   'Google': 'questions-words.txt'}


def get_data_resource_path(filename):
    return resource_filename(__name__, os.path.join('data',
                                                    'benchmark',
                                                    filename))


def prepare_word_pairs_file(src, dst, delimiter='\t'):
    """Transform formats of word pairs files to tsv."""
    df = pd.read_csv(src, header=None, delimiter=delimiter)
    df.loc[:, :2].to_csv(dst, sep=delimiter, index=False, header=False)


def evaluate_word_pairs(model):
    """
    Evaluate word pairs tasks.

    :param model: Words embedding.
    :return: DataFrame of evaluation results.
    """

    results = {}

    for name, filename in WORD_PAIRS_TASKS.items():
        path = get_data_resource_path(filename)
        (pearson,
         spearman,
         ratio_unknown_words) = model.evaluate_word_pairs(path)

        results[name] = {'pearson_r': pearson[0],
                         'pearson_pvalue': pearson[1],
                         'spearman_r': spearman.correlation,
                         'spearman_pvalue': spearman.pvalue,
                         'ratio_unkonwn_words': ratio_unknown_words}

    metric_names = next(iter(results.values())).keys()

    df = (pd.DataFrame(results)
          .reindex(metric_names)
          .transpose()
          .round(3))

    return df


def evaluate_word_analogies(model):
    """
    Evaluate word analogies tasks.

    :param model: Words embedding.
    :return: DataFrame of evaluation results.
    """

    results = {}

    for name, filename in ANALOGIES_TASKS.items():
        path = get_data_resource_path(filename)
        overall_score, _ = model.evaluate_word_analogies(path)

        results[name] = {'score': overall_score}

        df = (pd.DataFrame(results)
              .transpose()
              .round(3))

    return df


def evaluate_words_embedding(model):
    """
    Evaluate word pairs tasks and word analogies tasks.

    :param model: Words embedding.
    :return: Tuple of DataFrame for the evaluation results.
    """
    return evaluate_word_pairs(model), evaluate_word_analogies(model)
