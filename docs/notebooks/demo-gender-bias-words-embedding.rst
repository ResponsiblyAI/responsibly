
Demo - Gender Bias in Words Embedding
=====================================

Based on: Bolukbasi Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh
Saligrama, and Adam T. Kalai. `Man is to computer programmer as woman is
to homemaker? debiasing word
embeddings <https://arxiv.org/abs/1607.06520>`__. NIPS 2016.

Imports
-------

.. code:: ipython3

    import matplotlib.pylab as plt
    
    from gensim import downloader
    from gensim.models import KeyedVectors
    
    from ethically.we import GenderBiasWE

Google’s Word2Vec
-----------------

Download and load word2vec full model (it might take few minutes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_path = downloader.load('word2vec-google-news-300', return_path=True)
    print(w2v_path)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)


.. parsed-literal::

    /home/users/user/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz


Create gender bias words embedding object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_bias_we = GenderBiasWE(w2v_model, only_lower=False, verbose=True)


.. parsed-literal::

    Identify direction using pca method...
      Principal Component    Explained Variance Ratio
    ---------------------  --------------------------
                        1                  0.605292
                        2                  0.127255
                        3                  0.099281
                        4                  0.0483466
                        5                  0.0406355
                        6                  0.0252729
                        7                  0.0232224
                        8                  0.0123879
                        9                  0.00996098
                       10                  0.00834613


Evaluate the Words Embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_biased_evaluation = w2v_gender_bias_we.evaluate_words_embedding()

Word pairs
^^^^^^^^^^

.. code:: ipython3

    w2v_biased_evaluation[0]




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
          <th>pearson_r</th>
          <th>pearson_pvalue</th>
          <th>spearman_r</th>
          <th>spearman_pvalue</th>
          <th>ratio_unkonwn_words</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MEN</th>
          <td>0.682</td>
          <td>0.00</td>
          <td>0.699</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>Mturk</th>
          <td>0.632</td>
          <td>0.00</td>
          <td>0.656</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RG65</th>
          <td>0.801</td>
          <td>0.03</td>
          <td>0.685</td>
          <td>0.09</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RW</th>
          <td>0.523</td>
          <td>0.00</td>
          <td>0.553</td>
          <td>0.00</td>
          <td>33.727</td>
        </tr>
        <tr>
          <th>SimLex999</th>
          <td>0.447</td>
          <td>0.00</td>
          <td>0.436</td>
          <td>0.00</td>
          <td>0.100</td>
        </tr>
        <tr>
          <th>TR9856</th>
          <td>0.661</td>
          <td>0.00</td>
          <td>0.662</td>
          <td>0.00</td>
          <td>85.430</td>
        </tr>
        <tr>
          <th>WS353</th>
          <td>0.624</td>
          <td>0.00</td>
          <td>0.659</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
      </tbody>
    </table>
    </div>



Analogies
^^^^^^^^^

.. code:: ipython3

    w2v_biased_evaluation[1]




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
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Google</th>
          <td>0.740</td>
        </tr>
        <tr>
          <th>MSR-syntax</th>
          <td>0.736</td>
        </tr>
      </tbody>
    </table>
    </div>



Calculate direct gender bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_bias_we.calc_direct_bias()




.. parsed-literal::

    0.0730790424948194



Plot the projection of the most extreme professions on the gender direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_bias_we.plot_projection_scores();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_17_0.png


Plot the distribution of projections of the word groups that are being used for the auditing and adjusting the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **profession_name** - List of profession names, neutral and gender
   spcific.
2. **neutral_profession_name** - List of only neutral profession names.
3. **specific_seed** - Seed list of gender specific words.
4. **specific_full** - List of the learned specifc gender over all the
   vocabulary.
5. **specific_full_with_definitional** - **specific_full** with the
   words that were used to define the gender direction.
6. **neutral_words** - List of all the words in the vocabulary that are
   not part of **specific_full_with_definitional**.

.. code:: ipython3

    w2v_gender_bias_we.plot_dist_projections_on_direction();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_19_0.png


Generate analogies along the gender direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Skipping the first 50, because they are mostly appropriate gender
analogies

.. code:: ipython3

    w2v_gender_bias_we.generate_analogies(150)[50:]




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
          <th>x</th>
          <th>y</th>
          <th>distance</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>50</th>
          <td>diva</td>
          <td>superstar</td>
          <td>0.912041</td>
          <td>0.478637</td>
        </tr>
        <tr>
          <th>51</th>
          <td>Kylie</td>
          <td>Robbie</td>
          <td>0.965910</td>
          <td>0.469355</td>
        </tr>
        <tr>
          <th>52</th>
          <td>Ana</td>
          <td>Sergio</td>
          <td>0.932191</td>
          <td>0.459715</td>
        </tr>
        <tr>
          <th>53</th>
          <td>Lady_Vols</td>
          <td>Vols</td>
          <td>0.667961</td>
          <td>0.455486</td>
        </tr>
        <tr>
          <th>54</th>
          <td>Gloria</td>
          <td>Ernie</td>
          <td>0.987297</td>
          <td>0.455313</td>
        </tr>
        <tr>
          <th>55</th>
          <td>Susie</td>
          <td>Johnny</td>
          <td>0.982425</td>
          <td>0.451297</td>
        </tr>
        <tr>
          <th>56</th>
          <td>Veronica</td>
          <td>Dominic</td>
          <td>0.991988</td>
          <td>0.450322</td>
        </tr>
        <tr>
          <th>57</th>
          <td>Mother_Day</td>
          <td>Father_Day</td>
          <td>0.696741</td>
          <td>0.450317</td>
        </tr>
        <tr>
          <th>58</th>
          <td>waitress</td>
          <td>waiter</td>
          <td>0.695489</td>
          <td>0.449741</td>
        </tr>
        <tr>
          <th>59</th>
          <td>LPGA_Tour</td>
          <td>PGA_Tour</td>
          <td>0.654526</td>
          <td>0.445583</td>
        </tr>
        <tr>
          <th>60</th>
          <td>Sorenstam</td>
          <td>Jack_Nicklaus</td>
          <td>0.946246</td>
          <td>0.443112</td>
        </tr>
        <tr>
          <th>61</th>
          <td>softball</td>
          <td>baseball</td>
          <td>0.762902</td>
          <td>0.442747</td>
        </tr>
        <tr>
          <th>62</th>
          <td>mare</td>
          <td>gelding</td>
          <td>0.510441</td>
          <td>0.441356</td>
        </tr>
        <tr>
          <th>63</th>
          <td>filly</td>
          <td>colt</td>
          <td>0.475118</td>
          <td>0.441043</td>
        </tr>
        <tr>
          <th>64</th>
          <td>LPGA</td>
          <td>PGA</td>
          <td>0.834602</td>
          <td>0.440244</td>
        </tr>
        <tr>
          <th>65</th>
          <td>Princess</td>
          <td>Prince</td>
          <td>0.994014</td>
          <td>0.435972</td>
        </tr>
        <tr>
          <th>66</th>
          <td>volleyball</td>
          <td>football</td>
          <td>0.995761</td>
          <td>0.432789</td>
        </tr>
        <tr>
          <th>67</th>
          <td>Jackie</td>
          <td>Jimmy</td>
          <td>0.950708</td>
          <td>0.432025</td>
        </tr>
        <tr>
          <th>68</th>
          <td>girlfriends</td>
          <td>buddies</td>
          <td>0.915496</td>
          <td>0.431573</td>
        </tr>
        <tr>
          <th>69</th>
          <td>Louise</td>
          <td>Charles</td>
          <td>0.990803</td>
          <td>0.428367</td>
        </tr>
        <tr>
          <th>70</th>
          <td>hair</td>
          <td>beard</td>
          <td>0.895302</td>
          <td>0.426211</td>
        </tr>
        <tr>
          <th>71</th>
          <td>WTA</td>
          <td>ATP</td>
          <td>0.782440</td>
          <td>0.421674</td>
        </tr>
        <tr>
          <th>72</th>
          <td>Daughter</td>
          <td>Son</td>
          <td>0.937689</td>
          <td>0.417058</td>
        </tr>
        <tr>
          <th>73</th>
          <td>ladies</td>
          <td>gentlemen</td>
          <td>0.907104</td>
          <td>0.414876</td>
        </tr>
        <tr>
          <th>74</th>
          <td>I'ma</td>
          <td>he'sa</td>
          <td>0.664386</td>
          <td>0.413903</td>
        </tr>
        <tr>
          <th>75</th>
          <td>bra</td>
          <td>pants</td>
          <td>0.936291</td>
          <td>0.412608</td>
        </tr>
        <tr>
          <th>76</th>
          <td>soprano</td>
          <td>tenor</td>
          <td>0.844273</td>
          <td>0.408688</td>
        </tr>
        <tr>
          <th>77</th>
          <td>Girl_Scouts</td>
          <td>Scouts</td>
          <td>0.990917</td>
          <td>0.406515</td>
        </tr>
        <tr>
          <th>78</th>
          <td>Isabel</td>
          <td>Ivan</td>
          <td>0.935470</td>
          <td>0.402371</td>
        </tr>
        <tr>
          <th>79</th>
          <td>nun</td>
          <td>priest</td>
          <td>0.775789</td>
          <td>0.398276</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>120</th>
          <td>Ochoa</td>
          <td>Garcia</td>
          <td>0.923490</td>
          <td>0.306437</td>
        </tr>
        <tr>
          <th>121</th>
          <td>handbag</td>
          <td>wallet</td>
          <td>0.915948</td>
          <td>0.306354</td>
        </tr>
        <tr>
          <th>122</th>
          <td>Jennifer_Aniston</td>
          <td>George_Clooney</td>
          <td>0.900427</td>
          <td>0.303838</td>
        </tr>
        <tr>
          <th>123</th>
          <td>Palin</td>
          <td>McCain</td>
          <td>0.851226</td>
          <td>0.300329</td>
        </tr>
        <tr>
          <th>124</th>
          <td>M.</td>
          <td>D.</td>
          <td>0.559084</td>
          <td>0.297112</td>
        </tr>
        <tr>
          <th>125</th>
          <td>Abdul</td>
          <td>Omar</td>
          <td>0.942199</td>
          <td>0.296321</td>
        </tr>
        <tr>
          <th>126</th>
          <td>Oprah</td>
          <td>Rush_Limbaugh</td>
          <td>0.994474</td>
          <td>0.295990</td>
        </tr>
        <tr>
          <th>127</th>
          <td>dress</td>
          <td>wearing</td>
          <td>0.965128</td>
          <td>0.295465</td>
        </tr>
        <tr>
          <th>128</th>
          <td>Lindsay_Lohan</td>
          <td>Mel_Gibson</td>
          <td>0.897883</td>
          <td>0.290938</td>
        </tr>
        <tr>
          <th>129</th>
          <td>Wendy</td>
          <td>Denny</td>
          <td>0.955523</td>
          <td>0.289228</td>
        </tr>
        <tr>
          <th>130</th>
          <td>Hillary</td>
          <td>Kerry</td>
          <td>0.937335</td>
          <td>0.288849</td>
        </tr>
        <tr>
          <th>131</th>
          <td>Bhutto</td>
          <td>Mehsud</td>
          <td>0.951625</td>
          <td>0.288000</td>
        </tr>
        <tr>
          <th>132</th>
          <td>captivating</td>
          <td>masterful</td>
          <td>0.972531</td>
          <td>0.286297</td>
        </tr>
        <tr>
          <th>133</th>
          <td>bride</td>
          <td>groom</td>
          <td>0.805207</td>
          <td>0.285968</td>
        </tr>
        <tr>
          <th>134</th>
          <td>Coulter</td>
          <td>Dobson</td>
          <td>0.987666</td>
          <td>0.282488</td>
        </tr>
        <tr>
          <th>135</th>
          <td>Fergie</td>
          <td>Sir_Alex_Ferguson</td>
          <td>0.821957</td>
          <td>0.281196</td>
        </tr>
        <tr>
          <th>136</th>
          <td>designer</td>
          <td>architect</td>
          <td>0.998560</td>
          <td>0.280551</td>
        </tr>
        <tr>
          <th>137</th>
          <td>Dee</td>
          <td>Kenny</td>
          <td>0.967734</td>
          <td>0.277553</td>
        </tr>
        <tr>
          <th>138</th>
          <td>midfielder</td>
          <td>winger</td>
          <td>0.739950</td>
          <td>0.277374</td>
        </tr>
        <tr>
          <th>139</th>
          <td>hysterical</td>
          <td>comical</td>
          <td>0.917530</td>
          <td>0.277098</td>
        </tr>
        <tr>
          <th>140</th>
          <td>charming</td>
          <td>charismatic</td>
          <td>0.966619</td>
          <td>0.276466</td>
        </tr>
        <tr>
          <th>141</th>
          <td>Freshman</td>
          <td>Rookie</td>
          <td>0.860134</td>
          <td>0.275848</td>
        </tr>
        <tr>
          <th>142</th>
          <td>ultrasound</td>
          <td>MRI</td>
          <td>0.884500</td>
          <td>0.273803</td>
        </tr>
        <tr>
          <th>143</th>
          <td>servicemen</td>
          <td>veterans</td>
          <td>0.918718</td>
          <td>0.273633</td>
        </tr>
        <tr>
          <th>144</th>
          <td>cigarette</td>
          <td>cigar</td>
          <td>0.874594</td>
          <td>0.269909</td>
        </tr>
        <tr>
          <th>145</th>
          <td>Rae</td>
          <td>Campbell</td>
          <td>0.981424</td>
          <td>0.269635</td>
        </tr>
        <tr>
          <th>146</th>
          <td>backcourt</td>
          <td>playmaker</td>
          <td>0.975412</td>
          <td>0.269148</td>
        </tr>
        <tr>
          <th>147</th>
          <td>choreography</td>
          <td>footwork</td>
          <td>0.995236</td>
          <td>0.268480</td>
        </tr>
        <tr>
          <th>148</th>
          <td>Tony_Parker</td>
          <td>Tim_Duncan</td>
          <td>0.762779</td>
          <td>0.268100</td>
        </tr>
        <tr>
          <th>149</th>
          <td>kindness</td>
          <td>humility</td>
          <td>0.943152</td>
          <td>0.267360</td>
        </tr>
      </tbody>
    </table>
    <p>100 rows × 4 columns</p>
    </div>



Generate the Indirect Gender Bias in the direction ``softball``-``football``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_bias_we.generate_closest_words_indirect_bias('softball', 'football')




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
          <th></th>
          <th>projection</th>
          <th>indirect_bias</th>
        </tr>
        <tr>
          <th>end</th>
          <th>word</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">softball</th>
          <th>bookkeeper</th>
          <td>0.195044</td>
          <td>0.201158</td>
        </tr>
        <tr>
          <th>registered_nurse</th>
          <td>0.176933</td>
          <td>0.287150</td>
        </tr>
        <tr>
          <th>infielder</th>
          <td>0.171764</td>
          <td>-0.054049</td>
        </tr>
        <tr>
          <th>waitress</th>
          <td>0.163246</td>
          <td>0.317842</td>
        </tr>
        <tr>
          <th>receptionist</th>
          <td>0.159252</td>
          <td>0.672343</td>
        </tr>
        <tr>
          <th rowspan="5" valign="top">football</th>
          <th>philosopher</th>
          <td>-0.148482</td>
          <td>0.220857</td>
        </tr>
        <tr>
          <th>pundit</th>
          <td>-0.170339</td>
          <td>0.101227</td>
        </tr>
        <tr>
          <th>businessman</th>
          <td>-0.174114</td>
          <td>0.170078</td>
        </tr>
        <tr>
          <th>maestro</th>
          <td>-0.175094</td>
          <td>0.415804</td>
        </tr>
        <tr>
          <th>footballer</th>
          <td>-0.275374</td>
          <td>0.015366</td>
        </tr>
      </tbody>
    </table>
    </div>



Preform hard-debiasing
~~~~~~~~~~~~~~~~~~~~~~

The table shows the details of the equalize step on the equality sets.

.. code:: ipython3

    w2v_gender_debias_we = w2v_gender_bias_we.debias('hard', inplace=False)


.. parsed-literal::

    Neutralize...


.. parsed-literal::

    100%|██████████| 2997984/2997984 [03:28<00:00, 14359.90it/s]


.. parsed-literal::

    Equalize...
    Equalize Words Data (all equal for 1-dim bias space (direction):
                        equalized_projected_scalar    projected_scalar    scaling
    ----------------  ----------------------------  ------------------  ---------
    (0, 'she')                            0.443113           0.469059    0.443113
    (0, 'he')                            -0.443113          -0.362353    0.443113
    (1, 'Her')                            0.272142           0.267272    0.272142
    (1, 'His')                           -0.272142          -0.122555    0.272142
    (2, 'SHE')                            0.540225           0.385345    0.540225
    (2, 'HE')                            -0.540225          -0.120598    0.540225
    (3, 'Daughter')                       0.469635           0.22278     0.469635
    (3, 'Son')                           -0.469635          -0.16829     0.469635
    (4, 'her')                            0.430368           0.446157    0.430368
    (4, 'his')                           -0.430368          -0.333555    0.430368
    (5, 'HERSELF')                        0.388876           0.247386    0.388876
    (5, 'HIMSELF')                       -0.388876          -0.148449    0.388876
    (6, 'female')                         0.336739           0.282941    0.336739
    (6, 'male')                          -0.336739           0.083992    0.336739
    (7, 'MOTHER')                         0.391597           0.235139    0.391597
    (7, 'FATHER')                        -0.391597          -0.0389424   0.391597
    (8, 'Mary')                           0.488704           0.301974    0.488704
    (8, 'John')                          -0.488704          -0.251126    0.488704
    (9, 'FEMALE')                         0.483523           0.179938    0.483523
    (9, 'MALE')                          -0.483523           0.0635581   0.483523
    (10, 'woman')                         0.346934           0.340348    0.346934
    (10, 'man')                          -0.346934          -0.220952    0.346934
    (11, 'HER')                           0.509812           0.267332    0.509812
    (11, 'HIS')                          -0.509812          -0.0502074   0.509812
    (12, 'Mother')                        0.498544           0.312446    0.498544
    (12, 'Father')                       -0.498544          -0.197701    0.498544
    (13, 'She')                           0.316178           0.330259    0.316178
    (13, 'He')                           -0.316178          -0.178255    0.316178
    (14, 'WOMAN')                         0.416802           0.257037    0.416802
    (14, 'MAN')                          -0.416802          -0.0911706   0.416802
    (15, 'Gal')                           0.59801            0.110373    0.59801
    (15, 'Guy')                          -0.59801           -0.137855    0.59801
    (16, 'GIRL')                          0.395733           0.242016    0.395733
    (16, 'BOY')                          -0.395733          -0.0573038   0.395733
    (17, 'Herself')                       0.479517           0.29538     0.479517
    (17, 'Himself')                      -0.479517          -0.220471    0.479517
    (18, 'Woman')                         0.392149           0.238867    0.392149
    (18, 'Man')                          -0.392149          -0.184176    0.392149
    (19, 'Girl')                          0.398915           0.251796    0.398915
    (19, 'Boy')                          -0.398915          -0.0954833   0.398915
    (20, 'DAUGHTER')                      0.521962           0.171916    0.521962
    (20, 'SON')                          -0.521962           0.0122388   0.521962
    (21, 'mary')                          0.444525           0.192964    0.444525
    (21, 'john')                         -0.444525          -0.0204266   0.444525
    (22, 'MARY')                          0.356334           0.222307    0.356334
    (22, 'JOHN')                         -0.356334          -0.0818515   0.356334
    (23, 'mother')                        0.332768           0.300389    0.332768
    (23, 'father')                       -0.332768          -0.147961    0.332768
    (24, 'gal')                           0.51301            0.400741    0.51301
    (24, 'guy')                          -0.51301           -0.326011    0.51301
    (25, 'GAL')                           0.596187           0.185162    0.596187
    (25, 'GUY')                          -0.596187          -0.0558901   0.596187
    (26, 'Female')                        0.405189           0.198377    0.405189
    (26, 'Male')                         -0.405189           0.0366158   0.405189
    (27, 'herself')                       0.401098           0.378141    0.401098
    (27, 'himself')                      -0.401098          -0.38296     0.401098
    (28, 'daughter')                      0.289697           0.292953    0.289697
    (28, 'son')                          -0.289697          -0.121614    0.289697
    (29, 'girl')                          0.29452            0.318458    0.29452
    (29, 'boy')                          -0.29452           -0.0826128   0.29452


Now our model is gender debiased, let’s check what changed…
-----------------------------------------------------------

Evaluate the debiased model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The evaluation of the words embedding did not change so much because of
the debiasing:

.. code:: ipython3

    w2v_debiased_evaluation = w2v_gender_debias_we.evaluate_words_embedding()

.. code:: ipython3

    w2v_debiased_evaluation[0]




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
          <th>pearson_r</th>
          <th>pearson_pvalue</th>
          <th>spearman_r</th>
          <th>spearman_pvalue</th>
          <th>ratio_unkonwn_words</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MEN</th>
          <td>0.680</td>
          <td>0.000</td>
          <td>0.698</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>Mturk</th>
          <td>0.633</td>
          <td>0.000</td>
          <td>0.656</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RG65</th>
          <td>0.800</td>
          <td>0.031</td>
          <td>0.685</td>
          <td>0.09</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RW</th>
          <td>0.522</td>
          <td>0.000</td>
          <td>0.552</td>
          <td>0.00</td>
          <td>33.727</td>
        </tr>
        <tr>
          <th>SimLex999</th>
          <td>0.450</td>
          <td>0.000</td>
          <td>0.438</td>
          <td>0.00</td>
          <td>0.100</td>
        </tr>
        <tr>
          <th>TR9856</th>
          <td>0.661</td>
          <td>0.000</td>
          <td>0.662</td>
          <td>0.00</td>
          <td>85.430</td>
        </tr>
        <tr>
          <th>WS353</th>
          <td>0.623</td>
          <td>0.000</td>
          <td>0.657</td>
          <td>0.00</td>
          <td>0.000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    w2v_debiased_evaluation[1]




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
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Google</th>
          <td>0.737</td>
        </tr>
        <tr>
          <th>MSR-syntax</th>
          <td>0.736</td>
        </tr>
      </tbody>
    </table>
    </div>



Calculate direct gender bias
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_debias_we.calc_direct_bias()




.. parsed-literal::

    1.7964246601064155e-09



The words embedding is not biased any more (in the professions sense).

Plot the projection of the most extreme professions on the gender direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that (almost) all of the non-zero projection words are gender
specific.

The word *teenager* have a projection on the gender direction because it
was learned mistakenly as a gender-specific word by the linear SVM, and
thus it was not neutralized in the debias processes.

The words provost, serviceman and librarian have zero projection on the
gender direction.

.. code:: ipython3

    w2v_gender_debias_we.plot_projection_scores();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_36_0.png


Generate analogies along the gender direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_debias_we.generate_analogies(150)[50:]




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
          <th>x</th>
          <th>y</th>
          <th>distance</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>50</th>
          <td>estrogen</td>
          <td>hormone</td>
          <td>0.696338</td>
          <td>3.415194e-01</td>
        </tr>
        <tr>
          <th>51</th>
          <td>brides</td>
          <td>couples</td>
          <td>0.972528</td>
          <td>3.321108e-01</td>
        </tr>
        <tr>
          <th>52</th>
          <td>breasts</td>
          <td>penis</td>
          <td>0.831993</td>
          <td>3.135472e-01</td>
        </tr>
        <tr>
          <th>53</th>
          <td>compatriots</td>
          <td>countrymen</td>
          <td>0.668135</td>
          <td>2.980544e-01</td>
        </tr>
        <tr>
          <th>54</th>
          <td>Moms</td>
          <td>Kids</td>
          <td>0.956169</td>
          <td>2.873899e-01</td>
        </tr>
        <tr>
          <th>55</th>
          <td>Mothers</td>
          <td>Families</td>
          <td>0.992906</td>
          <td>2.803815e-01</td>
        </tr>
        <tr>
          <th>56</th>
          <td>maid</td>
          <td>prostitute</td>
          <td>0.996605</td>
          <td>2.730137e-01</td>
        </tr>
        <tr>
          <th>57</th>
          <td>entrepreneur</td>
          <td>businessman</td>
          <td>0.812313</td>
          <td>2.690392e-01</td>
        </tr>
        <tr>
          <th>58</th>
          <td>menopause</td>
          <td>osteoporosis</td>
          <td>0.916623</td>
          <td>2.671666e-01</td>
        </tr>
        <tr>
          <th>59</th>
          <td>maternal</td>
          <td>reproductive</td>
          <td>0.995438</td>
          <td>2.265359e-01</td>
        </tr>
        <tr>
          <th>60</th>
          <td>clan</td>
          <td>patriarch</td>
          <td>0.993231</td>
          <td>2.216309e-01</td>
        </tr>
        <tr>
          <th>61</th>
          <td>girlfriends</td>
          <td>buddies</td>
          <td>0.855100</td>
          <td>2.116219e-01</td>
        </tr>
        <tr>
          <th>62</th>
          <td>womb</td>
          <td>fetus</td>
          <td>0.843194</td>
          <td>1.891709e-01</td>
        </tr>
        <tr>
          <th>63</th>
          <td>Chicago_Bulls</td>
          <td>Bulls</td>
          <td>0.796379</td>
          <td>1.862223e-01</td>
        </tr>
        <tr>
          <th>64</th>
          <td>Pizza</td>
          <td>Papa</td>
          <td>0.982437</td>
          <td>1.739706e-01</td>
        </tr>
        <tr>
          <th>65</th>
          <td>counterparts</td>
          <td>brethren</td>
          <td>0.907441</td>
          <td>1.688463e-01</td>
        </tr>
        <tr>
          <th>66</th>
          <td>God</td>
          <td>Him</td>
          <td>0.768831</td>
          <td>1.635438e-01</td>
        </tr>
        <tr>
          <th>67</th>
          <td>mechanic</td>
          <td>salesman</td>
          <td>0.974865</td>
          <td>1.611643e-01</td>
        </tr>
        <tr>
          <th>68</th>
          <td>Rep.</td>
          <td>Congressman</td>
          <td>0.606849</td>
          <td>1.563420e-01</td>
        </tr>
        <tr>
          <th>69</th>
          <td>replied</td>
          <td>sir</td>
          <td>0.924511</td>
          <td>1.454485e-01</td>
        </tr>
        <tr>
          <th>70</th>
          <td>entrepreneurs</td>
          <td>businessmen</td>
          <td>0.870914</td>
          <td>1.430111e-01</td>
        </tr>
        <tr>
          <th>71</th>
          <td>muscle</td>
          <td>muscular</td>
          <td>0.879146</td>
          <td>1.414681e-01</td>
        </tr>
        <tr>
          <th>72</th>
          <td>bulls</td>
          <td>bull</td>
          <td>0.675282</td>
          <td>1.286088e-01</td>
        </tr>
        <tr>
          <th>73</th>
          <td>Bachelor</td>
          <td>bachelor_degree</td>
          <td>0.818799</td>
          <td>1.275013e-01</td>
        </tr>
        <tr>
          <th>74</th>
          <td>Gardenhire</td>
          <td>Leyland</td>
          <td>0.867532</td>
          <td>1.221324e-01</td>
        </tr>
        <tr>
          <th>75</th>
          <td>heirs</td>
          <td>heir</td>
          <td>0.950050</td>
          <td>1.147512e-01</td>
        </tr>
        <tr>
          <th>76</th>
          <td>Carl</td>
          <td>Earl</td>
          <td>0.907354</td>
          <td>1.089646e-01</td>
        </tr>
        <tr>
          <th>77</th>
          <td>Twins</td>
          <td>Minnesota_Twins</td>
          <td>0.569007</td>
          <td>1.061712e-01</td>
        </tr>
        <tr>
          <th>78</th>
          <td>Muslim_Brotherhood</td>
          <td>Brotherhood</td>
          <td>0.768524</td>
          <td>1.052288e-01</td>
        </tr>
        <tr>
          <th>79</th>
          <td>Sons</td>
          <td>Brothers</td>
          <td>0.974409</td>
          <td>1.051271e-01</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>120</th>
          <td>miles_northeast</td>
          <td>miles_southeast</td>
          <td>0.372832</td>
          <td>2.040777e-08</td>
        </tr>
        <tr>
          <th>121</th>
          <td>Ben_Bernanke</td>
          <td>Bernanke</td>
          <td>0.499767</td>
          <td>2.040426e-08</td>
        </tr>
        <tr>
          <th>122</th>
          <td>tumbled</td>
          <td>slumped</td>
          <td>0.593244</td>
          <td>2.029145e-08</td>
        </tr>
        <tr>
          <th>123</th>
          <td>#.#B</td>
          <td>##M</td>
          <td>0.619619</td>
          <td>2.008235e-08</td>
        </tr>
        <tr>
          <th>124</th>
          <td>#:##-#_pm</td>
          <td>#-#:##_pm</td>
          <td>0.275680</td>
          <td>2.004407e-08</td>
        </tr>
        <tr>
          <th>125</th>
          <td>riders</td>
          <td>cyclists</td>
          <td>0.682553</td>
          <td>1.942173e-08</td>
        </tr>
        <tr>
          <th>126</th>
          <td>Q3</td>
          <td>Q4</td>
          <td>0.374242</td>
          <td>1.879385e-08</td>
        </tr>
        <tr>
          <th>127</th>
          <td>Selectmen</td>
          <td>selectmen</td>
          <td>0.334434</td>
          <td>1.865442e-08</td>
        </tr>
        <tr>
          <th>128</th>
          <td>delicious</td>
          <td>tasty</td>
          <td>0.499245</td>
          <td>1.835767e-08</td>
        </tr>
        <tr>
          <th>129</th>
          <td>Mr</td>
          <td>Mrs</td>
          <td>0.466663</td>
          <td>1.832289e-08</td>
        </tr>
        <tr>
          <th>130</th>
          <td>Mauresmo</td>
          <td>Kuznetsova</td>
          <td>0.482997</td>
          <td>1.788800e-08</td>
        </tr>
        <tr>
          <th>131</th>
          <td>PCs</td>
          <td>desktops</td>
          <td>0.639124</td>
          <td>1.786450e-08</td>
        </tr>
        <tr>
          <th>132</th>
          <td>AME_Info_FZ_LLC</td>
          <td>Emap_Limited</td>
          <td>0.405051</td>
          <td>1.752216e-08</td>
        </tr>
        <tr>
          <th>133</th>
          <td>missile</td>
          <td>ballistic_missile</td>
          <td>0.541649</td>
          <td>1.738411e-08</td>
        </tr>
        <tr>
          <th>134</th>
          <td>medication</td>
          <td>medications</td>
          <td>0.497686</td>
          <td>1.718281e-08</td>
        </tr>
        <tr>
          <th>135</th>
          <td>Maple_Leafs</td>
          <td>Leafs</td>
          <td>0.382509</td>
          <td>1.705314e-08</td>
        </tr>
        <tr>
          <th>136</th>
          <td>RM###</td>
          <td>RM##</td>
          <td>0.302811</td>
          <td>1.705186e-08</td>
        </tr>
        <tr>
          <th>137</th>
          <td>Full_Page_Reprints</td>
          <td>taxes_suitable</td>
          <td>0.550853</td>
          <td>1.701411e-08</td>
        </tr>
        <tr>
          <th>138</th>
          <td>onions</td>
          <td>onion</td>
          <td>0.536725</td>
          <td>1.701317e-08</td>
        </tr>
        <tr>
          <th>139</th>
          <td>Jharkhand</td>
          <td>Chhattisgarh</td>
          <td>0.490719</td>
          <td>1.700521e-08</td>
        </tr>
        <tr>
          <th>140</th>
          <td>Palestinians</td>
          <td>Palestinian</td>
          <td>0.506989</td>
          <td>1.695730e-08</td>
        </tr>
        <tr>
          <th>141</th>
          <td>cheered</td>
          <td>hugged</td>
          <td>0.873946</td>
          <td>1.694949e-08</td>
        </tr>
        <tr>
          <th>142</th>
          <td>strewn</td>
          <td>littered</td>
          <td>0.644268</td>
          <td>1.692168e-08</td>
        </tr>
        <tr>
          <th>143</th>
          <td>closet</td>
          <td>bathroom</td>
          <td>0.938251</td>
          <td>1.681724e-08</td>
        </tr>
        <tr>
          <th>144</th>
          <td>suburbs</td>
          <td>suburb</td>
          <td>0.743467</td>
          <td>1.678293e-08</td>
        </tr>
        <tr>
          <th>145</th>
          <td>southwestern</td>
          <td>southern</td>
          <td>0.583812</td>
          <td>1.662719e-08</td>
        </tr>
        <tr>
          <th>146</th>
          <td>demonstrates</td>
          <td>recognizes</td>
          <td>0.882648</td>
          <td>1.653202e-08</td>
        </tr>
        <tr>
          <th>147</th>
          <td>Risk_Factors</td>
          <td>SEC_filings</td>
          <td>0.982240</td>
          <td>1.644299e-08</td>
        </tr>
        <tr>
          <th>148</th>
          <td>Aggies</td>
          <td>Longhorns</td>
          <td>0.618360</td>
          <td>1.643619e-08</td>
        </tr>
        <tr>
          <th>149</th>
          <td>Moroccan</td>
          <td>Morocco</td>
          <td>0.806024</td>
          <td>1.639907e-08</td>
        </tr>
      </tbody>
    </table>
    <p>100 rows × 4 columns</p>
    </div>



Generate the Indirect Gender Bias in the direction ``softball``-``football``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    w2v_gender_debias_we.generate_closest_words_indirect_bias('softball', 'football')




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
          <th></th>
          <th>projection</th>
          <th>indirect_bias</th>
        </tr>
        <tr>
          <th>end</th>
          <th>word</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">softball</th>
          <th>infielder</th>
          <td>0.149894</td>
          <td>1.008288e-07</td>
        </tr>
        <tr>
          <th>major_leaguer</th>
          <td>0.113700</td>
          <td>5.297945e-09</td>
        </tr>
        <tr>
          <th>bookkeeper</th>
          <td>0.104209</td>
          <td>3.627948e-08</td>
        </tr>
        <tr>
          <th>patrolman</th>
          <td>0.092638</td>
          <td>1.222811e-07</td>
        </tr>
        <tr>
          <th>investigator</th>
          <td>0.081746</td>
          <td>-1.263093e-09</td>
        </tr>
        <tr>
          <th rowspan="5" valign="top">football</th>
          <th>midfielder</th>
          <td>-0.153175</td>
          <td>2.186535e-08</td>
        </tr>
        <tr>
          <th>lecturer</th>
          <td>-0.153629</td>
          <td>1.659876e-08</td>
        </tr>
        <tr>
          <th>vice_chancellor</th>
          <td>-0.159645</td>
          <td>1.804495e-07</td>
        </tr>
        <tr>
          <th>cleric</th>
          <td>-0.166934</td>
          <td>-3.282093e-08</td>
        </tr>
        <tr>
          <th>footballer</th>
          <td>-0.325018</td>
          <td>4.989304e-08</td>
        </tr>
      </tbody>
    </table>
    </div>



Facebook’s FastText words embedding
-----------------------------------

.. code:: ipython3

    fasttext_path = downloader.load('fasttext-wiki-news-subwords-300', return_path=True)
    print(fasttext_path)
    fasttext_model = KeyedVectors.load_word2vec_format(fasttext_path)
    
    fasttext_gender_bias_we = GenderBiasWE(fasttext_model, only_lower=False, verbose=True)


.. parsed-literal::

    /home/users/user/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz
    Identify direction using pca method...
      Principal Component    Explained Variance Ratio
    ---------------------  --------------------------
                        1                   0.531331
                        2                   0.18376
                        3                   0.089777
                        4                   0.0517856
                        5                   0.0407739
                        6                   0.0328988
                        7                   0.0223339
                        8                   0.0193495
                        9                   0.0143259
                       10                   0.0136648


We can compare the projections of neutral profession names on the gender direction for the two original words embeddings
------------------------------------------------------------------------------------------------------------------------

.. code:: ipython3

    f, ax = plt.subplots(1, figsize=(14, 10))
    GenderBiasWE.plot_bias_across_words_embeddings({'Word2Vec': w2v_gender_bias_we,
                                                    'FastText': fasttext_gender_bias_we},
                                                   ax=ax)



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_44_0.png


And now let’s preform the same steps for FastText
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    fasttext_biased_evaluation = fasttext_gender_bias_we.evaluate_words_embedding()

.. code:: ipython3

    fasttext_biased_evaluation[0]




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
          <th>pearson_r</th>
          <th>pearson_pvalue</th>
          <th>spearman_r</th>
          <th>spearman_pvalue</th>
          <th>ratio_unkonwn_words</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MEN</th>
          <td>0.669</td>
          <td>0.000</td>
          <td>0.673</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>Mturk</th>
          <td>0.676</td>
          <td>0.000</td>
          <td>0.682</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RG65</th>
          <td>0.766</td>
          <td>0.044</td>
          <td>0.667</td>
          <td>0.102</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RW</th>
          <td>0.546</td>
          <td>0.000</td>
          <td>0.546</td>
          <td>0.000</td>
          <td>16.519</td>
        </tr>
        <tr>
          <th>SimLex999</th>
          <td>0.432</td>
          <td>0.000</td>
          <td>0.418</td>
          <td>0.000</td>
          <td>0.100</td>
        </tr>
        <tr>
          <th>TR9856</th>
          <td>0.648</td>
          <td>0.000</td>
          <td>0.626</td>
          <td>0.000</td>
          <td>85.217</td>
        </tr>
        <tr>
          <th>WS353</th>
          <td>0.606</td>
          <td>0.000</td>
          <td>0.596</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_biased_evaluation[1]




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
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Google</th>
          <td>0.883</td>
        </tr>
        <tr>
          <th>MSR-syntax</th>
          <td>0.917</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_gender_bias_we.calc_direct_bias()




.. parsed-literal::

    0.07633256240142092



.. code:: ipython3

    fasttext_gender_bias_we.plot_projection_scores();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_50_0.png


.. code:: ipython3

    fasttext_gender_bias_we.plot_dist_projections_on_direction();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_51_0.png


.. code:: ipython3

    w2v_gender_bias_we.generate_analogies(150)[100:]




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
          <th>x</th>
          <th>y</th>
          <th>distance</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>100</th>
          <td>female</td>
          <td>male</td>
          <td>0.564742</td>
          <td>0.352283</td>
        </tr>
        <tr>
          <th>101</th>
          <td>singer</td>
          <td>frontman</td>
          <td>0.893373</td>
          <td>0.346936</td>
        </tr>
        <tr>
          <th>102</th>
          <td>Volleyball</td>
          <td>Football</td>
          <td>0.938328</td>
          <td>0.344304</td>
        </tr>
        <tr>
          <th>103</th>
          <td>feminist</td>
          <td>liberal</td>
          <td>0.997903</td>
          <td>0.341058</td>
        </tr>
        <tr>
          <th>104</th>
          <td>kids</td>
          <td>guys</td>
          <td>0.946709</td>
          <td>0.340327</td>
        </tr>
        <tr>
          <th>105</th>
          <td>estrogen</td>
          <td>testosterone</td>
          <td>0.796327</td>
          <td>0.335635</td>
        </tr>
        <tr>
          <th>106</th>
          <td>adorable</td>
          <td>goofy</td>
          <td>0.952828</td>
          <td>0.331766</td>
        </tr>
        <tr>
          <th>107</th>
          <td>feisty</td>
          <td>combative</td>
          <td>0.935207</td>
          <td>0.326646</td>
        </tr>
        <tr>
          <th>108</th>
          <td>skirts</td>
          <td>shorts</td>
          <td>0.953106</td>
          <td>0.326539</td>
        </tr>
        <tr>
          <th>109</th>
          <td>dresses</td>
          <td>shirts</td>
          <td>0.996928</td>
          <td>0.325354</td>
        </tr>
        <tr>
          <th>110</th>
          <td>Pelosi</td>
          <td>Boehner</td>
          <td>0.770625</td>
          <td>0.317477</td>
        </tr>
        <tr>
          <th>111</th>
          <td>cosmetics</td>
          <td>pharmaceuticals</td>
          <td>0.985631</td>
          <td>0.316112</td>
        </tr>
        <tr>
          <th>112</th>
          <td>breasts</td>
          <td>penis</td>
          <td>0.831993</td>
          <td>0.313547</td>
        </tr>
        <tr>
          <th>113</th>
          <td>LC</td>
          <td>JB</td>
          <td>0.972957</td>
          <td>0.312775</td>
        </tr>
        <tr>
          <th>114</th>
          <td>pink</td>
          <td>blue</td>
          <td>0.855175</td>
          <td>0.311602</td>
        </tr>
        <tr>
          <th>115</th>
          <td>Vogue</td>
          <td>Rolling_Stone</td>
          <td>0.980652</td>
          <td>0.310209</td>
        </tr>
        <tr>
          <th>116</th>
          <td>scarf</td>
          <td>shirt</td>
          <td>0.914326</td>
          <td>0.309644</td>
        </tr>
        <tr>
          <th>117</th>
          <td>Jolie</td>
          <td>Brad_Pitt</td>
          <td>0.779394</td>
          <td>0.309448</td>
        </tr>
        <tr>
          <th>118</th>
          <td>them</td>
          <td>him</td>
          <td>0.853787</td>
          <td>0.306956</td>
        </tr>
        <tr>
          <th>119</th>
          <td>Marie</td>
          <td>Rene</td>
          <td>0.964208</td>
          <td>0.306776</td>
        </tr>
        <tr>
          <th>120</th>
          <td>Ochoa</td>
          <td>Garcia</td>
          <td>0.923490</td>
          <td>0.306437</td>
        </tr>
        <tr>
          <th>121</th>
          <td>handbag</td>
          <td>wallet</td>
          <td>0.915948</td>
          <td>0.306354</td>
        </tr>
        <tr>
          <th>122</th>
          <td>Jennifer_Aniston</td>
          <td>George_Clooney</td>
          <td>0.900427</td>
          <td>0.303838</td>
        </tr>
        <tr>
          <th>123</th>
          <td>Palin</td>
          <td>McCain</td>
          <td>0.851226</td>
          <td>0.300329</td>
        </tr>
        <tr>
          <th>124</th>
          <td>M.</td>
          <td>D.</td>
          <td>0.559084</td>
          <td>0.297112</td>
        </tr>
        <tr>
          <th>125</th>
          <td>Abdul</td>
          <td>Omar</td>
          <td>0.942199</td>
          <td>0.296321</td>
        </tr>
        <tr>
          <th>126</th>
          <td>Oprah</td>
          <td>Rush_Limbaugh</td>
          <td>0.994474</td>
          <td>0.295990</td>
        </tr>
        <tr>
          <th>127</th>
          <td>dress</td>
          <td>wearing</td>
          <td>0.965128</td>
          <td>0.295465</td>
        </tr>
        <tr>
          <th>128</th>
          <td>Lindsay_Lohan</td>
          <td>Mel_Gibson</td>
          <td>0.897883</td>
          <td>0.290938</td>
        </tr>
        <tr>
          <th>129</th>
          <td>Wendy</td>
          <td>Denny</td>
          <td>0.955523</td>
          <td>0.289228</td>
        </tr>
        <tr>
          <th>130</th>
          <td>Hillary</td>
          <td>Kerry</td>
          <td>0.937335</td>
          <td>0.288849</td>
        </tr>
        <tr>
          <th>131</th>
          <td>Bhutto</td>
          <td>Mehsud</td>
          <td>0.951625</td>
          <td>0.288000</td>
        </tr>
        <tr>
          <th>132</th>
          <td>captivating</td>
          <td>masterful</td>
          <td>0.972531</td>
          <td>0.286297</td>
        </tr>
        <tr>
          <th>133</th>
          <td>bride</td>
          <td>groom</td>
          <td>0.805207</td>
          <td>0.285968</td>
        </tr>
        <tr>
          <th>134</th>
          <td>Coulter</td>
          <td>Dobson</td>
          <td>0.987666</td>
          <td>0.282488</td>
        </tr>
        <tr>
          <th>135</th>
          <td>Fergie</td>
          <td>Sir_Alex_Ferguson</td>
          <td>0.821957</td>
          <td>0.281196</td>
        </tr>
        <tr>
          <th>136</th>
          <td>designer</td>
          <td>architect</td>
          <td>0.998560</td>
          <td>0.280551</td>
        </tr>
        <tr>
          <th>137</th>
          <td>Dee</td>
          <td>Kenny</td>
          <td>0.967734</td>
          <td>0.277553</td>
        </tr>
        <tr>
          <th>138</th>
          <td>midfielder</td>
          <td>winger</td>
          <td>0.739950</td>
          <td>0.277374</td>
        </tr>
        <tr>
          <th>139</th>
          <td>hysterical</td>
          <td>comical</td>
          <td>0.917530</td>
          <td>0.277098</td>
        </tr>
        <tr>
          <th>140</th>
          <td>charming</td>
          <td>charismatic</td>
          <td>0.966619</td>
          <td>0.276466</td>
        </tr>
        <tr>
          <th>141</th>
          <td>Freshman</td>
          <td>Rookie</td>
          <td>0.860134</td>
          <td>0.275848</td>
        </tr>
        <tr>
          <th>142</th>
          <td>ultrasound</td>
          <td>MRI</td>
          <td>0.884500</td>
          <td>0.273803</td>
        </tr>
        <tr>
          <th>143</th>
          <td>servicemen</td>
          <td>veterans</td>
          <td>0.918718</td>
          <td>0.273633</td>
        </tr>
        <tr>
          <th>144</th>
          <td>cigarette</td>
          <td>cigar</td>
          <td>0.874594</td>
          <td>0.269909</td>
        </tr>
        <tr>
          <th>145</th>
          <td>Rae</td>
          <td>Campbell</td>
          <td>0.981424</td>
          <td>0.269635</td>
        </tr>
        <tr>
          <th>146</th>
          <td>backcourt</td>
          <td>playmaker</td>
          <td>0.975412</td>
          <td>0.269148</td>
        </tr>
        <tr>
          <th>147</th>
          <td>choreography</td>
          <td>footwork</td>
          <td>0.995236</td>
          <td>0.268480</td>
        </tr>
        <tr>
          <th>148</th>
          <td>Tony_Parker</td>
          <td>Tim_Duncan</td>
          <td>0.762779</td>
          <td>0.268100</td>
        </tr>
        <tr>
          <th>149</th>
          <td>kindness</td>
          <td>humility</td>
          <td>0.943152</td>
          <td>0.267360</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    w2v_gender_bias_we.generate_closest_words_indirect_bias('softball', 'football')




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
          <th></th>
          <th>projection</th>
          <th>indirect_bias</th>
        </tr>
        <tr>
          <th>end</th>
          <th>word</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">softball</th>
          <th>bookkeeper</th>
          <td>0.195044</td>
          <td>0.201158</td>
        </tr>
        <tr>
          <th>registered_nurse</th>
          <td>0.176933</td>
          <td>0.287150</td>
        </tr>
        <tr>
          <th>infielder</th>
          <td>0.171764</td>
          <td>-0.054049</td>
        </tr>
        <tr>
          <th>waitress</th>
          <td>0.163246</td>
          <td>0.317842</td>
        </tr>
        <tr>
          <th>receptionist</th>
          <td>0.159252</td>
          <td>0.672343</td>
        </tr>
        <tr>
          <th rowspan="5" valign="top">football</th>
          <th>philosopher</th>
          <td>-0.148482</td>
          <td>0.220857</td>
        </tr>
        <tr>
          <th>pundit</th>
          <td>-0.170339</td>
          <td>0.101227</td>
        </tr>
        <tr>
          <th>businessman</th>
          <td>-0.174114</td>
          <td>0.170078</td>
        </tr>
        <tr>
          <th>maestro</th>
          <td>-0.175094</td>
          <td>0.415804</td>
        </tr>
        <tr>
          <th>footballer</th>
          <td>-0.275374</td>
          <td>0.015366</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_gender_bias_we.debias('hard')


.. parsed-literal::

    Neutralize...


.. parsed-literal::

    100%|██████████| 998683/998683 [01:11<00:00, 13918.45it/s]


.. parsed-literal::

    Equalize...
    Equalize Words Data (all equal for 1-dim bias space (direction):
                        equalized_projected_scalar    projected_scalar    scaling
    ----------------  ----------------------------  ------------------  ---------
    (0, 'she')                            0.327879          0.272553     0.327879
    (0, 'he')                            -0.327879         -0.314674     0.327879
    (1, 'Her')                            0.348146          0.308703     0.348146
    (1, 'His')                           -0.348146         -0.247824     0.348146
    (2, 'SHE')                            0.457835          0.293494     0.457835
    (2, 'HE')                            -0.457835         -0.130146     0.457835
    (3, 'Daughter')                       0.387263          0.12953      0.387263
    (3, 'Son')                           -0.387263         -0.164091     0.387263
    (4, 'her')                            0.367055          0.329466     0.367055
    (4, 'his')                           -0.367055         -0.301367     0.367055
    (5, 'HERSELF')                        0.389722          0.27689      0.389722
    (5, 'HIMSELF')                       -0.389722         -0.256944     0.389722
    (6, 'female')                         0.22163           0.245028     0.22163
    (6, 'male')                          -0.22163           0.0909211    0.22163
    (7, 'MOTHER')                         0.294313          0.15481      0.294313
    (7, 'FATHER')                        -0.294313         -0.0130569    0.294313
    (8, 'Mary')                           0.344783          0.185954     0.344783
    (8, 'John')                          -0.344783         -0.226975     0.344783
    (9, 'FEMALE')                         0.349677          0.195223     0.349677
    (9, 'MALE')                          -0.349677          0.0749343    0.349677
    (10, 'woman')                         0.307139          0.240712     0.307139
    (10, 'man')                          -0.307139         -0.204656     0.307139
    (11, 'HER')                           0.436793          0.287152     0.436793
    (11, 'HIS')                          -0.436793         -0.211046     0.436793
    (12, 'Mother')                        0.339165          0.145467     0.339165
    (12, 'Father')                       -0.339165         -0.17219      0.339165
    (13, 'She')                           0.368328          0.309528     0.368328
    (13, 'He')                           -0.368328         -0.288727     0.368328
    (14, 'WOMAN')                         0.393236          0.22015      0.393236
    (14, 'MAN')                          -0.393236         -0.0749488    0.393236
    (15, 'Gal')                           0.564393          0.0204254    0.564393
    (15, 'Guy')                          -0.564393         -0.176119     0.564393
    (16, 'GIRL')                          0.310282          0.172866     0.310282
    (16, 'BOY')                          -0.310282         -0.0538997    0.310282
    (17, 'Herself')                       0.42383           0.231814     0.42383
    (17, 'Himself')                      -0.42383          -0.176926     0.42383
    (18, 'Woman')                         0.337546          0.187381     0.337546
    (18, 'Man')                          -0.337546         -0.170178     0.337546
    (19, 'Girl')                          0.282268          0.176313     0.282268
    (19, 'Boy')                          -0.282268         -0.119613     0.282268
    (20, 'DAUGHTER')                      0.435387          0.11358      0.435387
    (20, 'SON')                          -0.435387         -0.0219735    0.435387
    (21, 'mary')                          0.462497          0.147471     0.462497
    (21, 'john')                         -0.462497         -0.117967     0.462497
    (22, 'MARY')                          0.362825          0.187168     0.362825
    (22, 'JOHN')                         -0.362825         -0.171325     0.362825
    (23, 'mother')                        0.241317          0.163933     0.241317
    (23, 'father')                       -0.241317         -0.134988     0.241317
    (24, 'gal')                           0.47685           0.237141     0.47685
    (24, 'guy')                          -0.47685          -0.290092     0.47685
    (25, 'GAL')                           0.590532          0.0750057    0.590532
    (25, 'GUY')                          -0.590532         -0.0574314    0.590532
    (26, 'Female')                        0.283831          0.255345     0.283831
    (26, 'Male')                         -0.283831          0.00701683   0.283831
    (27, 'herself')                       0.354767          0.361745     0.354767
    (27, 'himself')                      -0.354767         -0.288689     0.354767
    (28, 'daughter')                      0.240065          0.134447     0.240065
    (28, 'son')                          -0.240065         -0.145326     0.240065
    (29, 'girl')                          0.249019          0.205119     0.249019
    (29, 'boy')                          -0.249019         -0.152492     0.249019


.. code:: ipython3

    fasttext_debiased_evaluation = fasttext_gender_bias_we.evaluate_words_embedding()

.. code:: ipython3

    fasttext_debiased_evaluation[0]




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
          <th>pearson_r</th>
          <th>pearson_pvalue</th>
          <th>spearman_r</th>
          <th>spearman_pvalue</th>
          <th>ratio_unkonwn_words</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>MEN</th>
          <td>0.669</td>
          <td>0.000</td>
          <td>0.673</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>Mturk</th>
          <td>0.677</td>
          <td>0.000</td>
          <td>0.682</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RG65</th>
          <td>0.767</td>
          <td>0.044</td>
          <td>0.667</td>
          <td>0.102</td>
          <td>0.000</td>
        </tr>
        <tr>
          <th>RW</th>
          <td>0.545</td>
          <td>0.000</td>
          <td>0.546</td>
          <td>0.000</td>
          <td>16.519</td>
        </tr>
        <tr>
          <th>SimLex999</th>
          <td>0.433</td>
          <td>0.000</td>
          <td>0.419</td>
          <td>0.000</td>
          <td>0.100</td>
        </tr>
        <tr>
          <th>TR9856</th>
          <td>0.647</td>
          <td>0.000</td>
          <td>0.625</td>
          <td>0.000</td>
          <td>85.217</td>
        </tr>
        <tr>
          <th>WS353</th>
          <td>0.609</td>
          <td>0.000</td>
          <td>0.598</td>
          <td>0.000</td>
          <td>0.000</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_debiased_evaluation[1]




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
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Google</th>
          <td>0.882</td>
        </tr>
        <tr>
          <th>MSR-syntax</th>
          <td>0.916</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_gender_bias_we.calc_direct_bias()




.. parsed-literal::

    1.4306556948502593e-09



.. code:: ipython3

    fasttext_gender_bias_we.plot_projection_scores();



.. image:: demo-gender-bias-words-embedding_files/demo-gender-bias-words-embedding_59_0.png


.. code:: ipython3

    fasttext_gender_bias_we.generate_analogies(150)[100:]




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
          <th>x</th>
          <th>y</th>
          <th>distance</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>100</th>
          <td>3-4</td>
          <td>3-5</td>
          <td>0.451103</td>
          <td>1.995020e-08</td>
        </tr>
        <tr>
          <th>101</th>
          <td>77</td>
          <td>71</td>
          <td>0.367198</td>
          <td>1.971283e-08</td>
        </tr>
        <tr>
          <th>102</th>
          <td>23rd</td>
          <td>24th</td>
          <td>0.231645</td>
          <td>1.946727e-08</td>
        </tr>
        <tr>
          <th>103</th>
          <td>Feb</td>
          <td>Oct</td>
          <td>0.261598</td>
          <td>1.938152e-08</td>
        </tr>
        <tr>
          <th>104</th>
          <td>biotech</td>
          <td>biotechnology</td>
          <td>0.534111</td>
          <td>1.923940e-08</td>
        </tr>
        <tr>
          <th>105</th>
          <td>October</td>
          <td>November</td>
          <td>0.172787</td>
          <td>1.919794e-08</td>
        </tr>
        <tr>
          <th>106</th>
          <td>Tuesday</td>
          <td>Thursday</td>
          <td>0.217532</td>
          <td>1.911626e-08</td>
        </tr>
        <tr>
          <th>107</th>
          <td>non-profit</td>
          <td>nonprofit</td>
          <td>0.416925</td>
          <td>1.909318e-08</td>
        </tr>
        <tr>
          <th>108</th>
          <td>3.4</td>
          <td>4.6</td>
          <td>0.427952</td>
          <td>1.906302e-08</td>
        </tr>
        <tr>
          <th>109</th>
          <td>1814</td>
          <td>1811</td>
          <td>0.420682</td>
          <td>1.865536e-08</td>
        </tr>
        <tr>
          <th>110</th>
          <td>1937</td>
          <td>1935</td>
          <td>0.194789</td>
          <td>1.851965e-08</td>
        </tr>
        <tr>
          <th>111</th>
          <td>30th</td>
          <td>15th</td>
          <td>0.419066</td>
          <td>1.848245e-08</td>
        </tr>
        <tr>
          <th>112</th>
          <td>earthquake</td>
          <td>quake</td>
          <td>0.465103</td>
          <td>1.823179e-08</td>
        </tr>
        <tr>
          <th>113</th>
          <td>1929</td>
          <td>1927</td>
          <td>0.255561</td>
          <td>1.814888e-08</td>
        </tr>
        <tr>
          <th>114</th>
          <td>194</td>
          <td>182</td>
          <td>0.501411</td>
          <td>1.793092e-08</td>
        </tr>
        <tr>
          <th>115</th>
          <td>Somali</td>
          <td>Indian</td>
          <td>0.995805</td>
          <td>1.790283e-08</td>
        </tr>
        <tr>
          <th>116</th>
          <td>1795</td>
          <td>1807</td>
          <td>0.446287</td>
          <td>1.787080e-08</td>
        </tr>
        <tr>
          <th>117</th>
          <td>1796</td>
          <td>1808</td>
          <td>0.425368</td>
          <td>1.778199e-08</td>
        </tr>
        <tr>
          <th>118</th>
          <td>206</td>
          <td>299</td>
          <td>0.623349</td>
          <td>1.773416e-08</td>
        </tr>
        <tr>
          <th>119</th>
          <td>burden</td>
          <td>burdens</td>
          <td>0.491540</td>
          <td>1.769373e-08</td>
        </tr>
        <tr>
          <th>120</th>
          <td>Yunnan</td>
          <td>Shenzhen</td>
          <td>0.950271</td>
          <td>1.768796e-08</td>
        </tr>
        <tr>
          <th>121</th>
          <td>25th</td>
          <td>26th</td>
          <td>0.289118</td>
          <td>1.753309e-08</td>
        </tr>
        <tr>
          <th>122</th>
          <td>54</td>
          <td>53</td>
          <td>0.132329</td>
          <td>1.751296e-08</td>
        </tr>
        <tr>
          <th>123</th>
          <td>Kraków</td>
          <td>Kiev</td>
          <td>0.916026</td>
          <td>1.744519e-08</td>
        </tr>
        <tr>
          <th>124</th>
          <td>burnt</td>
          <td>burned</td>
          <td>0.443849</td>
          <td>1.737333e-08</td>
        </tr>
        <tr>
          <th>125</th>
          <td>264</td>
          <td>251</td>
          <td>0.481102</td>
          <td>1.724800e-08</td>
        </tr>
        <tr>
          <th>126</th>
          <td>consist</td>
          <td>consists</td>
          <td>0.567469</td>
          <td>1.722748e-08</td>
        </tr>
        <tr>
          <th>127</th>
          <td>152</td>
          <td>148</td>
          <td>0.367319</td>
          <td>1.721183e-08</td>
        </tr>
        <tr>
          <th>128</th>
          <td>279</td>
          <td>259</td>
          <td>0.432283</td>
          <td>1.720215e-08</td>
        </tr>
        <tr>
          <th>129</th>
          <td>4-1</td>
          <td>5-1</td>
          <td>0.357604</td>
          <td>1.712565e-08</td>
        </tr>
        <tr>
          <th>130</th>
          <td>financially</td>
          <td>spiritually</td>
          <td>0.818715</td>
          <td>1.707935e-08</td>
        </tr>
        <tr>
          <th>131</th>
          <td>3.</td>
          <td>2.</td>
          <td>0.321363</td>
          <td>1.698265e-08</td>
        </tr>
        <tr>
          <th>132</th>
          <td>al-Qaida</td>
          <td>al-Qaeda</td>
          <td>0.347117</td>
          <td>1.690197e-08</td>
        </tr>
        <tr>
          <th>133</th>
          <td>depart</td>
          <td>leave</td>
          <td>0.879645</td>
          <td>1.686580e-08</td>
        </tr>
        <tr>
          <th>134</th>
          <td>February</td>
          <td>January</td>
          <td>0.173507</td>
          <td>1.679425e-08</td>
        </tr>
        <tr>
          <th>135</th>
          <td>7.2</td>
          <td>6.1</td>
          <td>0.393743</td>
          <td>1.677078e-08</td>
        </tr>
        <tr>
          <th>136</th>
          <td>Walmart</td>
          <td>Wal-Mart</td>
          <td>0.443242</td>
          <td>1.663763e-08</td>
        </tr>
        <tr>
          <th>137</th>
          <td>Hungarian</td>
          <td>Greek</td>
          <td>0.967373</td>
          <td>1.653301e-08</td>
        </tr>
        <tr>
          <th>138</th>
          <td>T-shirt</td>
          <td>t-shirt</td>
          <td>0.391520</td>
          <td>1.652421e-08</td>
        </tr>
        <tr>
          <th>139</th>
          <td>deem</td>
          <td>classify</td>
          <td>0.957838</td>
          <td>1.649298e-08</td>
        </tr>
        <tr>
          <th>140</th>
          <td>Somehow</td>
          <td>Surely</td>
          <td>0.896766</td>
          <td>1.642662e-08</td>
        </tr>
        <tr>
          <th>141</th>
          <td>277</td>
          <td>289</td>
          <td>0.433515</td>
          <td>1.635572e-08</td>
        </tr>
        <tr>
          <th>142</th>
          <td>flyers</td>
          <td>fliers</td>
          <td>0.486239</td>
          <td>1.631699e-08</td>
        </tr>
        <tr>
          <th>143</th>
          <td>flavour</td>
          <td>flavor</td>
          <td>0.532512</td>
          <td>1.630458e-08</td>
        </tr>
        <tr>
          <th>144</th>
          <td>flaws</td>
          <td>drawbacks</td>
          <td>0.797717</td>
          <td>1.626930e-08</td>
        </tr>
        <tr>
          <th>145</th>
          <td>analog</td>
          <td>analogue</td>
          <td>0.473394</td>
          <td>1.621330e-08</td>
        </tr>
        <tr>
          <th>146</th>
          <td>Sundays</td>
          <td>Fridays</td>
          <td>0.587774</td>
          <td>1.621318e-08</td>
        </tr>
        <tr>
          <th>147</th>
          <td>vineyard</td>
          <td>vineyards</td>
          <td>0.482495</td>
          <td>1.616479e-08</td>
        </tr>
        <tr>
          <th>148</th>
          <td>280</td>
          <td>220</td>
          <td>0.423477</td>
          <td>1.611552e-08</td>
        </tr>
        <tr>
          <th>149</th>
          <td>Madison</td>
          <td>Baden</td>
          <td>0.983458</td>
          <td>1.607278e-08</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    fasttext_gender_bias_we.generate_closest_words_indirect_bias('softball', 'football')




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
          <th></th>
          <th>projection</th>
          <th>indirect_bias</th>
        </tr>
        <tr>
          <th>end</th>
          <th>word</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">softball</th>
          <th>infielder</th>
          <td>0.186223</td>
          <td>4.205870e-08</td>
        </tr>
        <tr>
          <th>alderman</th>
          <td>0.100549</td>
          <td>7.837655e-08</td>
        </tr>
        <tr>
          <th>handyman</th>
          <td>0.066572</td>
          <td>-6.646021e-03</td>
        </tr>
        <tr>
          <th>mediator</th>
          <td>0.060695</td>
          <td>1.667881e-07</td>
        </tr>
        <tr>
          <th>ranger</th>
          <td>0.059893</td>
          <td>1.444559e-07</td>
        </tr>
        <tr>
          <th rowspan="5" valign="top">football</th>
          <th>coach</th>
          <td>-0.185766</td>
          <td>5.122636e-08</td>
        </tr>
        <tr>
          <th>sportsman</th>
          <td>-0.204266</td>
          <td>1.253264e-07</td>
        </tr>
        <tr>
          <th>goalkeeper</th>
          <td>-0.234506</td>
          <td>7.194591e-08</td>
        </tr>
        <tr>
          <th>midfielder</th>
          <td>-0.236263</td>
          <td>1.018931e-08</td>
        </tr>
        <tr>
          <th>footballer</th>
          <td>-0.385277</td>
          <td>1.098612e-07</td>
        </tr>
      </tbody>
    </table>
    </div>


