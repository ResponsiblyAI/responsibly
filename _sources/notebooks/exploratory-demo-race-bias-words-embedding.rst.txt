
Demo - Race Bias in Words Embedding
===================================

Exploratory - API will change in a future release
-------------------------------------------------

Based on:
https://github.com/tolga-b/debiaswe/blob/master/tutorial_example1.ipynb

.. code:: ipython3

    import matplotlib.pylab as plt
    
    from ethically.we import BiasWordsEmbedding
    from ethically.we.data import load_w2v_small, BOLUKBASI_DATA

.. code:: ipython3

    w2v_small_model = load_w2v_small()

.. code:: ipython3

    names = ["Emily", "Aisha", "Anne", "Keisha",
             "Jill", "Tamika", "Allison", "Lakisha",
             "Laurie", "Tanisha", "Sarah", "Latoya",
             "Meredith", "Kenya", "Carrie", "Latonya",
             "Kristen", "Ebony", "Todd", "Rasheed",
             "Neil", "Tremayne", "Geoffrey", "Kareem",
             "Brett", "Darnell", "Brendan", "Tyrone",
             "Greg", "Hakim", "Matthew", "Jamal",
             "Jay", "Leroy", "Brad", "Jermaine"]
    
    group1 = names[::2]
    group2 = names[1::2]

.. code:: ipython3

    race_bias_we = BiasWordsEmbedding(w2v_small_model,
                                      verbose=True)

.. code:: ipython3

    race_bias_we._identify_direction('group1', 'group2',
                                     [group1, group2],
                                     method='sum')


.. parsed-literal::

    Identify direction using sum method...


.. code:: ipython3

    profession_names = race_bias_we._filter_words_by_model(BOLUKBASI_DATA['gender']['profession_names'])

.. code:: ipython3

    race_bias_we.calc_direct_bias(profession_names)




.. parsed-literal::

    0.057185549110977264



.. code:: ipython3

    race_bias_we.plot_dist_projections_on_direction({'profession_names': profession_names,
                                                     'group1': group1,
                                                     'group2': group2});



.. image:: exploratory-demo-race-bias-words-embedding_files/exploratory-demo-race-bias-words-embedding_8_0.png


.. code:: ipython3

    race_bias_we.generate_analogies(30)




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
          <th>0</th>
          <td>Sarah</td>
          <td>Keisha</td>
          <td>0.928895</td>
          <td>0.670521</td>
        </tr>
        <tr>
          <th>1</th>
          <td>defensemen</td>
          <td>cornerbacks</td>
          <td>0.995745</td>
          <td>0.371968</td>
        </tr>
        <tr>
          <th>2</th>
          <td>hipster</td>
          <td>hip_hop</td>
          <td>0.990213</td>
          <td>0.359780</td>
        </tr>
        <tr>
          <th>3</th>
          <td>punter</td>
          <td>cornerback</td>
          <td>0.904813</td>
          <td>0.352770</td>
        </tr>
        <tr>
          <th>4</th>
          <td>singer_songwriter</td>
          <td>rapper</td>
          <td>0.999137</td>
          <td>0.343185</td>
        </tr>
        <tr>
          <th>5</th>
          <td>defenseman</td>
          <td>defensive_tackle</td>
          <td>0.965712</td>
          <td>0.342796</td>
        </tr>
        <tr>
          <th>6</th>
          <td>pole_vault</td>
          <td>triple_jump</td>
          <td>0.463255</td>
          <td>0.339006</td>
        </tr>
        <tr>
          <th>7</th>
          <td>musicians</td>
          <td>artistes</td>
          <td>0.859174</td>
          <td>0.328106</td>
        </tr>
        <tr>
          <th>8</th>
          <td>tavern</td>
          <td>barbershop</td>
          <td>0.976077</td>
          <td>0.306346</td>
        </tr>
        <tr>
          <th>9</th>
          <td>freestyle_relay</td>
          <td>meter_hurdles</td>
          <td>0.748041</td>
          <td>0.301123</td>
        </tr>
        <tr>
          <th>10</th>
          <td>bacon</td>
          <td>fried_chicken</td>
          <td>0.955136</td>
          <td>0.298132</td>
        </tr>
        <tr>
          <th>11</th>
          <td>equipment</td>
          <td>equipments</td>
          <td>0.750463</td>
          <td>0.294613</td>
        </tr>
        <tr>
          <th>12</th>
          <td>hockey</td>
          <td>basketball</td>
          <td>0.879606</td>
          <td>0.285121</td>
        </tr>
        <tr>
          <th>13</th>
          <td>wool</td>
          <td>cotton</td>
          <td>0.963230</td>
          <td>0.280360</td>
        </tr>
        <tr>
          <th>14</th>
          <td>unassisted_goal</td>
          <td>layup</td>
          <td>0.870970</td>
          <td>0.280274</td>
        </tr>
        <tr>
          <th>15</th>
          <td>chocolates</td>
          <td>sweets</td>
          <td>0.776279</td>
          <td>0.280231</td>
        </tr>
        <tr>
          <th>16</th>
          <td>buddy</td>
          <td>cousin</td>
          <td>0.968458</td>
          <td>0.273012</td>
        </tr>
        <tr>
          <th>17</th>
          <td>priest</td>
          <td>preacher</td>
          <td>0.988195</td>
          <td>0.272797</td>
        </tr>
        <tr>
          <th>18</th>
          <td>blue</td>
          <td>black</td>
          <td>0.949484</td>
          <td>0.269887</td>
        </tr>
        <tr>
          <th>19</th>
          <td>quirky</td>
          <td>funky</td>
          <td>0.904340</td>
          <td>0.266961</td>
        </tr>
        <tr>
          <th>20</th>
          <td>rabbi</td>
          <td>imam</td>
          <td>0.946823</td>
          <td>0.265076</td>
        </tr>
        <tr>
          <th>21</th>
          <td>grapes</td>
          <td>mango</td>
          <td>0.978360</td>
          <td>0.264401</td>
        </tr>
        <tr>
          <th>22</th>
          <td>telecommunications</td>
          <td>telecommunication</td>
          <td>0.512355</td>
          <td>0.261840</td>
        </tr>
        <tr>
          <th>23</th>
          <td>passages</td>
          <td>verses</td>
          <td>0.909501</td>
          <td>0.255701</td>
        </tr>
        <tr>
          <th>24</th>
          <td>er</td>
          <td>o</td>
          <td>0.919915</td>
          <td>0.254029</td>
        </tr>
        <tr>
          <th>25</th>
          <td>acoustic</td>
          <td>soulful</td>
          <td>0.921887</td>
          <td>0.253801</td>
        </tr>
        <tr>
          <th>26</th>
          <td>punting</td>
          <td>punt_returns</td>
          <td>0.851885</td>
          <td>0.253574</td>
        </tr>
        <tr>
          <th>27</th>
          <td>thefts</td>
          <td>armed_robbery</td>
          <td>0.987904</td>
          <td>0.251027</td>
        </tr>
        <tr>
          <th>28</th>
          <td>bar</td>
          <td>nightclub</td>
          <td>0.913451</td>
          <td>0.251026</td>
        </tr>
        <tr>
          <th>29</th>
          <td>digs</td>
          <td>rebounds</td>
          <td>0.948400</td>
          <td>0.249695</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    f, ax = plt.subplots(figsize=(15, 15))
    race_bias_we.plot_projection_scores(profession_names, 15, ax=ax);



.. image:: exploratory-demo-race-bias-words-embedding_files/exploratory-demo-race-bias-words-embedding_10_0.png

