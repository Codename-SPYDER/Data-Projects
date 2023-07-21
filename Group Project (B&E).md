```python
# Import needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Import Break and Enter dataset
Robbery = pd.read_csv('./Basic_Methods-(Data)/Group Project/Break_and_Enter_2014_to_2019.csv',encoding = 'unicode_escape')
```


```python
Robbery.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43302 entries, 0 to 43301
    Data columns (total 29 columns):
    X                      43302 non-null float64
    Y                      43302 non-null float64
    Index_                 43302 non-null int64
    event_unique_id        43302 non-null object
    occurrencedate         43302 non-null object
    reporteddate           43302 non-null object
    premisetype            43302 non-null object
    ucr_code               43302 non-null int64
    ucr_ext                43302 non-null int64
    offence                43302 non-null object
    reportedyear           43302 non-null int64
    reportedmonth          43302 non-null object
    reportedday            43302 non-null int64
    reporteddayofyear      43302 non-null int64
    reporteddayofweek      43302 non-null object
    reportedhour           43302 non-null int64
    occurrenceyear         43301 non-null float64
    occurrencemonth        43301 non-null object
    occurrenceday          43301 non-null float64
    occurrencedayofyear    43301 non-null float64
    occurrencedayofweek    43301 non-null object
    occurrencehour         43302 non-null int64
    MCI                    43302 non-null object
    Division               43302 non-null object
    Hood_ID                43302 non-null int64
    Neighbourhood          43302 non-null object
    Lat                    43302 non-null float64
    Long                   43302 non-null float64
    ObjectId               43302 non-null int64
    dtypes: float64(7), int64(10), object(12)
    memory usage: 9.6+ MB



```python
Robbery.describe()
```




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
      <th>X</th>
      <th>Y</th>
      <th>Index_</th>
      <th>ucr_code</th>
      <th>ucr_ext</th>
      <th>reportedyear</th>
      <th>reportedday</th>
      <th>reporteddayofyear</th>
      <th>reportedhour</th>
      <th>occurrenceyear</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>Lat</th>
      <th>Long</th>
      <th>ObjectId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>4.330200e+04</td>
      <td>4.330200e+04</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43301.000000</td>
      <td>43301.000000</td>
      <td>43301.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
      <td>43302.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>-8.837574e+06</td>
      <td>5.420199e+06</td>
      <td>98308.070274</td>
      <td>2120.001132</td>
      <td>202.948132</td>
      <td>2016.598032</td>
      <td>15.781534</td>
      <td>187.371553</td>
      <td>12.518175</td>
      <td>2016.585067</td>
      <td>15.695250</td>
      <td>187.801229</td>
      <td>10.996097</td>
      <td>73.977438</td>
      <td>43.706895</td>
      <td>-79.389278</td>
      <td>21651.500000</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.066907e+04</td>
      <td>8.236009e+03</td>
      <td>58980.101633</td>
      <td>0.069465</td>
      <td>6.919186</td>
      <td>1.750686</td>
      <td>8.612507</td>
      <td>105.187760</td>
      <td>6.133789</td>
      <td>1.758883</td>
      <td>8.645634</td>
      <td>105.106452</td>
      <td>6.966893</td>
      <td>38.722861</td>
      <td>0.053475</td>
      <td>0.095842</td>
      <td>12500.355015</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-8.864928e+06</td>
      <td>5.401765e+06</td>
      <td>2.000000</td>
      <td>2120.000000</td>
      <td>200.000000</td>
      <td>2014.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2002.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>43.587093</td>
      <td>-79.635000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-8.843888e+06</td>
      <td>5.413291e+06</td>
      <td>44710.250000</td>
      <td>2120.000000</td>
      <td>200.000000</td>
      <td>2015.000000</td>
      <td>8.000000</td>
      <td>98.000000</td>
      <td>8.000000</td>
      <td>2015.000000</td>
      <td>8.000000</td>
      <td>98.000000</td>
      <td>5.000000</td>
      <td>41.000000</td>
      <td>43.662045</td>
      <td>-79.446000</td>
      <td>10826.250000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>-8.837877e+06</td>
      <td>5.418729e+06</td>
      <td>98412.500000</td>
      <td>2120.000000</td>
      <td>200.000000</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>191.000000</td>
      <td>13.000000</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>191.000000</td>
      <td>11.000000</td>
      <td>76.000000</td>
      <td>43.697376</td>
      <td>-79.392000</td>
      <td>21651.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>-8.829973e+06</td>
      <td>5.427003e+06</td>
      <td>152150.750000</td>
      <td>2120.000000</td>
      <td>200.000000</td>
      <td>2018.000000</td>
      <td>23.000000</td>
      <td>280.000000</td>
      <td>17.000000</td>
      <td>2018.000000</td>
      <td>23.000000</td>
      <td>280.000000</td>
      <td>17.000000</td>
      <td>105.000000</td>
      <td>43.751087</td>
      <td>-79.321000</td>
      <td>32476.750000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>-8.808043e+06</td>
      <td>5.440879e+06</td>
      <td>196520.000000</td>
      <td>2125.000000</td>
      <td>230.000000</td>
      <td>2019.000000</td>
      <td>31.000000</td>
      <td>366.000000</td>
      <td>23.000000</td>
      <td>2019.000000</td>
      <td>31.000000</td>
      <td>366.000000</td>
      <td>23.000000</td>
      <td>140.000000</td>
      <td>43.841065</td>
      <td>-79.124000</td>
      <td>43302.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# More than 10,000 occurrences of B&E
Robbery['offence'].value_counts()
```




    B&E                             36305
    B&E W'Intent                     5630
    Unlawfully In Dwelling-House     1272
    B&E Out                            78
    B&E - To Steal Firearm              9
    B&E - M/Veh To Steal Firearm        8
    Name: offence, dtype: int64




```python
Robbery['offence'].value_counts().plot(kind = 'barh')
plt.title('B&E Distribution')
```




    Text(0.5, 1.0, 'B&E Distribution')




    
![png](output_5_1.png)
    



```python
# Cut out categories other than B&E
is_BE = Robbery['offence'] == 'B&E'
```


```python
# Cut out categories other than B&E
Robbery_BE = Robbery[is_BE]
```


```python
# Make sure B&E is all that remains
Robbery_BE['offence'].value_counts()
```




    B&E    36305
    Name: offence, dtype: int64




```python
# More than 10,000 occurrences of House and Commercial
Robbery_BE['premisetype'].value_counts()
```




    House         12873
    Commercial    11622
    Apartment      9576
    Other          2221
    Outside          13
    Name: premisetype, dtype: int64




```python
Robbery_BE['premisetype'].value_counts().plot(kind = 'barh')
plt.title('Premise Distribution')
```




    Text(0.5, 1.0, 'Premise Distribution')




    
![png](output_10_1.png)
    



```python
Robbery_BE['premisetype'].value_counts()
```




    House         12873
    Commercial    11622
    Apartment      9576
    Other          2221
    Outside          13
    Name: premisetype, dtype: int64




```python
# Use House since this category has to do with our objective
is_House = Robbery_BE['premisetype'] == 'House'
```


```python
# Use House since this category has to do with our objective
Robbery_P = Robbery_BE[is_House]
```


```python
# Check to make sure House is all that remains
Robbery_P['premisetype'].value_counts()
```




    House    12873
    Name: premisetype, dtype: int64




```python
Robbery_P.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12873 entries, 16 to 43284
    Data columns (total 29 columns):
    X                      12873 non-null float64
    Y                      12873 non-null float64
    Index_                 12873 non-null int64
    event_unique_id        12873 non-null object
    occurrencedate         12873 non-null object
    reporteddate           12873 non-null object
    premisetype            12873 non-null object
    ucr_code               12873 non-null int64
    ucr_ext                12873 non-null int64
    offence                12873 non-null object
    reportedyear           12873 non-null int64
    reportedmonth          12873 non-null object
    reportedday            12873 non-null int64
    reporteddayofyear      12873 non-null int64
    reporteddayofweek      12873 non-null object
    reportedhour           12873 non-null int64
    occurrenceyear         12873 non-null float64
    occurrencemonth        12873 non-null object
    occurrenceday          12873 non-null float64
    occurrencedayofyear    12873 non-null float64
    occurrencedayofweek    12873 non-null object
    occurrencehour         12873 non-null int64
    MCI                    12873 non-null object
    Division               12873 non-null object
    Hood_ID                12873 non-null int64
    Neighbourhood          12873 non-null object
    Lat                    12873 non-null float64
    Long                   12873 non-null float64
    ObjectId               12873 non-null int64
    dtypes: float64(7), int64(10), object(12)
    memory usage: 2.9+ MB



```python
df = Robbery_P.copy()
```


```python
# Drop unecessary columns
df.drop(['X','Y','Index_','event_unique_id','reporteddate','ucr_code','ucr_ext'], axis = 1, inplace = True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>reportedyear</th>
      <th>reportedmonth</th>
      <th>reportedday</th>
      <th>reporteddayofyear</th>
      <th>reporteddayofweek</th>
      <th>reportedhour</th>
      <th>occurrenceyear</th>
      <th>...</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>MCI</th>
      <th>Division</th>
      <th>Hood_ID</th>
      <th>Neighbourhood</th>
      <th>Lat</th>
      <th>Long</th>
      <th>ObjectId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014</td>
      <td>January</td>
      <td>7</td>
      <td>7</td>
      <td>Tuesday</td>
      <td>21</td>
      <td>2014.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>Break and Enter</td>
      <td>D22</td>
      <td>9</td>
      <td>Edenbridge-Humber Valley (9)</td>
      <td>43.679287</td>
      <td>-79.525</td>
      <td>17</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014</td>
      <td>January</td>
      <td>14</td>
      <td>14</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>2014.0</td>
      <td>...</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>Break and Enter</td>
      <td>D41</td>
      <td>119</td>
      <td>Wexford/Maryvale (119)</td>
      <td>43.746746</td>
      <td>-79.303</td>
      <td>22</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014</td>
      <td>January</td>
      <td>31</td>
      <td>31</td>
      <td>Friday</td>
      <td>18</td>
      <td>2014.0</td>
      <td>...</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>Break and Enter</td>
      <td>D31</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.754845</td>
      <td>-79.506</td>
      <td>24</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014</td>
      <td>January</td>
      <td>30</td>
      <td>30</td>
      <td>Thursday</td>
      <td>19</td>
      <td>2014.0</td>
      <td>...</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>Break and Enter</td>
      <td>D33</td>
      <td>45</td>
      <td>Parkwoods-Donalda (45)</td>
      <td>43.742348</td>
      <td>-79.323</td>
      <td>26</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014</td>
      <td>February</td>
      <td>15</td>
      <td>46</td>
      <td>Saturday</td>
      <td>21</td>
      <td>2014.0</td>
      <td>...</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>Break and Enter</td>
      <td>D53</td>
      <td>103</td>
      <td>Lawrence Park South (103)</td>
      <td>43.718899</td>
      <td>-79.404</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
# Drop unecessary columns
df.drop(['reportedyear','reportedmonth','reportedday','reporteddayofweek','reportedhour','MCI','Division'], axis = 1, inplace = True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>reporteddayofyear</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>Neighbourhood</th>
      <th>Lat</th>
      <th>Long</th>
      <th>ObjectId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>7</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>Edenbridge-Humber Valley (9)</td>
      <td>43.679287</td>
      <td>-79.525</td>
      <td>17</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>14</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>Wexford/Maryvale (119)</td>
      <td>43.746746</td>
      <td>-79.303</td>
      <td>22</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>31</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.754845</td>
      <td>-79.506</td>
      <td>24</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>30</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>Parkwoods-Donalda (45)</td>
      <td>43.742348</td>
      <td>-79.323</td>
      <td>26</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>46</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>Lawrence Park South (103)</td>
      <td>43.718899</td>
      <td>-79.404</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12873 entries, 16 to 43284
    Data columns (total 15 columns):
    occurrencedate         12873 non-null object
    premisetype            12873 non-null object
    offence                12873 non-null object
    reporteddayofyear      12873 non-null int64
    occurrenceyear         12873 non-null float64
    occurrencemonth        12873 non-null object
    occurrenceday          12873 non-null float64
    occurrencedayofyear    12873 non-null float64
    occurrencedayofweek    12873 non-null object
    occurrencehour         12873 non-null int64
    Hood_ID                12873 non-null int64
    Neighbourhood          12873 non-null object
    Lat                    12873 non-null float64
    Long                   12873 non-null float64
    ObjectId               12873 non-null int64
    dtypes: float64(5), int64(4), object(6)
    memory usage: 1.6+ MB



```python
df['Hood_ID'].value_counts(bins=5)
```




    (28.8, 56.6]      3508
    (112.2, 140.0]    3484
    (84.4, 112.2]     2072
    (0.86, 28.8]      2067
    (56.6, 84.4]      1742
    Name: Hood_ID, dtype: int64




```python
# Drop unecessary columns
df.drop(['Neighbourhood','Lat','Long','ObjectId','reporteddayofyear'], axis = 1, inplace = True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['occurrencedayofweek'].value_counts()
```




    Friday        2342
    Thursday      2077
    Wednesday     1971
    Tuesday       1906
    Monday        1755
    Saturday      1687
    Sunday        1135
    Name: occurrencedayofweek, dtype: int64




```python
# Can't replace so will use dummy variables
df['occurrencedayofweek'].replace('Wednesday', 3, inplace=True)
```


```python
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop unecessary columns
df.loc[df['occurrencedayofweek'] == 'Wednesday', 'occurrencedayofweek'].replace('Wednesday', 3, inplace=True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Didn't Work
conditions = [
    (df['occurrencedayofweek'] == 'Monday'),
    (df['occurrencedayofweek'] == 'Tuesday'),
    (df['occurrencedayofweek'] == 'Wednesday'),
    (df['occurrencedayofweek'] == 'Thursday'),
    (df['occurrencedayofweek'] == 'Friday'),
    (df['occurrencedayofweek'] == 'Saturday'),
    (df['occurrencedayofweek'] == 'Sunday'),
    ]





```


```python
# Didn't Work
values = [1,2,3,4,5,6,7]
```


```python
# Didn't Work
df['week_day'] = np.select(conditions, values)
```


```python
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change months to numerical data
df['occurrencemonth'].replace({"July": 7}, inplace=True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>January</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>February</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change months to numerical data
df['occurrencemonth'].replace({'January': 1,'February':2,'March':3,'April':4,'May':5 }, inplace=True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change months to numerical data
df['occurrencemonth'].replace({'June': 6,'August':8,'September':9,'October':10,'November':11,'December':12 }, inplace=True)
df.head(10)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>offence</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>0</td>
    </tr>
    <tr>
      <td>34</td>
      <td>2014/01/16 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>1</td>
      <td>16.0</td>
      <td>16.0</td>
      <td>Thursday</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>35</td>
      <td>2014/02/09 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>9.0</td>
      <td>40.0</td>
      <td>Sunday</td>
      <td>14</td>
      <td>102</td>
      <td>0</td>
    </tr>
    <tr>
      <td>39</td>
      <td>2014/02/27 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>27.0</td>
      <td>58.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <td>40</td>
      <td>2014/02/27 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>27.0</td>
      <td>58.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>140</td>
      <td>0</td>
    </tr>
    <tr>
      <td>41</td>
      <td>2014/02/27 05:00:00+00</td>
      <td>House</td>
      <td>B&amp;E</td>
      <td>2014.0</td>
      <td>2</td>
      <td>27.0</td>
      <td>58.0</td>
      <td>Thursday</td>
      <td>12</td>
      <td>132</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['offence','week_day'], axis = 1, inplace = True)
df.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check to see if months have been successfully changed
df['occurrencemonth'].value_counts()
```




    11    1336
    10    1256
    12    1191
    8     1113
    6     1039
    1     1038
    9     1019
    5      997
    7      992
    3      991
    4      952
    2      949
    Name: occurrencemonth, dtype: int64




```python
# Check for null values
df.isnull().sum()
```




    occurrencedate         0
    premisetype            0
    occurrenceyear         0
    occurrencemonth        0
    occurrenceday          0
    occurrencedayofyear    0
    occurrencedayofweek    0
    occurrencehour         0
    Hood_ID                0
    dtype: int64




```python
# Check for zeros
df[df==0].count(axis=0, level=None, numeric_only=False)
```




    occurrencedate           0
    premisetype              0
    occurrenceyear           0
    occurrencemonth          0
    occurrenceday            0
    occurrencedayofyear      0
    occurrencedayofweek      0
    occurrencehour         608
    Hood_ID                  0
    dtype: int64




```python
# Check for abnormalities in numbers
df['occurrenceyear'].hist(figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a185210>




    
![png](output_37_1.png)
    



```python
df['occurrenceyear'].value_counts()
```




    2014.0    2399
    2015.0    2222
    2017.0    2204
    2018.0    2108
    2016.0    2097
    2019.0    1823
    2013.0      19
    2010.0       1
    Name: occurrenceyear, dtype: int64




```python
df['occurrenceyear'].value_counts().plot(kind = 'barh')
plt.title('Year')
```




    Text(0.5, 1.0, 'Year')




    
![png](output_39_1.png)
    



```python
# Choose to use last 6 years of data
df_outlier = df[df['occurrenceyear'].isin([2014,2015,2016,2017,2018,2019])]
```


```python
df_outlier['occurrenceyear'].value_counts()
```




    2014.0    2399
    2015.0    2222
    2017.0    2204
    2018.0    2108
    2016.0    2097
    2019.0    1823
    Name: occurrenceyear, dtype: int64




```python
# Checking data
df_outlier['occurrencemonth'].hist(figsize = (4,4))
plt.title('Month')
```




    Text(0.5, 1.0, 'Month')




    
![png](output_42_1.png)
    



```python
# Checking data
df_outlier['occurrenceday'].hist(figsize = (4,4))
plt.title('Day')
```




    Text(0.5, 1.0, 'Day')




    
![png](output_43_1.png)
    



```python
df_outlier['occurrencehour'].value_counts()
```




    12    1037
    10     837
    13     832
    9      817
    14     806
    8      794
    11     771
    18     736
    19     617
    0      600
    17     586
    15     575
    16     556
    20     511
    21     425
    7      399
    22     372
    23     321
    1      275
    2      248
    3      215
    6      190
    4      183
    5      150
    Name: occurrencehour, dtype: int64




```python
# Checking data
df_outlier['occurrencehour'].hist(figsize = (4,4))
plt.title('Hour')
```




    Text(0.5, 1.0, 'Hour')




    
![png](output_45_1.png)
    



```python
# Make sure data needed has been converted to numerical data
df_outlier.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12853 entries, 16 to 43284
    Data columns (total 9 columns):
    occurrencedate         12853 non-null object
    premisetype            12853 non-null object
    occurrenceyear         12853 non-null float64
    occurrencemonth        12853 non-null int64
    occurrenceday          12853 non-null float64
    occurrencedayofyear    12853 non-null float64
    occurrencedayofweek    12853 non-null object
    occurrencehour         12853 non-null int64
    Hood_ID                12853 non-null int64
    dtypes: float64(3), int64(3), object(3)
    memory usage: 1004.1+ KB



```python
# Check data qualities
df_outlier.describe()
```




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
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>12853.000000</td>
      <td>12853.000000</td>
      <td>12853.000000</td>
      <td>12853.000000</td>
      <td>12853.000000</td>
      <td>12853.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2016.378822</td>
      <td>6.791722</td>
      <td>15.747919</td>
      <td>191.271921</td>
      <td>12.378200</td>
      <td>72.888898</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.697119</td>
      <td>3.497648</td>
      <td>8.648212</td>
      <td>106.816966</td>
      <td>5.935517</td>
      <td>41.594673</td>
    </tr>
    <tr>
      <td>min</td>
      <td>2014.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2015.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>98.000000</td>
      <td>9.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2016.000000</td>
      <td>7.000000</td>
      <td>16.000000</td>
      <td>197.000000</td>
      <td>12.000000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2018.000000</td>
      <td>10.000000</td>
      <td>23.000000</td>
      <td>288.000000</td>
      <td>17.000000</td>
      <td>116.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2019.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
      <td>366.000000</td>
      <td>23.000000</td>
      <td>140.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make dummy variables
dummy_week_day = pd.get_dummies(df_outlier['occurrencedayofweek'])
```


```python
dummy_week_day.head(5)
```




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
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Added two tables together
df_concat = pd.concat([df_outlier,dummy_week_day], axis = 1)
df_concat.head(5)
```




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
      <th>occurrencedate</th>
      <th>premisetype</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Hood_ID</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014/01/06 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014/01/14 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>119</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014/01/31 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014/01/30 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014/02/15 05:00:00+00</td>
      <td>House</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>103</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_concat.drop(['occurrencedate','premisetype','Hood_ID'], axis = 1, inplace = True)
df_concat.head(5)
```




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
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Created column to count break and enter occurrences when next team member uses .groupby
# Will allow to group and count theft occurrences for different dates and times
df_concat['occurr_count'] = 1
df_concat.head(5)
```




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
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
      <th>occurrencedayofyear</th>
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>Friday</th>
      <th>Monday</th>
      <th>Saturday</th>
      <th>Sunday</th>
      <th>Thursday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>occurr_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16</td>
      <td>2014.0</td>
      <td>1</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>Monday</td>
      <td>22</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>21</td>
      <td>2014.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>14.0</td>
      <td>Tuesday</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23</td>
      <td>2014.0</td>
      <td>1</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>Friday</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>25</td>
      <td>2014.0</td>
      <td>1</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>Thursday</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>31</td>
      <td>2014.0</td>
      <td>2</td>
      <td>15.0</td>
      <td>46.0</td>
      <td>Saturday</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_concat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 12853 entries, 16 to 43284
    Data columns (total 14 columns):
    occurrenceyear         12853 non-null float64
    occurrencemonth        12853 non-null int64
    occurrenceday          12853 non-null float64
    occurrencedayofyear    12853 non-null float64
    occurrencedayofweek    12853 non-null object
    occurrencehour         12853 non-null int64
    Friday                 12853 non-null uint8
    Monday                 12853 non-null uint8
    Saturday               12853 non-null uint8
    Sunday                 12853 non-null uint8
    Thursday               12853 non-null uint8
    Tuesday                12853 non-null uint8
    Wednesday              12853 non-null uint8
    occurr_count           12853 non-null int64
    dtypes: float64(3), int64(3), object(1), uint8(7)
    memory usage: 891.2+ KB



```python
df_outlier.to_csv('GP_No_DummyList.csv')
```


```python
df_concat.to_csv('GP_DummyList.csv')
```


```python

```
