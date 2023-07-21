```python
# Import needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# Import auto theft dataset
Police = pd.read_csv('./Basic_Methods-(Data)/Lab2/Auto_Theft_2014_to_2019.csv',encoding = 'unicode_escape')
```


```python
# Check to see how many rows & columns
# Check to see data types
Police.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23380 entries, 0 to 23379
    Data columns (total 29 columns):
    X                      23380 non-null float64
    Y                      23380 non-null float64
    Index_                 23380 non-null int64
    event_unique_id        23380 non-null object
    occurrencedate         23380 non-null object
    reporteddate           23380 non-null object
    premisetype            23380 non-null object
    ucr_code               23380 non-null int64
    ucr_ext                23380 non-null int64
    offence                23380 non-null object
    reportedyear           23380 non-null int64
    reportedmonth          23380 non-null object
    reportedday            23380 non-null int64
    reporteddayofyear      23380 non-null int64
    reporteddayofweek      23380 non-null object
    reportedhour           23380 non-null int64
    occurrenceyear         23377 non-null float64
    occurrencemonth        23377 non-null object
    occurrenceday          23377 non-null float64
    occurrencedayofyear    23377 non-null float64
    occurrencedayofweek    23377 non-null object
    occurrencehour         23380 non-null int64
    MCI                    23380 non-null object
    Division               23380 non-null object
    Hood_ID                23380 non-null int64
    Neighbourhood          23380 non-null object
    Lat                    23380 non-null float64
    Long                   23380 non-null float64
    ObjectId               23380 non-null int64
    dtypes: float64(7), int64(10), object(12)
    memory usage: 5.2+ MB



```python
# Check to see data descriptive statistics
Police.describe()
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
      <td>2.338000e+04</td>
      <td>2.338000e+04</td>
      <td>23380.000000</td>
      <td>23380.0</td>
      <td>23380.0</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23377.000000</td>
      <td>23377.000000</td>
      <td>23377.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
      <td>23380.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>-8.842468e+06</td>
      <td>5.421871e+06</td>
      <td>139836.194696</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2016.785329</td>
      <td>15.639179</td>
      <td>191.357314</td>
      <td>11.632849</td>
      <td>2016.769731</td>
      <td>15.606750</td>
      <td>190.598409</td>
      <td>14.170616</td>
      <td>58.762618</td>
      <td>43.717755</td>
      <td>-79.433239</td>
      <td>11690.500000</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.287636e+04</td>
      <td>7.911834e+03</td>
      <td>58722.388374</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.752191</td>
      <td>8.709075</td>
      <td>103.619117</td>
      <td>5.557112</td>
      <td>1.772063</td>
      <td>8.776699</td>
      <td>103.556662</td>
      <td>7.357208</td>
      <td>45.683707</td>
      <td>0.051370</td>
      <td>0.115670</td>
      <td>6749.368983</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-8.865373e+06</td>
      <td>5.401805e+06</td>
      <td>57370.000000</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2014.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>43.587353</td>
      <td>-79.639000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>-8.853044e+06</td>
      <td>5.415802e+06</td>
      <td>63214.750000</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2015.000000</td>
      <td>8.000000</td>
      <td>102.000000</td>
      <td>8.000000</td>
      <td>2015.000000</td>
      <td>8.000000</td>
      <td>102.000000</td>
      <td>9.000000</td>
      <td>21.000000</td>
      <td>43.678360</td>
      <td>-79.528250</td>
      <td>5845.750000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>-8.843666e+06</td>
      <td>5.422107e+06</td>
      <td>129399.500000</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>197.000000</td>
      <td>11.000000</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>195.000000</td>
      <td>17.000000</td>
      <td>46.000000</td>
      <td>43.719307</td>
      <td>-79.444000</td>
      <td>11690.500000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>-8.832756e+06</td>
      <td>5.427931e+06</td>
      <td>200590.250000</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2018.000000</td>
      <td>23.000000</td>
      <td>283.000000</td>
      <td>16.000000</td>
      <td>2018.000000</td>
      <td>23.000000</td>
      <td>282.000000</td>
      <td>20.000000</td>
      <td>104.000000</td>
      <td>43.757109</td>
      <td>-79.346000</td>
      <td>17535.250000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>-8.808155e+06</td>
      <td>5.442380e+06</td>
      <td>206435.000000</td>
      <td>2135.0</td>
      <td>210.0</td>
      <td>2019.000000</td>
      <td>31.000000</td>
      <td>366.000000</td>
      <td>23.000000</td>
      <td>2019.000000</td>
      <td>31.000000</td>
      <td>366.000000</td>
      <td>23.000000</td>
      <td>140.000000</td>
      <td>43.850788</td>
      <td>-79.125000</td>
      <td>23380.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# See how actual dataset looks
Police.head(5)
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
      <th>event_unique_id</th>
      <th>occurrencedate</th>
      <th>reporteddate</th>
      <th>premisetype</th>
      <th>ucr_code</th>
      <th>ucr_ext</th>
      <th>offence</th>
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
      <td>0</td>
      <td>-8.858137e+06</td>
      <td>5.428285e+06</td>
      <td>58170</td>
      <td>GO-20141673399</td>
      <td>2014/03/09 05:00:00+00</td>
      <td>2014/03/10 04:00:00+00</td>
      <td>House</td>
      <td>2135</td>
      <td>210</td>
      <td>Theft Of Motor Vehicle</td>
      <td>...</td>
      <td>68.0</td>
      <td>Sunday</td>
      <td>17</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>21</td>
      <td>Humber Summit (21)</td>
      <td>43.759407</td>
      <td>-79.574</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>-8.851792e+06</td>
      <td>5.427584e+06</td>
      <td>58171</td>
      <td>GO-20142467943</td>
      <td>2014/07/10 04:00:00+00</td>
      <td>2014/07/10 04:00:00+00</td>
      <td>Outside</td>
      <td>2135</td>
      <td>210</td>
      <td>Theft Of Motor Vehicle</td>
      <td>...</td>
      <td>191.0</td>
      <td>Thursday</td>
      <td>9</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>25</td>
      <td>Glenfield-Jane Heights (25)</td>
      <td>43.754856</td>
      <td>-79.517</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-8.848230e+06</td>
      <td>5.428533e+06</td>
      <td>58172</td>
      <td>GO-20142693275</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2014/08/13 04:00:00+00</td>
      <td>Commercial</td>
      <td>2135</td>
      <td>210</td>
      <td>Theft Of Motor Vehicle</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>0</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.761013</td>
      <td>-79.485</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>-8.859473e+06</td>
      <td>5.424140e+06</td>
      <td>58173</td>
      <td>GO-20142530993</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2014/07/19 04:00:00+00</td>
      <td>Outside</td>
      <td>2135</td>
      <td>210</td>
      <td>Theft Of Motor Vehicle</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>Auto Theft</td>
      <td>D23</td>
      <td>2</td>
      <td>Mount Olive-Silverstone-Jamestown (2)</td>
      <td>43.732510</td>
      <td>-79.586</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>-8.841105e+06</td>
      <td>5.412458e+06</td>
      <td>58174</td>
      <td>GO-20142520781</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2014/07/18 04:00:00+00</td>
      <td>Outside</td>
      <td>2135</td>
      <td>210</td>
      <td>Theft Of Motor Vehicle</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>Auto Theft</td>
      <td>D14</td>
      <td>80</td>
      <td>Palmerston-Little Italy (80)</td>
      <td>43.656632</td>
      <td>-79.421</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>




```python
# Drop unecessary columns
Police.drop(['X','Y','Index_','event_unique_id','reporteddate','premisetype','ucr_ext','offence'], axis = 1, inplace = True)
Police.head(5)
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
      <th>ucr_code</th>
      <th>reportedyear</th>
      <th>reportedmonth</th>
      <th>reportedday</th>
      <th>reporteddayofyear</th>
      <th>reporteddayofweek</th>
      <th>reportedhour</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
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
      <td>0</td>
      <td>2014/03/09 05:00:00+00</td>
      <td>2135</td>
      <td>2014</td>
      <td>March</td>
      <td>10</td>
      <td>69</td>
      <td>Monday</td>
      <td>8</td>
      <td>2014.0</td>
      <td>March</td>
      <td>...</td>
      <td>68.0</td>
      <td>Sunday</td>
      <td>17</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>21</td>
      <td>Humber Summit (21)</td>
      <td>43.759407</td>
      <td>-79.574</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014/07/10 04:00:00+00</td>
      <td>2135</td>
      <td>2014</td>
      <td>July</td>
      <td>10</td>
      <td>191</td>
      <td>Thursday</td>
      <td>9</td>
      <td>2014.0</td>
      <td>July</td>
      <td>...</td>
      <td>191.0</td>
      <td>Thursday</td>
      <td>9</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>25</td>
      <td>Glenfield-Jane Heights (25)</td>
      <td>43.754856</td>
      <td>-79.517</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2135</td>
      <td>2014</td>
      <td>August</td>
      <td>13</td>
      <td>225</td>
      <td>Wednesday</td>
      <td>9</td>
      <td>2014.0</td>
      <td>July</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>0</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.761013</td>
      <td>-79.485</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2135</td>
      <td>2014</td>
      <td>July</td>
      <td>19</td>
      <td>200</td>
      <td>Saturday</td>
      <td>19</td>
      <td>2014.0</td>
      <td>July</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>Auto Theft</td>
      <td>D23</td>
      <td>2</td>
      <td>Mount Olive-Silverstone-Jamestown (2)</td>
      <td>43.732510</td>
      <td>-79.586</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>2135</td>
      <td>2014</td>
      <td>July</td>
      <td>18</td>
      <td>199</td>
      <td>Friday</td>
      <td>8</td>
      <td>2014.0</td>
      <td>July</td>
      <td>...</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>Auto Theft</td>
      <td>D14</td>
      <td>80</td>
      <td>Palmerston-Little Italy (80)</td>
      <td>43.656632</td>
      <td>-79.421</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
Police.drop(['ucr_code','reportedyear','reportedmonth','reporteddayofyear','reporteddayofweek'], axis = 1, inplace = True)
Police.head(5)
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
      <th>reportedday</th>
      <th>reportedhour</th>
      <th>occurrenceyear</th>
      <th>occurrencemonth</th>
      <th>occurrenceday</th>
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
      <td>0</td>
      <td>2014/03/09 05:00:00+00</td>
      <td>10</td>
      <td>8</td>
      <td>2014.0</td>
      <td>March</td>
      <td>9.0</td>
      <td>68.0</td>
      <td>Sunday</td>
      <td>17</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>21</td>
      <td>Humber Summit (21)</td>
      <td>43.759407</td>
      <td>-79.574</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014/07/10 04:00:00+00</td>
      <td>10</td>
      <td>9</td>
      <td>2014.0</td>
      <td>July</td>
      <td>10.0</td>
      <td>191.0</td>
      <td>Thursday</td>
      <td>9</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>25</td>
      <td>Glenfield-Jane Heights (25)</td>
      <td>43.754856</td>
      <td>-79.517</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>13</td>
      <td>9</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>0</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.761013</td>
      <td>-79.485</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>19</td>
      <td>19</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>Auto Theft</td>
      <td>D23</td>
      <td>2</td>
      <td>Mount Olive-Silverstone-Jamestown (2)</td>
      <td>43.732510</td>
      <td>-79.586</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014/07/17 04:00:00+00</td>
      <td>18</td>
      <td>8</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>198.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>Auto Theft</td>
      <td>D14</td>
      <td>80</td>
      <td>Palmerston-Little Italy (80)</td>
      <td>43.656632</td>
      <td>-79.421</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
Police.drop(['reportedday','reportedhour','occurrencedate','occurrencedayofyear'], axis = 1, inplace = True)
Police.head(5)
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
      <td>0</td>
      <td>2014.0</td>
      <td>March</td>
      <td>9.0</td>
      <td>Sunday</td>
      <td>17</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>21</td>
      <td>Humber Summit (21)</td>
      <td>43.759407</td>
      <td>-79.574</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014.0</td>
      <td>July</td>
      <td>10.0</td>
      <td>Thursday</td>
      <td>9</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>25</td>
      <td>Glenfield-Jane Heights (25)</td>
      <td>43.754856</td>
      <td>-79.517</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>0</td>
      <td>Auto Theft</td>
      <td>D31</td>
      <td>27</td>
      <td>York University Heights (27)</td>
      <td>43.761013</td>
      <td>-79.485</td>
      <td>3</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>Auto Theft</td>
      <td>D23</td>
      <td>2</td>
      <td>Mount Olive-Silverstone-Jamestown (2)</td>
      <td>43.732510</td>
      <td>-79.586</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>Auto Theft</td>
      <td>D14</td>
      <td>80</td>
      <td>Palmerston-Little Italy (80)</td>
      <td>43.656632</td>
      <td>-79.421</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
Police.drop(['MCI','Division','Hood_ID','Neighbourhood','Lat','Long','ObjectId'], axis = 1, inplace = True)
Police.head(5)
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014.0</td>
      <td>March</td>
      <td>9.0</td>
      <td>Sunday</td>
      <td>17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014.0</td>
      <td>July</td>
      <td>10.0</td>
      <td>Thursday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014.0</td>
      <td>July</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check descriptive statistics for new dataset
Police.describe()
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
      <th>occurrenceday</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>23377.000000</td>
      <td>23377.000000</td>
      <td>23380.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2016.769731</td>
      <td>15.606750</td>
      <td>14.170616</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.772063</td>
      <td>8.776699</td>
      <td>7.357208</td>
    </tr>
    <tr>
      <td>min</td>
      <td>2000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2015.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>2017.000000</td>
      <td>16.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>2018.000000</td>
      <td>23.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2019.000000</td>
      <td>31.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change months to numerical data
Police['occurrencemonth'].replace({"July": 7}, inplace=True)
Police.head(5)

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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014.0</td>
      <td>March</td>
      <td>9.0</td>
      <td>Sunday</td>
      <td>17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014.0</td>
      <td>7</td>
      <td>10.0</td>
      <td>Thursday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014.0</td>
      <td>7</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014.0</td>
      <td>7</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014.0</td>
      <td>7</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
Police['occurrencemonth'].replace({'January': 1,'February':2,'March':3,'April':4,'May':5 }, inplace=True)
```


```python
Police['occurrencemonth'].replace({'June': 6,'August':8,'September':9,'October':10,'November':11,'December':12 }, inplace=True)
Police.head(10)
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>Sunday</td>
      <td>17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>Thursday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2014.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>Sunday</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2013.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>Wednesday</td>
      <td>12</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2013.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>Sunday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2014.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>Sunday</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change day of week to numerical data
# For some reason this line of code was not working
    # Could not figure out why
Police['occurrencedayofweek'].replace({'Thursday':4}, inplace=True)
Police.head(10)
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2014.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>Sunday</td>
      <td>17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>10.0</td>
      <td>Thursday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2014.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>Sunday</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2013.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>Wednesday</td>
      <td>12</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2013.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>Sunday</td>
      <td>9</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2014.0</td>
      <td>7.0</td>
      <td>17.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2014.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>Sunday</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check to see if months have been successfully changed
Police['occurrencemonth'].value_counts()
```




    10.0    2279
    11.0    2234
    7.0     2091
    8.0     2085
    9.0     1992
    6.0     1981
    12.0    1908
    5.0     1892
    3.0     1868
    4.0     1782
    1.0     1713
    2.0     1552
    Name: occurrencemonth, dtype: int64




```python
# Check for null values
Police.isnull().sum()
```




    occurrenceyear         3
    occurrencemonth        3
    occurrenceday          3
    occurrencedayofweek    3
    occurrencehour         0
    dtype: int64




```python
# Check for zeros
Police[Police==0].count(axis=0, level=None, numeric_only=False)
```




    occurrenceyear            0
    occurrencemonth           0
    occurrenceday             0
    occurrencedayofweek       0
    occurrencehour         1530
    dtype: int64




```python
# large number of zeros in 'occurrencehour' just means auto theft happened
# at 24:00
Police['occurrencehour'].value_counts()
```




    22    2111
    21    1885
    23    1837
    20    1718
    0     1530
    19    1508
    18    1457
    17    1179
    12     998
    16     947
    1      748
    15     730
    14     705
    8      689
    9      685
    7      658
    10     631
    13     617
    2      573
    11     533
    6      460
    3      436
    5      383
    4      362
    Name: occurrencehour, dtype: int64




```python
# Check if there null values are consistent across rows
nan_values = Police[Police['occurrencedayofweek'].isna()]
print(nan_values)
```

          occurrenceyear  occurrencemonth  occurrenceday occurrencedayofweek  \
    56               NaN              NaN            NaN                 NaN   
    6302             NaN              NaN            NaN                 NaN   
    6303             NaN              NaN            NaN                 NaN   
    
          occurrencehour  
    56                 0  
    6302               0  
    6303               0  



```python
# Drop null values
Police.dropna(subset = ['occurrencedayofweek'], inplace=True)
```


```python
# No null values found
Police.isnull().sum()
```




    occurrenceyear         0
    occurrencemonth        0
    occurrenceday          0
    occurrencedayofweek    0
    occurrencehour         0
    dtype: int64




```python
#Police['occurrenceyear'] = pd.to_datetime(Police['occurrenceyear'])
#Police.set_index('occurrenceyear', inplace=True)
```


```python
# Check for abnormalities in numbers
Police['occurrenceyear'].hist(figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116a12d10>




    
![png](output_22_1.png)
    



```python
# Chose to use last 5 years of data
Police['occurrenceyear'].value_counts()
```




    2019.0    5137
    2018.0    4710
    2017.0    3545
    2014.0    3485
    2016.0    3258
    2015.0    3195
    2013.0      32
    2010.0       5
    2008.0       3
    2012.0       2
    2000.0       2
    2011.0       1
    2009.0       1
    2001.0       1
    Name: occurrenceyear, dtype: int64




```python
# Chose to use last 5 years of data
Police_outlier = Police[Police['occurrenceyear'].isin([2015,2016,2017,2018,2019])]
```


```python
Police_outlier['occurrenceyear'].value_counts()
```




    2019.0    5137
    2018.0    4710
    2017.0    3545
    2016.0    3258
    2015.0    3195
    Name: occurrenceyear, dtype: int64




```python
Police_outlier['occurrenceyear'].sort_values(ascending = True)
```




    2017     2015.0
    5642     2015.0
    5643     2015.0
    5644     2015.0
    5645     2015.0
              ...  
    19283    2019.0
    19282    2019.0
    19281    2019.0
    19290    2019.0
    23379    2019.0
    Name: occurrenceyear, Length: 19845, dtype: float64




```python
Police_outlier
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2017</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>Thursday</td>
      <td>22</td>
    </tr>
    <tr>
      <td>2106</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>Wednesday</td>
      <td>19</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>2109</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>2115</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>Friday</td>
      <td>6</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>23375</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>17</td>
    </tr>
    <tr>
      <td>23376</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>2</td>
    </tr>
    <tr>
      <td>23377</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>4</td>
    </tr>
    <tr>
      <td>23378</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>23</td>
    </tr>
    <tr>
      <td>23379</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
<p>19845 rows × 5 columns</p>
</div>




```python
# Checking data
Police['occurrencemonth'].hist(figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b26f690>




    
![png](output_28_1.png)
    



```python
# Checking data
Police['occurrenceday'].hist(figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b26f250>




    
![png](output_29_1.png)
    



```python
# Checking data
Police['occurrencehour'].hist(figsize = (4,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11d23b5d0>




    
![png](output_30_1.png)
    



```python
dummy = pd.get_dummies(Police_outlier['occurrencedayofweek'])
```


```python
# Created dummy variables for 'occurrencedayofweek' since i could not change to numerical value
dummy.head(5)
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
      <td>2017</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2106</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2109</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2115</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Police_outlier.head(5)
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2017</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>Thursday</td>
      <td>22</td>
    </tr>
    <tr>
      <td>2106</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>Wednesday</td>
      <td>19</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>18</td>
    </tr>
    <tr>
      <td>2109</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>21</td>
    </tr>
    <tr>
      <td>2115</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>Friday</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Added two tables together
Police_concat = pd.concat([Police_outlier,dummy], axis = 1)
```


```python
Police_concat
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
      <td>2017</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>Thursday</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2106</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>Wednesday</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2109</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2115</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>Friday</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>23375</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23376</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23377</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>23378</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23379</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>19845 rows × 12 columns</p>
</div>




```python
# Created column to count auto theft occurrences when next team member uses .groupby
# Will allow to group and count theft occurrences for different dates and times
Police_concat['occurr_count'] = 1
```


```python
Police_concat
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
      <td>2017</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>Thursday</td>
      <td>22</td>
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
      <td>2106</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>Wednesday</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>18</td>
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
      <td>2109</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>21</td>
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
      <td>2115</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>Friday</td>
      <td>6</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>23375</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>17</td>
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
      <td>23376</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>2</td>
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
      <td>23377</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>Tuesday</td>
      <td>4</td>
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
      <td>23378</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>23379</td>
      <td>2019.0</td>
      <td>8.0</td>
      <td>7.0</td>
      <td>Wednesday</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>19845 rows × 13 columns</p>
</div>




```python
#Police_concat.to_csv('DummyList.csv')
```


```python
Police_outlier['occurr_count'] = 1
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
Police_outlier.head(5)
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
      <th>occurrencedayofweek</th>
      <th>occurrencehour</th>
      <th>occurr_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2017</td>
      <td>2015.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>Thursday</td>
      <td>22</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2106</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>Wednesday</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2107</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2109</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>Thursday</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2115</td>
      <td>2015.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>Friday</td>
      <td>6</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Police_outlier.to_csv('No_DummyList.csv')
```


```python

```
