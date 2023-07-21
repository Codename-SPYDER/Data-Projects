```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
#from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
% matplotlib inline
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
```


```python

#import data
heart = pd.read_csv("Heart.csv")
```

# Data Preparation


```python
heart.head(7)
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
      <th>Patient_Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPain</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>Oldpeak</th>
      <th>Slope</th>
      <th>Ca</th>
      <th>Thal</th>
      <th>AHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>typical</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>fixed</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>normal</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>reversable</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37</td>
      <td>1</td>
      <td>nonanginal</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>41</td>
      <td>0</td>
      <td>nontypical</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>56</td>
      <td>1</td>
      <td>nontypical</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>62</td>
      <td>0</td>
      <td>asymptomatic</td>
      <td>140</td>
      <td>268</td>
      <td>0</td>
      <td>2</td>
      <td>160</td>
      <td>0</td>
      <td>3.6</td>
      <td>3</td>
      <td>2.0</td>
      <td>normal</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.countplot(x="AHD", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da9e118860>




    
![png](output_4_1.png)
    



```python
sns.countplot(x="AHD", hue="Sex", data= heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da9e343cf8>




    
![png](output_5_1.png)
    



```python
sns.countplot(x="AHD", hue="ChestPain", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da892db470>




    
![png](output_6_1.png)
    



```python
heart["Age"].plot.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da89331240>




    
![png](output_7_1.png)
    



```python
heart["Age"].plot.hist(bins=20, figsize=(15,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da9e38e208>




    
![png](output_8_1.png)
    



```python
heart.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 15 columns):
    Patient_Id    303 non-null int64
    Age           303 non-null int64
    Sex           303 non-null int64
    ChestPain     303 non-null object
    RestBP        303 non-null int64
    Chol          303 non-null int64
    Fbs           303 non-null int64
    RestECG       303 non-null int64
    MaxHR         303 non-null int64
    ExAng         303 non-null int64
    Oldpeak       303 non-null float64
    Slope         303 non-null int64
    Ca            299 non-null float64
    Thal          301 non-null object
    AHD           303 non-null object
    dtypes: float64(2), int64(10), object(3)
    memory usage: 35.6+ KB



```python
heart.isnull().sum()
```




    Patient_Id    0
    Age           0
    Sex           0
    ChestPain     0
    RestBP        0
    Chol          0
    Fbs           0
    RestECG       0
    MaxHR         0
    ExAng         0
    Oldpeak       0
    Slope         0
    Ca            4
    Thal          2
    AHD           0
    dtype: int64




```python
sns.boxplot(x="AHD", y="Age", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da895d3940>




    
![png](output_11_1.png)
    



```python
sns.countplot(x="Ca", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da8946d588>




    
![png](output_12_1.png)
    



```python
heart2 = heart.dropna()
```


```python
heart2.isnull().sum()
```




    Patient_Id    0
    Age           0
    Sex           0
    ChestPain     0
    RestBP        0
    Chol          0
    Fbs           0
    RestECG       0
    MaxHR         0
    ExAng         0
    Oldpeak       0
    Slope         0
    Ca            0
    Thal          0
    AHD           0
    dtype: int64




```python
sns.countplot(x="Thal", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da89502dd8>




    
![png](output_15_1.png)
    



```python
sns.countplot(x="ChestPain", data=heart)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da8957a080>




    
![png](output_16_1.png)
    



```python
pd.get_dummies(heart['ChestPain'])
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
      <th>asymptomatic</th>
      <th>nonanginal</th>
      <th>nontypical</th>
      <th>typical</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>273</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>275</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>276</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>277</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>278</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>279</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>280</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>281</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>282</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>283</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>285</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>286</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>287</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>289</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>290</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>292</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>293</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>294</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>295</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>296</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>302</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>303 rows × 4 columns</p>
</div>




```python
ChestPain_mv = pd.get_dummies(heart['ChestPain'], prefix="ChestPain", prefix_sep ='_')
```


```python
heart.head(5)
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
      <th>Patient_Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPain</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>Oldpeak</th>
      <th>Slope</th>
      <th>Ca</th>
      <th>Thal</th>
      <th>AHD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>typical</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>fixed</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>normal</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>reversable</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37</td>
      <td>1</td>
      <td>nonanginal</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>41</td>
      <td>0</td>
      <td>nontypical</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
Thal_mv = pd.get_dummies(heart['Thal'], prefix="Thal", prefix_sep ='_')
```


```python
heart_mv = pd.concat([heart, ChestPain_mv, Thal_mv], axis=1)
```


```python
heart_mv.head(7)
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
      <th>Patient_Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>ChestPain</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>...</th>
      <th>Ca</th>
      <th>Thal</th>
      <th>AHD</th>
      <th>ChestPain_asymptomatic</th>
      <th>ChestPain_nonanginal</th>
      <th>ChestPain_nontypical</th>
      <th>ChestPain_typical</th>
      <th>Thal_fixed</th>
      <th>Thal_normal</th>
      <th>Thal_reversable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>typical</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>fixed</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>...</td>
      <td>3.0</td>
      <td>normal</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>67</td>
      <td>1</td>
      <td>asymptomatic</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>...</td>
      <td>2.0</td>
      <td>reversable</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37</td>
      <td>1</td>
      <td>nonanginal</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>41</td>
      <td>0</td>
      <td>nontypical</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>56</td>
      <td>1</td>
      <td>nontypical</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>178</td>
      <td>0</td>
      <td>...</td>
      <td>0.0</td>
      <td>normal</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>62</td>
      <td>0</td>
      <td>asymptomatic</td>
      <td>140</td>
      <td>268</td>
      <td>0</td>
      <td>2</td>
      <td>160</td>
      <td>0</td>
      <td>...</td>
      <td>2.0</td>
      <td>normal</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7 rows × 22 columns</p>
</div>




```python
heart_mv.drop(['ChestPain', 'Thal'],axis=1,inplace=True)
```


```python
heart_mv.head(7)
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
      <th>Patient_Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>Oldpeak</th>
      <th>Slope</th>
      <th>Ca</th>
      <th>AHD</th>
      <th>ChestPain_asymptomatic</th>
      <th>ChestPain_nonanginal</th>
      <th>ChestPain_nontypical</th>
      <th>ChestPain_typical</th>
      <th>Thal_fixed</th>
      <th>Thal_normal</th>
      <th>Thal_reversable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>67</td>
      <td>1</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>67</td>
      <td>1</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37</td>
      <td>1</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>41</td>
      <td>0</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>56</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>62</td>
      <td>0</td>
      <td>140</td>
      <td>268</td>
      <td>0</td>
      <td>2</td>
      <td>160</td>
      <td>0</td>
      <td>3.6</td>
      <td>3</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
AHD_dv = heart['AHD']
```


```python
AHD_dv.head(7)
```




    0     No
    1    Yes
    2    Yes
    3     No
    4     No
    5     No
    6    Yes
    Name: AHD, dtype: object




```python
#heart_mv = pd.concat([heart_mv, AHD_dv], axis=1)
```


```python
heart_mv.head(7)
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
      <th>Patient_Id</th>
      <th>Age</th>
      <th>Sex</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>Oldpeak</th>
      <th>Slope</th>
      <th>Ca</th>
      <th>AHD</th>
      <th>ChestPain_asymptomatic</th>
      <th>ChestPain_nonanginal</th>
      <th>ChestPain_nontypical</th>
      <th>ChestPain_typical</th>
      <th>Thal_fixed</th>
      <th>Thal_normal</th>
      <th>Thal_reversable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>67</td>
      <td>1</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>67</td>
      <td>1</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>37</td>
      <td>1</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>41</td>
      <td>0</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>56</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>0</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>1</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>62</td>
      <td>0</td>
      <td>140</td>
      <td>268</td>
      <td>0</td>
      <td>2</td>
      <td>160</td>
      <td>0</td>
      <td>3.6</td>
      <td>3</td>
      <td>2.0</td>
      <td>Yes</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#heart_mv.drop(['Patient_Id', 'AHD'],axis=1,inplace=True)
heart_mv.drop(['Patient_Id'],axis=1,inplace=True)
```


```python
heart_mv.isnull().sum()
```




    Age                       0
    Sex                       0
    RestBP                    0
    Chol                      0
    Fbs                       0
    RestECG                   0
    MaxHR                     0
    ExAng                     0
    Oldpeak                   0
    Slope                     0
    Ca                        4
    AHD                       0
    ChestPain_asymptomatic    0
    ChestPain_nonanginal      0
    ChestPain_nontypical      0
    ChestPain_typical         0
    Thal_fixed                0
    Thal_normal               0
    Thal_reversable           0
    dtype: int64




```python
heart_mv2 = heart_mv.dropna()
```


```python
heart_mv2.isnull().sum()
```




    Age                       0
    Sex                       0
    RestBP                    0
    Chol                      0
    Fbs                       0
    RestECG                   0
    MaxHR                     0
    ExAng                     0
    Oldpeak                   0
    Slope                     0
    Ca                        0
    AHD                       0
    ChestPain_asymptomatic    0
    ChestPain_nonanginal      0
    ChestPain_nontypical      0
    ChestPain_typical         0
    Thal_fixed                0
    Thal_normal               0
    Thal_reversable           0
    dtype: int64




```python
sns.pairplot(heart_mv, hue='AHD')
```

    C:\Users\zazem\Anaconda3\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    C:\Users\zazem\Anaconda3\lib\site-packages\statsmodels\nonparametric\kde.py:448: RuntimeWarning: invalid value encountered in greater
      X = X[np.logical_and(X > clip[0], X < clip[1])] # won't work for two columns.
    C:\Users\zazem\Anaconda3\lib\site-packages\statsmodels\nonparametric\kde.py:448: RuntimeWarning: invalid value encountered in less
      X = X[np.logical_and(X > clip[0], X < clip[1])] # won't work for two columns.





    <seaborn.axisgrid.PairGrid at 0x1da8a4917f0>




    
![png](output_33_2.png)
    



```python
a4_dims = (15, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(heart.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=3, linecolor='black',ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1da967faeb8>




    
![png](output_34_1.png)
    


# Partition Dataset into: Training & Validation Subsets.


```python
X = heart_mv2.drop("AHD", axis=1)
Y = heart_mv2["AHD"]
```


```python
X.head(5)
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
      <th>Age</th>
      <th>Sex</th>
      <th>RestBP</th>
      <th>Chol</th>
      <th>Fbs</th>
      <th>RestECG</th>
      <th>MaxHR</th>
      <th>ExAng</th>
      <th>Oldpeak</th>
      <th>Slope</th>
      <th>Ca</th>
      <th>ChestPain_asymptomatic</th>
      <th>ChestPain_nonanginal</th>
      <th>ChestPain_nontypical</th>
      <th>ChestPain_typical</th>
      <th>Thal_fixed</th>
      <th>Thal_normal</th>
      <th>Thal_reversable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>2</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>1</td>
      <td>160</td>
      <td>286</td>
      <td>0</td>
      <td>2</td>
      <td>108</td>
      <td>1</td>
      <td>1.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67</td>
      <td>1</td>
      <td>120</td>
      <td>229</td>
      <td>0</td>
      <td>2</td>
      <td>129</td>
      <td>1</td>
      <td>2.6</td>
      <td>2</td>
      <td>2.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>1</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>0</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41</td>
      <td>0</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>2</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
Y.head(5)
```




    0     No
    1    Yes
    2    Yes
    3     No
    4     No
    Name: AHD, dtype: object




```python
X_tr, X_val, Y_tr, Y_val = train_test_split(X,Y, test_size=.23, random_state=77)
```

# Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
```


```python
# create an instance of your object.... ie an instance of a logistic regression model.
model_LogReg = LogisticRegression()
```


```python
model_LogReg.fit(X_tr, Y_tr)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



The output above outlines the various parameters or dials available to you to modify how the model is built.  Experiment with different settings to generate different models using the same data and the same algorithm.


```python
model_LogReg_predict = model_LogReg.predict(X_val)
```

Evaluate the quality of predictions using the classification report function.


```python
from sklearn.metrics import classification_report
```


```python
classification_report(Y_val, model_LogReg_predict)

```




    '             precision    recall  f1-score   support\n\n         No       0.83      0.89      0.86        38\n        Yes       0.86      0.77      0.81        31\n\navg / total       0.84      0.84      0.84        69\n'




```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(Y_val, model_LogReg_predict)
```




    array([[34,  4],
           [ 7, 24]], dtype=int64)




```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(Y_val, model_LogReg_predict)*100
```




    84.05797101449275



Consider transforming some variables if warranted to produce new models.


```python
from sklearn.preprocessing import StandardScaler
```


```python
sc = StandardScaler()
X_tr2 = sc.fit_transform(X_tr)
X_val2 = sc.fit_transform(X_val)
```


```python
model_LogReg2 = LogisticRegression()
model_LogReg2.fit(X_tr2, Y_tr)
model_LogReg2_predict = model_LogReg2.predict(X_val2)
accuracy_score(Y_val, model_LogReg2_predict)*100
```




    84.05797101449275



Notice that the overall accuracy of model_LogReg and model_LogReg2 is identical.  That means in this scenario transforming the data had no impact on model performance.  That makes sense because the variables used as inputs for model_LogReg didn't have drastically different ranges.


```python
confusion_matrix(Y_val, model_LogReg_predict)
```




    array([[34,  4],
           [ 7, 24]], dtype=int64)



# Decision Tree


```python
# Import Decision-Tree related libraries 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, roc_curve



```


```python
model_dt1 = DecisionTreeClassifier(criterion='entropy', random_state=100,
                                  max_depth=7, min_samples_leaf=25)
```


```python
model_dt1.fit(X_tr, Y_tr)
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=7,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=25, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=100,
                splitter='best')




```python
# Decision Tree parameters displayed above provide you with opportunities to experiment 
#with different model variations.
```


```python
model_dt1_predict = model_dt1.predict(X_val)
```


```python
model_dt1_predict
```




    array(['No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No',
           'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
           'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes',
           'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No',
           'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes',
           'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No',
           'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
          dtype=object)




```python
accuracy_score(Y_val, model_dt1_predict)*100
```




    71.01449275362319




```python
confusion_matrix(Y_val, model_dt1_predict)
```




    array([[26, 12],
           [ 8, 23]], dtype=int64)



Decision Tree Visualization


```python
model_dt1
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=7,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=25, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=100,
                splitter='best')




```python

```

# Random Forest


```python

```


```python

```


```python
from sklearn.ensemble import RandomForestClassifier
```

    C:\Users\zazem\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d



```python
print(dir(RandomForestClassifier))
```

    ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_abc_impl', '_estimator_type', '_get_param_names', '_make_estimator', '_set_oob_score', '_validate_X_predict', '_validate_estimator', '_validate_y_class_weight', 'apply', 'decision_path', 'feature_importances_', 'fit', 'get_params', 'predict', 'predict_log_proba', 'predict_proba', 'score', 'set_params']



```python
print(RandomForestClassifier())
```

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



```python
from sklearn.model_selection import KFold, cross_val_score
```


```python
model_rf1 = RandomForestClassifier(n_estimators=200, random_state = 0)
```


```python
y_tr_array = np.ravel(Y_tr)
```


```python
model_rf1.fit(X_tr, y_tr_array)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)




```python
model_rf1_predict = model_rf1.predict(X_val)
```


```python
classification_report(Y_val, model_rf1_predict)
```




    '             precision    recall  f1-score   support\n\n         No       0.80      0.84      0.82        38\n        Yes       0.79      0.74      0.77        31\n\navg / total       0.80      0.80      0.80        69\n'




```python
confusion_matrix(Y_val, model_rf1_predict)
```




    array([[32,  6],
           [ 8, 23]], dtype=int64)




```python
accuracy_score(Y_val, model_rf1_predict)*100
```




    79.71014492753623




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
