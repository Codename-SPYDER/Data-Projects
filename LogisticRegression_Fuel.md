## PART 1: IMPORT DATASET AND BASIC CHECKS


```python
import os
print(os.getcwd())
```

    /Users/siddiqkhan/Documents/Big Data/Basic_Methods/Module_2_Python



```python
#set working directory
os.chdir('/Users/siddiqkhan/Documents/Big Data/Basic_Methods/Module_2_Python') # Provide the path here
```


```python
#we will use pandas for data manipulation
import pandas as pd
data = pd.read_csv('PREMIUM_FUEL.csv')

```


```python
#get the number of rows -
print(len(data.index))
```

    98714



```python
#get number of columns
len(data.columns)
```




    21




```python
#see a sample of data
data.head(5)
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
      <th>TARGET</th>
      <th>ID</th>
      <th>CLIENT_GENDER</th>
      <th>CLIENT_AGE_BAND</th>
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>CLIENT_REGION</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>REWARD_HIST_SEGMENT</th>
      <th>...</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>99105133</td>
      <td>F</td>
      <td>18 TO 24</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>F001</td>
      <td>3.0</td>
      <td>3.00411</td>
      <td>No Redemptions L18 months</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>45.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>8.747049</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>99105134</td>
      <td>M</td>
      <td>65 TO 74</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>L002</td>
      <td>3.0</td>
      <td>13.70595</td>
      <td>Retail Low-Volume Saver</td>
      <td>...</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.75</td>
      <td>0.25</td>
      <td>2.0</td>
      <td>24.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>13.796435</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>99105135</td>
      <td>M</td>
      <td>45 TO 54</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>B002</td>
      <td>2.0</td>
      <td>62.60922</td>
      <td>Retail High-Volume</td>
      <td>...</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>13.087894</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>99105136</td>
      <td>F</td>
      <td>35 TO 44</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>C003</td>
      <td>6.0</td>
      <td>32.83978</td>
      <td>Retail Low-Volume Saver</td>
      <td>...</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>30.82</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0</td>
      <td>18.092770</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>99105137</td>
      <td>M</td>
      <td>65 TO 74</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>D004</td>
      <td>4.0</td>
      <td>12.36114</td>
      <td>Retail Low-Volume Saver</td>
      <td>...</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>78.12</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12.480857</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
#see some basic info about your dataset (similar to str() in R)
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 98714 entries, 0 to 98713
    Data columns (total 21 columns):
    TARGET                        98714 non-null int64
    ID                            98714 non-null int64
    CLIENT_GENDER                 89002 non-null object
    CLIENT_AGE_BAND               88604 non-null object
    NUM_FMLY_MEMBERS              98714 non-null float64
    NUM_CARS_HH                   98714 non-null float64
    CLIENT_REGION                 89547 non-null object
    PARTNERS_SHOPPED              98714 non-null float64
    AVG_WKLY_SPND_ALL_PARTNERS    98714 non-null float64
    REWARD_HIST_SEGMENT           98164 non-null object
    ENDING_PT_BALANCE             98714 non-null int64
    CNT_VISITS_GAS_STATION        98714 non-null float64
    PROP_WKDAY_EVE                98714 non-null float64
    PROP_WKDAY_DAY                98714 non-null float64
    PROP_WKND                     98714 non-null float64
    FUEL_TXNS_L12                 98714 non-null float64
    AVG_FUEL_VOL_L12              98714 non-null float64
    SHOP_TXNS_L12                 98714 non-null float64
    SHOP_AVG_SPEND_L12            98714 non-null float64
    FUEL_CARD_FLAG                98714 non-null int64
    DISTANCE_KM                   98714 non-null float64
    dtypes: float64(13), int64(4), object(4)
    memory usage: 15.8+ MB



```python
data.info('CLIENT_AGE_BAND')
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 98714 entries, 0 to 98713
    Data columns (total 21 columns):
    TARGET                        98714 non-null int64
    ID                            98714 non-null int64
    CLIENT_GENDER                 89002 non-null object
    CLIENT_AGE_BAND               88604 non-null object
    NUM_FMLY_MEMBERS              98714 non-null float64
    NUM_CARS_HH                   98714 non-null float64
    CLIENT_REGION                 89547 non-null object
    PARTNERS_SHOPPED              98714 non-null float64
    AVG_WKLY_SPND_ALL_PARTNERS    98714 non-null float64
    REWARD_HIST_SEGMENT           98164 non-null object
    ENDING_PT_BALANCE             98714 non-null int64
    CNT_VISITS_GAS_STATION        98714 non-null float64
    PROP_WKDAY_EVE                98714 non-null float64
    PROP_WKDAY_DAY                98714 non-null float64
    PROP_WKND                     98714 non-null float64
    FUEL_TXNS_L12                 98714 non-null float64
    AVG_FUEL_VOL_L12              98714 non-null float64
    SHOP_TXNS_L12                 98714 non-null float64
    SHOP_AVG_SPEND_L12            98714 non-null float64
    FUEL_CARD_FLAG                98714 non-null int64
    DISTANCE_KM                   98714 non-null float64
    dtypes: float64(13), int64(4), object(4)
    memory usage: 15.8+ MB



```python
#see some basic summary stats about the data (similar to summary() in R)
data.describe()
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
      <th>TARGET</th>
      <th>ID</th>
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>98714.000000</td>
      <td>9.871400e+04</td>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.208998</td>
      <td>9.915449e+07</td>
      <td>2.894737</td>
      <td>1.728236</td>
      <td>2.963693</td>
      <td>46.167885</td>
      <td>4919.443433</td>
      <td>11.118323</td>
      <td>0.135107</td>
      <td>0.580036</td>
      <td>0.284857</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.037077</td>
      <td>10.033388</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.406595</td>
      <td>2.849642e+04</td>
      <td>1.344243</td>
      <td>0.923183</td>
      <td>1.236660</td>
      <td>81.502148</td>
      <td>11529.172773</td>
      <td>16.817524</td>
      <td>0.228338</td>
      <td>0.343604</td>
      <td>0.307625</td>
      <td>16.846474</td>
      <td>747.989781</td>
      <td>13.335370</td>
      <td>7.600509</td>
      <td>0.188951</td>
      <td>3.689799</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>9.910513e+07</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
      <td>9.912981e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>12.087390</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>35.010000</td>
      <td>1.000000</td>
      <td>2.150000</td>
      <td>0.000000</td>
      <td>9.812254</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
      <td>9.915449e+07</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>29.583220</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.617021</td>
      <td>0.209302</td>
      <td>4.000000</td>
      <td>99.030000</td>
      <td>2.000000</td>
      <td>3.290000</td>
      <td>0.000000</td>
      <td>10.966584</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.000000</td>
      <td>9.917917e+07</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.245590</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>0.200000</td>
      <td>0.904762</td>
      <td>0.433209</td>
      <td>12.000000</td>
      <td>344.015000</td>
      <td>5.000000</td>
      <td>5.092500</td>
      <td>0.000000</td>
      <td>11.713106</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>9.920385e+07</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>15342.317340</td>
      <td>693385.000000</td>
      <td>1012.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>649.000000</td>
      <td>37232.370000</td>
      <td>526.000000</td>
      <td>416.690000</td>
      <td>1.000000</td>
      <td>24.559018</td>
    </tr>
  </tbody>
</table>
</div>




```python
#see number of missing values for each column
#ask the class, if you want % what would you do?
data.isnull().sum()
```




    TARGET                            0
    ID                                0
    CLIENT_GENDER                  9712
    CLIENT_AGE_BAND               10110
    NUM_FMLY_MEMBERS              41790
    NUM_CARS_HH                   46139
    CLIENT_REGION                  9167
    PARTNERS_SHOPPED                248
    AVG_WKLY_SPND_ALL_PARTNERS      249
    REWARD_HIST_SEGMENT             550
    ENDING_PT_BALANCE                 0
    CNT_VISITS_GAS_STATION            1
    PROP_WKDAY_EVE                    0
    PROP_WKDAY_DAY                    0
    PROP_WKND                         0
    FUEL_TXNS_L12                 13883
    AVG_FUEL_VOL_L12              13806
    SHOP_TXNS_L12                 50455
    SHOP_AVG_SPEND_L12            50455
    FUEL_CARD_FLAG                    0
    DISTANCE_KM                       0
    dtype: int64




```python
#check distribution of target variable
data.hist(column='TARGET')
data[['TARGET']].describe()
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
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.208998</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.406595</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](output_11_1.png)
    


## PART 2: DATA EXPLORATION: CATEGORICAL FEATURES


```python
freq_table = pd.Series(data['TARGET']).value_counts()
print(freq_table)
```

    0    78083
    1    20631
    Name: TARGET, dtype: int64



```python
#Next we look at CLIENT_GENDER w.r.t. our Target Variable...
pd.crosstab(data.CLIENT_GENDER, data.TARGET, margins=True)
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
      <th>TARGET</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>CLIENT_GENDER</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>F</td>
      <td>39738</td>
      <td>7743</td>
      <td>47481</td>
    </tr>
    <tr>
      <td>M</td>
      <td>29881</td>
      <td>11640</td>
      <td>41521</td>
    </tr>
    <tr>
      <td>All</td>
      <td>69619</td>
      <td>19383</td>
      <td>89002</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2B. cross-tab for 2 categorical features, this time with percentages
pd.crosstab(data.CLIENT_GENDER, data.TARGET, margins=True, normalize='index')
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
      <th>TARGET</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>CLIENT_GENDER</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>F</td>
      <td>0.836924</td>
      <td>0.163076</td>
    </tr>
    <tr>
      <td>M</td>
      <td>0.719660</td>
      <td>0.280340</td>
    </tr>
    <tr>
      <td>All</td>
      <td>0.782218</td>
      <td>0.217782</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Part 2B: Exploring categorical features w.r.t other Numeric features 


```python
data.groupby('CLIENT_GENDER').mean().T
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
      <th>CLIENT_GENDER</th>
      <th>F</th>
      <th>M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>TARGET</td>
      <td>1.630758e-01</td>
      <td>2.803401e-01</td>
    </tr>
    <tr>
      <td>ID</td>
      <td>9.915462e+07</td>
      <td>9.915424e+07</td>
    </tr>
    <tr>
      <td>NUM_FMLY_MEMBERS</td>
      <td>2.932537e+00</td>
      <td>2.846680e+00</td>
    </tr>
    <tr>
      <td>NUM_CARS_HH</td>
      <td>1.718676e+00</td>
      <td>1.741685e+00</td>
    </tr>
    <tr>
      <td>PARTNERS_SHOPPED</td>
      <td>3.061625e+00</td>
      <td>3.142181e+00</td>
    </tr>
    <tr>
      <td>AVG_WKLY_SPND_ALL_PARTNERS</td>
      <td>4.531798e+01</td>
      <td>5.302445e+01</td>
    </tr>
    <tr>
      <td>ENDING_PT_BALANCE</td>
      <td>4.589083e+03</td>
      <td>6.052430e+03</td>
    </tr>
    <tr>
      <td>CNT_VISITS_GAS_STATION</td>
      <td>9.867313e+00</td>
      <td>1.438422e+01</td>
    </tr>
    <tr>
      <td>PROP_WKDAY_EVE</td>
      <td>1.252379e-01</td>
      <td>1.424004e-01</td>
    </tr>
    <tr>
      <td>PROP_WKDAY_DAY</td>
      <td>5.865683e-01</td>
      <td>5.693473e-01</td>
    </tr>
    <tr>
      <td>PROP_WKND</td>
      <td>2.881938e-01</td>
      <td>2.882523e-01</td>
    </tr>
    <tr>
      <td>FUEL_TXNS_L12</td>
      <td>8.372631e+00</td>
      <td>1.362678e+01</td>
    </tr>
    <tr>
      <td>AVG_FUEL_VOL_L12</td>
      <td>2.276106e+02</td>
      <td>4.814512e+02</td>
    </tr>
    <tr>
      <td>SHOP_TXNS_L12</td>
      <td>5.222695e+00</td>
      <td>6.442779e+00</td>
    </tr>
    <tr>
      <td>SHOP_AVG_SPEND_L12</td>
      <td>4.452832e+00</td>
      <td>4.899408e+00</td>
    </tr>
    <tr>
      <td>FUEL_CARD_FLAG</td>
      <td>2.377793e-02</td>
      <td>5.322608e-02</td>
    </tr>
    <tr>
      <td>DISTANCE_KM</td>
      <td>1.097187e+01</td>
      <td>1.096252e+01</td>
    </tr>
  </tbody>
</table>
</div>



## PART 3: DATA VISUALIZATION - CATEGORICAL FEATURES




```python
#we will use matplotlib
import matplotlib.pyplot as plt
```


```python
#simple bar plot of counts of male vs female
freq_table = pd.Series(data['TARGET']).value_counts()
plt.figure()
freq_table.plot.bar();
```


    
![png](output_21_0.png)
    



```python
#create a mosaic plot of my freq table
import statsmodels.graphics.mosaicplot as mp
freq_table = pd.Series(data['TARGET']).value_counts()
mp.mosaic(freq_table);
```


    
![png](output_22_0.png)
    



```python
#but if we were to look at a feature with meany distinct classes, it may be more useful
freq_table = pd.Series(data['REWARD_HIST_SEGMENT']).value_counts()
mp.mosaic(freq_table);
```


    
![png](output_23_0.png)
    



```python
#here we try to reformat it
#from matplotlib.pyplot import figure
fig = plt.figure(figsize=(30,6))
ax = fig.add_subplot(111)
mp.mosaic(freq_table, ax=ax);

```


    
![png](output_24_0.png)
    



```python
#try other options - change figure size and make it horizonal...
#ask the class to read documentation and see what else can be changed...
#link to doc: http://www.statsmodels.org/dev/generated/statsmodels.graphics.mosaicplot.mosaic.html
#another very useful link in general:
   #https://d1b10bmlvqabco.cloudfront.net/attach/iwjz7hyhqd4qd/isrin57rtrp2sq/j1dyzbdq5qdx/DataVisualizationinPython.html

fig = plt.figure(figsize=(3,8))
ax = fig.add_subplot(111)
mp.mosaic(freq_table, ax=ax, horizontal=0);
```


    
![png](output_25_0.png)
    



```python
#create plots for cross-tab information (mosiac and bar plots)
#bar plots for x-tab data (ex: gender vs target)
#1. We create the dataframe that holds the cross tabe resutls
crosstab_1 = pd.crosstab(data.CLIENT_GENDER, data.TARGET, margins=True, normalize='index')

#2. Plot a bar plot
plt.figure()
crosstab_1.plot.bar();

#3. Plot mosacia plot of cross-tab
mp.mosaic(crosstab_1.stack());

```


    <Figure size 432x288 with 0 Axes>



    
![png](output_26_1.png)
    



    
![png](output_26_2.png)
    


#### Next, lets look at one more example this time showing the relationship between 2 categorical features


```python
#same idea as before: 1. create a dataframe that hold the cross-tab and then plot it
crosstab_2 = pd.crosstab(data.CLIENT_GENDER, data.REWARD_HIST_SEGMENT, margins=True, normalize='index')
crosstab_2.plot.bar()


#looks a bit hard to read, so lets switch positions of GENDER & REWARD SEGMENT

#notice how there's some text (can very long at times) before your actual plot.
#This is useful sometimes, but you can supress it as well by either:
#1. Add semi-column after plot. ex: crosstab_2.plot.bar();
#2. Explicity type in plt.show()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116845d90>




    
![png](output_28_1.png)
    



```python
#Same plot, but this time we switched the positions of GENDER & REWARD SEGMENT
crosstab_2 = pd.crosstab(data.REWARD_HIST_SEGMENT, data.CLIENT_GENDER, margins=True, normalize='index')
crosstab_2.plot.bar()
#In addition, let's say we want to make this a stacked bar plot -- how do we do this?
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10e7c8d0>




    
![png](output_29_1.png)
    


### Try to make this a proportionally scaled stacked bar-plot. 

As you will see making this a stacked-proportional bar plot in Python is not quite as easy as R's ggplot2. 

Here's one approach to make it happen. There are code recipes for these type of things, but until they get intergrated into matplotlib or seaborn you might write some custom code for any data viz that beyond the normal templates offered.


#### Try to make this a stacked & filled (i.e. proporitional) bar plot

General recipe for making any custom plots
1. Manipulate your data and calculate the exact metrics you need
2. Plot those metrics using matplotlib's standard plot functions


```python
#in this case step 1 we need:
#by Reward Segment the % of males vs females
#the code below performs these calculation in creates a "melted" (i.e. skinny)
#dataframe that can be used as input for plotting.

#Note: you won't need to know how to do all the data manipulation below,
    #but rather focus on structure of what we put as an input into the plotting fucntion
plot_data = pd.merge(pd.DataFrame(data.groupby(['REWARD_HIST_SEGMENT','CLIENT_GENDER'])\
                                 .size()).reset_index(),pd.DataFrame(data.groupby('REWARD_HIST_SEGMENT')\
                                 .size()).reset_index(), how='left', on='REWARD_HIST_SEGMENT')

plot_data.columns = ['REWARD_HIST_SEGMENT', 'CLIENT_GENDER', 'number_combination', 'number_answer']
```


```python
#preview the structure
plot_data.head(10)
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
      <th>REWARD_HIST_SEGMENT</th>
      <th>CLIENT_GENDER</th>
      <th>number_combination</th>
      <th>number_answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Breakage</td>
      <td>F</td>
      <td>346</td>
      <td>876</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Breakage</td>
      <td>M</td>
      <td>455</td>
      <td>876</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gift-card Redeemer</td>
      <td>F</td>
      <td>3453</td>
      <td>6273</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Gift-card Redeemer</td>
      <td>M</td>
      <td>2788</td>
      <td>6273</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Never Redeemed</td>
      <td>F</td>
      <td>4270</td>
      <td>17327</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Never Redeemed</td>
      <td>M</td>
      <td>5009</td>
      <td>17327</td>
    </tr>
    <tr>
      <td>6</td>
      <td>No Redemptions L18 months</td>
      <td>F</td>
      <td>3290</td>
      <td>7922</td>
    </tr>
    <tr>
      <td>7</td>
      <td>No Redemptions L18 months</td>
      <td>M</td>
      <td>4285</td>
      <td>7922</td>
    </tr>
    <tr>
      <td>8</td>
      <td>No Redemptions L24 months</td>
      <td>F</td>
      <td>4910</td>
      <td>9481</td>
    </tr>
    <tr>
      <td>9</td>
      <td>No Redemptions L24 months</td>
      <td>M</td>
      <td>4407</td>
      <td>9481</td>
    </tr>
  </tbody>
</table>
</div>




```python
#next we just need to calculate % from available numerator & denominator
perc=plot_data['number_combination']/plot_data['number_answer']
melted = plot_data[['REWARD_HIST_SEGMENT','CLIENT_GENDER']]
melted['perc'] = perc
melted


```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.





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
      <th>REWARD_HIST_SEGMENT</th>
      <th>CLIENT_GENDER</th>
      <th>perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Breakage</td>
      <td>F</td>
      <td>0.394977</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Breakage</td>
      <td>M</td>
      <td>0.519406</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Gift-card Redeemer</td>
      <td>F</td>
      <td>0.550454</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Gift-card Redeemer</td>
      <td>M</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Never Redeemed</td>
      <td>F</td>
      <td>0.246436</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Never Redeemed</td>
      <td>M</td>
      <td>0.289086</td>
    </tr>
    <tr>
      <td>6</td>
      <td>No Redemptions L18 months</td>
      <td>F</td>
      <td>0.415299</td>
    </tr>
    <tr>
      <td>7</td>
      <td>No Redemptions L18 months</td>
      <td>M</td>
      <td>0.540899</td>
    </tr>
    <tr>
      <td>8</td>
      <td>No Redemptions L24 months</td>
      <td>F</td>
      <td>0.517878</td>
    </tr>
    <tr>
      <td>9</td>
      <td>No Redemptions L24 months</td>
      <td>M</td>
      <td>0.464824</td>
    </tr>
    <tr>
      <td>10</td>
      <td>Retail High-Volume</td>
      <td>F</td>
      <td>0.516164</td>
    </tr>
    <tr>
      <td>11</td>
      <td>Retail High-Volume</td>
      <td>M</td>
      <td>0.472821</td>
    </tr>
    <tr>
      <td>12</td>
      <td>Retail Low-Volume Non-Saver</td>
      <td>F</td>
      <td>0.624166</td>
    </tr>
    <tr>
      <td>13</td>
      <td>Retail Low-Volume Non-Saver</td>
      <td>M</td>
      <td>0.361008</td>
    </tr>
    <tr>
      <td>14</td>
      <td>Retail Low-Volume Saver</td>
      <td>F</td>
      <td>0.585413</td>
    </tr>
    <tr>
      <td>15</td>
      <td>Retail Low-Volume Saver</td>
      <td>M</td>
      <td>0.404511</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Retail X-mas Redeemer</td>
      <td>F</td>
      <td>0.523083</td>
    </tr>
    <tr>
      <td>17</td>
      <td>Retail X-mas Redeemer</td>
      <td>M</td>
      <td>0.445640</td>
    </tr>
    <tr>
      <td>18</td>
      <td>Travel Only Redeemer</td>
      <td>F</td>
      <td>0.423825</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Travel Only Redeemer</td>
      <td>M</td>
      <td>0.566631</td>
    </tr>
    <tr>
      <td>20</td>
      <td>Travel and Retail Redeemer</td>
      <td>F</td>
      <td>0.492482</td>
    </tr>
    <tr>
      <td>21</td>
      <td>Travel and Retail Redeemer</td>
      <td>M</td>
      <td>0.500358</td>
    </tr>
  </tbody>
</table>
</div>




```python
#from this manipulated dataset, we can plot a stacked bar plot...
melted.groupby(['REWARD_HIST_SEGMENT','CLIENT_GENDER']).\
    perc.sum().unstack().plot(kind='bar', stacked=True, ylim=(0,1))
    
#not theres still a slight issue with missing values
#bonus homework: try to fix this!
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11624cd50>




    
![png](output_36_1.png)
    


## PART 4: DATA EXPLORATION: NUMERIC VARIABLES


```python
#1. Remember the prebuilt describe() fucntion is quite useful for numeric features
data.describe()
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
      <th>TARGET</th>
      <th>ID</th>
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>9.871400e+04</td>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.208998</td>
      <td>9.915449e+07</td>
      <td>2.894737</td>
      <td>1.728236</td>
      <td>2.963693</td>
      <td>46.167885</td>
      <td>4919.443433</td>
      <td>11.118323</td>
      <td>0.135107</td>
      <td>0.580036</td>
      <td>0.284857</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.037077</td>
      <td>10.033388</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.406595</td>
      <td>2.849642e+04</td>
      <td>1.344243</td>
      <td>0.923183</td>
      <td>1.236660</td>
      <td>81.502148</td>
      <td>11529.172773</td>
      <td>16.817524</td>
      <td>0.228338</td>
      <td>0.343604</td>
      <td>0.307625</td>
      <td>16.846474</td>
      <td>747.989781</td>
      <td>13.335370</td>
      <td>7.600509</td>
      <td>0.188951</td>
      <td>3.689799</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>9.910513e+07</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.912981e+07</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>12.087390</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>35.010000</td>
      <td>1.000000</td>
      <td>2.150000</td>
      <td>0.000000</td>
      <td>9.812254</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>9.915449e+07</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>29.583220</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.617021</td>
      <td>0.209302</td>
      <td>4.000000</td>
      <td>99.030000</td>
      <td>2.000000</td>
      <td>3.290000</td>
      <td>0.000000</td>
      <td>10.966584</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>9.917917e+07</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.245590</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>0.200000</td>
      <td>0.904762</td>
      <td>0.433209</td>
      <td>12.000000</td>
      <td>344.015000</td>
      <td>5.000000</td>
      <td>5.092500</td>
      <td>0.000000</td>
      <td>11.713106</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>9.920385e+07</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>15342.317340</td>
      <td>693385.000000</td>
      <td>1012.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>649.000000</td>
      <td>37232.370000</td>
      <td>526.000000</td>
      <td>416.690000</td>
      <td>1.000000</td>
      <td>24.559018</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. Lets try to get a histogram for the numeric feature NUM_FMLY_MEMBERS
#this just grabs one column from out data frame and puts it in a series (set of numbers)
x=pd.Series(data['NUM_FMLY_MEMBERS'])
x.head()
```




    0    2.0
    1    2.0
    2    NaN
    3    2.0
    4    2.0
    Name: NUM_FMLY_MEMBERS, dtype: float64




```python
#2 Hisograms are you friend
#remember we will use matplotlib
import matplotlib.pyplot as plt
plt.hist(x)

#we get an error! WHY?
#because hist() can't deal with NaN values, so we can quickly remove them before plotting
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/numpy/lib/histograms.py:829: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/numpy/lib/histograms.py:830: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)





    (array([ 7192., 19444., 11714., 12196.,  4496.,  1266.,   376.,   135.,
               64.,    41.]),
     array([ 1. ,  1.9,  2.8,  3.7,  4.6,  5.5,  6.4,  7.3,  8.2,  9.1, 10. ]),
     <a list of 10 Patch objects>)




    
![png](output_40_2.png)
    



```python
#here we remove NaN values using dropna() function before plotting
plt.hist(x.dropna());

```


    
![png](output_41_0.png)
    



```python
#also in the future you probably want this all in one line (without creating intermediate variable x)
plt.hist(pd.Series(data['NUM_FMLY_MEMBERS']).dropna());

#not I also increase bin size to make it smoother
```


    
![png](output_42_0.png)
    



```python
#we could create similar histograms for other numeric features. For ex:
plt.hist(pd.Series(data['PARTNERS_SHOPPED']).dropna(), bins=12)
#note: I also changed the # of bins to 12 (default was 10)
```




    (array([1.0419e+04, 2.5380e+04, 3.4091e+04, 1.8849e+04, 6.7480e+03,
            2.0400e+03, 6.4500e+02, 2.0000e+02, 6.1000e+01, 2.1000e+01,
            9.0000e+00, 3.0000e+00]),
     array([ 0.        ,  1.08333333,  2.16666667,  3.25      ,  4.33333333,
             5.41666667,  6.5       ,  7.58333333,  8.66666667,  9.75      ,
            10.83333333, 11.91666667, 13.        ]),
     <a list of 12 Patch objects>)




    
![png](output_43_1.png)
    



```python
#let's try another one:
plt.hist(pd.Series(data['AVG_WKLY_SPND_ALL_PARTNERS']).dropna());

#what's the problem here?
#how do I fix it?
```


    
![png](output_44_0.png)
    



```python
#We'll come back to fix the problem with outliers systematically, but for now let's do a quick fix
#multistep approach
x=pd.Series(data['AVG_WKLY_SPND_ALL_PARTNERS']).dropna()
x.describe()

```




    count    98465.000000
    mean        46.167885
    std         81.502148
    min          0.000000
    25%         12.087390
    50%         29.583220
    75%         62.245590
    max      15342.317340
    Name: AVG_WKLY_SPND_ALL_PARTNERS, dtype: float64




```python
#create new copy of this, this time all values > 3000 are set to 3000
x[x>3000] = 3000
x.describe()
```




    count    98465.000000
    mean        45.919038
    std         57.437184
    min          0.000000
    25%         12.087390
    50%         29.583220
    75%         62.245590
    max       3000.000000
    Name: AVG_WKLY_SPND_ALL_PARTNERS, dtype: float64




```python
#finally we now plot x2
plt.hist(x, range(0, 500));
#plt.show()
#where(lambda x : x!=1)
```


    
![png](output_47_0.png)
    


#### If we wanted to do this for all numeric features, we would need a faster way!


```python
#there are many approaches including writing your own for loop to iterate over each column
#thankfully there is a prebuilt package that does this

#however it only works for numberic features (nature of histograms) so just need to subset it first
#Step 1: Get only numeric colums
numeric_cols = data.select_dtypes(['number'])

#Step 2: Use .diff().hist() approach to create multiple
numeric_cols.diff().hist(color='k', alpha=0.5, bins=50);

#Step 3: Go back and change plot size/format etc. to make it look better
```


    
![png](output_49_0.png)
    


### Next lets look at a numeric feature against a cateogorical feature (ex: our target variable)


```python
plt.scatter(data["TARGET"].astype(str), data["AVG_WKLY_SPND_ALL_PARTNERS"])
plt.margins(x=0.5)
plt.show()

#this is not very useful!
#what else can we do?
```


    
![png](output_51_0.png)
    



```python
#this doesn't look very useful...so what now?

#Approach to explore numeric variables vs binary target
#1. Bin numeric variables (ex: into 10 groups, this is called deciling) to create a factor variable - review this with the class
#2. And then:
  #Create table that shows avg value of target by bin
  #Or visualize on a graph
```


```python
#we use pandas cut function
pd.qcut(data['AVG_WKLY_SPND_ALL_PARTNERS'], 10).head()

#compare this to - what's the difference?
#pd.qcut(data['AVG_WKLY_SPND_ALL_PARTNERS'], 10).head()
```




    0     (-0.001, 4.102]
    1     (9.337, 15.018]
    2     (53.13, 73.448]
    3    (29.583, 39.735]
    4     (9.337, 15.018]
    Name: AVG_WKLY_SPND_ALL_PARTNERS, dtype: category
    Categories (10, interval[float64]): [(-0.001, 4.102] < (4.102, 9.337] < (9.337, 15.018] < (15.018, 21.655] ... (39.735, 53.13] < (53.13, 73.448] < (73.448, 108.222] < (108.222, 15342.317]]




```python
#so now that we understand bucketing/binning, we can plot binned version of this column vs target variable

#we use the same reciple as before
#crosstab_2 = pd.crosstab(data.REWARD_HIST_SEGMENT, data.CLIENT_GENDER, margins=True, normalize='index')
#crosstab_2.plot.bar()

#except now we use this newly created binned column -> pd.qcut(data['AVG_WKLY_SPND_ALL_PARTNERS'], 10)
crosstab = pd.crosstab(pd.qcut(data['AVG_WKLY_SPND_ALL_PARTNERS'], 10), data.CLIENT_GENDER, margins=True, normalize='index')
crosstab.plot.bar();

#now what do you see?
    #much easier to see trends --> males tense to have higher avg weekly spend accross all partners...
    #excercise for the class - try looking at this for various other numeric features vs target variable (or other categorical features)
    
```


    
![png](output_54_0.png)
    


### Next we look at relationships b/w numeric features --> CORRELATION MATRIX and PLOTS


```python
### Next we look at relationships b/w numeric features --> CORRELATION MATRIX and PLOTS
#to calculate the correlation matrix
corr = numeric_cols.corr()
corr
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
      <th>TARGET</th>
      <th>ID</th>
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TARGET</th>
      <td>1.000000</td>
      <td>-0.002353</td>
      <td>-0.011216</td>
      <td>0.018125</td>
      <td>0.085977</td>
      <td>0.114810</td>
      <td>0.084221</td>
      <td>0.178991</td>
      <td>0.019791</td>
      <td>0.012721</td>
      <td>-0.028899</td>
      <td>0.401784</td>
      <td>0.326011</td>
      <td>0.142917</td>
      <td>0.000700</td>
      <td>0.090333</td>
      <td>0.074627</td>
    </tr>
    <tr>
      <th>ID</th>
      <td>-0.002353</td>
      <td>1.000000</td>
      <td>-0.000174</td>
      <td>-0.005372</td>
      <td>0.001132</td>
      <td>0.006267</td>
      <td>0.000986</td>
      <td>-0.001978</td>
      <td>0.005307</td>
      <td>-0.003632</td>
      <td>0.000118</td>
      <td>0.004710</td>
      <td>0.003595</td>
      <td>0.000112</td>
      <td>-0.003622</td>
      <td>-0.000539</td>
      <td>-0.004398</td>
    </tr>
    <tr>
      <th>NUM_FMLY_MEMBERS</th>
      <td>-0.011216</td>
      <td>-0.000174</td>
      <td>1.000000</td>
      <td>0.327015</td>
      <td>-0.004017</td>
      <td>0.061405</td>
      <td>-0.014012</td>
      <td>0.023752</td>
      <td>0.035535</td>
      <td>-0.019147</td>
      <td>-0.004699</td>
      <td>0.033005</td>
      <td>0.026717</td>
      <td>0.001790</td>
      <td>0.015526</td>
      <td>0.006694</td>
      <td>-0.012277</td>
    </tr>
    <tr>
      <th>NUM_CARS_HH</th>
      <td>0.018125</td>
      <td>-0.005372</td>
      <td>0.327015</td>
      <td>1.000000</td>
      <td>0.012202</td>
      <td>0.046483</td>
      <td>0.034932</td>
      <td>0.085354</td>
      <td>0.011634</td>
      <td>0.023037</td>
      <td>-0.034185</td>
      <td>0.023257</td>
      <td>0.042605</td>
      <td>0.007538</td>
      <td>0.003251</td>
      <td>0.009750</td>
      <td>-0.043463</td>
    </tr>
    <tr>
      <th>PARTNERS_SHOPPED</th>
      <td>0.085977</td>
      <td>0.001132</td>
      <td>-0.004017</td>
      <td>0.012202</td>
      <td>1.000000</td>
      <td>0.188331</td>
      <td>0.165562</td>
      <td>0.201091</td>
      <td>-0.010925</td>
      <td>-0.013338</td>
      <td>0.023004</td>
      <td>0.039558</td>
      <td>0.039756</td>
      <td>0.006809</td>
      <td>-0.024363</td>
      <td>0.000242</td>
      <td>0.294027</td>
    </tr>
    <tr>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <td>0.114810</td>
      <td>0.006267</td>
      <td>0.061405</td>
      <td>0.046483</td>
      <td>0.188331</td>
      <td>1.000000</td>
      <td>0.324419</td>
      <td>0.268996</td>
      <td>-0.007900</td>
      <td>0.021267</td>
      <td>-0.017892</td>
      <td>0.188485</td>
      <td>0.268852</td>
      <td>0.103085</td>
      <td>0.034839</td>
      <td>0.062484</td>
      <td>0.096377</td>
    </tr>
    <tr>
      <th>ENDING_PT_BALANCE</th>
      <td>0.084221</td>
      <td>0.000986</td>
      <td>-0.014012</td>
      <td>0.034932</td>
      <td>0.165562</td>
      <td>0.324419</td>
      <td>1.000000</td>
      <td>0.170502</td>
      <td>0.006516</td>
      <td>-0.014239</td>
      <td>0.011067</td>
      <td>0.079507</td>
      <td>0.103408</td>
      <td>0.038981</td>
      <td>0.006048</td>
      <td>0.016294</td>
      <td>0.083297</td>
    </tr>
    <tr>
      <th>CNT_VISITS_GAS_STATION</th>
      <td>0.178991</td>
      <td>-0.001978</td>
      <td>0.023752</td>
      <td>0.085354</td>
      <td>0.201091</td>
      <td>0.268996</td>
      <td>0.170502</td>
      <td>1.000000</td>
      <td>0.012389</td>
      <td>0.023865</td>
      <td>-0.035852</td>
      <td>0.265830</td>
      <td>0.297751</td>
      <td>0.078604</td>
      <td>-0.005009</td>
      <td>0.135171</td>
      <td>0.133006</td>
    </tr>
    <tr>
      <th>PROP_WKDAY_EVE</th>
      <td>0.019791</td>
      <td>0.005307</td>
      <td>0.035535</td>
      <td>0.011634</td>
      <td>-0.010925</td>
      <td>-0.007900</td>
      <td>0.006516</td>
      <td>0.012389</td>
      <td>1.000000</td>
      <td>-0.481588</td>
      <td>-0.204347</td>
      <td>0.016424</td>
      <td>0.010855</td>
      <td>0.011298</td>
      <td>0.007778</td>
      <td>-0.005833</td>
      <td>-0.018590</td>
    </tr>
    <tr>
      <th>PROP_WKDAY_DAY</th>
      <td>0.012721</td>
      <td>-0.003632</td>
      <td>-0.019147</td>
      <td>0.023037</td>
      <td>-0.013338</td>
      <td>0.021267</td>
      <td>-0.014239</td>
      <td>0.023865</td>
      <td>-0.481588</td>
      <td>1.000000</td>
      <td>-0.759493</td>
      <td>0.043274</td>
      <td>0.054080</td>
      <td>0.017900</td>
      <td>0.001748</td>
      <td>0.050794</td>
      <td>-0.012442</td>
    </tr>
    <tr>
      <th>PROP_WKND</th>
      <td>-0.028899</td>
      <td>0.000118</td>
      <td>-0.004699</td>
      <td>-0.034185</td>
      <td>0.023004</td>
      <td>-0.017892</td>
      <td>0.011067</td>
      <td>-0.035852</td>
      <td>-0.204347</td>
      <td>-0.759493</td>
      <td>1.000000</td>
      <td>-0.061425</td>
      <td>-0.069465</td>
      <td>-0.029278</td>
      <td>-0.007965</td>
      <td>-0.052405</td>
      <td>0.027696</td>
    </tr>
    <tr>
      <th>FUEL_TXNS_L12</th>
      <td>0.401784</td>
      <td>0.004710</td>
      <td>0.033005</td>
      <td>0.023257</td>
      <td>0.039558</td>
      <td>0.188485</td>
      <td>0.079507</td>
      <td>0.265830</td>
      <td>0.016424</td>
      <td>0.043274</td>
      <td>-0.061425</td>
      <td>1.000000</td>
      <td>0.746430</td>
      <td>0.352927</td>
      <td>0.025634</td>
      <td>0.198078</td>
      <td>0.105714</td>
    </tr>
    <tr>
      <th>AVG_FUEL_VOL_L12</th>
      <td>0.326011</td>
      <td>0.003595</td>
      <td>0.026717</td>
      <td>0.042605</td>
      <td>0.039756</td>
      <td>0.268852</td>
      <td>0.103408</td>
      <td>0.297751</td>
      <td>0.010855</td>
      <td>0.054080</td>
      <td>-0.069465</td>
      <td>0.746430</td>
      <td>1.000000</td>
      <td>0.226697</td>
      <td>0.085512</td>
      <td>0.232347</td>
      <td>0.078904</td>
    </tr>
    <tr>
      <th>SHOP_TXNS_L12</th>
      <td>0.142917</td>
      <td>0.000112</td>
      <td>0.001790</td>
      <td>0.007538</td>
      <td>0.006809</td>
      <td>0.103085</td>
      <td>0.038981</td>
      <td>0.078604</td>
      <td>0.011298</td>
      <td>0.017900</td>
      <td>-0.029278</td>
      <td>0.352927</td>
      <td>0.226697</td>
      <td>1.000000</td>
      <td>0.050676</td>
      <td>0.071410</td>
      <td>0.034045</td>
    </tr>
    <tr>
      <th>SHOP_AVG_SPEND_L12</th>
      <td>0.000700</td>
      <td>-0.003622</td>
      <td>0.015526</td>
      <td>0.003251</td>
      <td>-0.024363</td>
      <td>0.034839</td>
      <td>0.006048</td>
      <td>-0.005009</td>
      <td>0.007778</td>
      <td>0.001748</td>
      <td>-0.007965</td>
      <td>0.025634</td>
      <td>0.085512</td>
      <td>0.050676</td>
      <td>1.000000</td>
      <td>0.037423</td>
      <td>-0.001790</td>
    </tr>
    <tr>
      <th>FUEL_CARD_FLAG</th>
      <td>0.090333</td>
      <td>-0.000539</td>
      <td>0.006694</td>
      <td>0.009750</td>
      <td>0.000242</td>
      <td>0.062484</td>
      <td>0.016294</td>
      <td>0.135171</td>
      <td>-0.005833</td>
      <td>0.050794</td>
      <td>-0.052405</td>
      <td>0.198078</td>
      <td>0.232347</td>
      <td>0.071410</td>
      <td>0.037423</td>
      <td>1.000000</td>
      <td>0.025172</td>
    </tr>
    <tr>
      <th>DISTANCE_KM</th>
      <td>0.074627</td>
      <td>-0.004398</td>
      <td>-0.012277</td>
      <td>-0.043463</td>
      <td>0.294027</td>
      <td>0.096377</td>
      <td>0.083297</td>
      <td>0.133006</td>
      <td>-0.018590</td>
      <td>-0.012442</td>
      <td>0.027696</td>
      <td>0.105714</td>
      <td>0.078904</td>
      <td>0.034045</td>
      <td>-0.001790</td>
      <td>0.025172</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#bonus exercise - see if you can "melt" this to make it easier to ready...
```

### Correlation Matrix


```python
#next you can plot this using matplotlib (what we've been using so far) - this 
plt.matshow(numeric_cols.corr())
```




    <matplotlib.image.AxesImage at 0x17fb0c18>




    
![png](output_59_1.png)
    



```python
#you can also try seaborn another popular data viz package for python for this heat-map
import seaborn as sns
corr = numeric_cols.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x187a7be0>




    
![png](output_60_1.png)
    



```python
#to see thcorr.head()
```

## PART 5: VISUALIZING & DEALING WITH MISSING VALUES 


```python
#there's a cool package that neatly shows patterns in missing data
#check out this link: https://github.com/ResidentMario/missingno
```


```python
#!pip install missingno
```


```python
import missingno as msno
%matplotlib inline
msno.matrix(data.sample(1000))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1893a518>




    
![png](output_65_1.png)
    



```python
#some key advantages of the visualization above is that it helps you track which columns are often missing data "together"
#often this helps you diagnose data entry or collection system probelms (web page is broken)
```


```python
#another cool visulalization of missing data from this package is the heatmap
msno.heatmap(data)

#this looks at the corrleation of missing data
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18982470>




    
![png](output_67_1.png)
    


### DEALING WITH MISSING VALUES


```python
#easiest thing would be to delete any row that has 1+ missing value
#the problem with this is that we lose lots of data (esp due to curse of dimensionality)
#so generally we look for other methods, but its okay to delete certain rows where x%+ columns are missing etc.
```


```python
#Before you start sometimes you may replace 0 with NaN (i.e. missing), but be careful before you do this though (does not always apply!)
#data = data.replace(0, numpy.NaN)

#check the number of NaN values in each column
print(data.isnull().sum())
```

    TARGET                            0
    ID                                0
    CLIENT_GENDER                  9712
    CLIENT_AGE_BAND               10110
    NUM_FMLY_MEMBERS              41790
    NUM_CARS_HH                   46139
    CLIENT_REGION                  9167
    PARTNERS_SHOPPED                248
    AVG_WKLY_SPND_ALL_PARTNERS      249
    REWARD_HIST_SEGMENT             550
    ENDING_PT_BALANCE                 0
    CNT_VISITS_GAS_STATION            1
    PROP_WKDAY_EVE                    0
    PROP_WKDAY_DAY                    0
    PROP_WKND                         0
    FUEL_TXNS_L12                 13883
    AVG_FUEL_VOL_L12              13806
    SHOP_TXNS_L12                 50455
    SHOP_AVG_SPEND_L12            50455
    FUEL_CARD_FLAG                    0
    DISTANCE_KM                       0
    dtype: int64


#### Let's impute missing values with mean (try to change to median, hard-coded value, mode, etc.)


```python
#fill missing values with mean/median/other column values
#also create a new dataframe so my original is still in-tact
data_imputed = data
data_imputed.fillna(data.mean(), inplace=True)
```


```python
#check missing values now
data.isnull().sum()
```




    TARGET                            0
    ID                                0
    CLIENT_GENDER                  9712
    CLIENT_AGE_BAND               10110
    NUM_FMLY_MEMBERS                  0
    NUM_CARS_HH                       0
    CLIENT_REGION                  9167
    PARTNERS_SHOPPED                  0
    AVG_WKLY_SPND_ALL_PARTNERS        0
    REWARD_HIST_SEGMENT             550
    ENDING_PT_BALANCE                 0
    CNT_VISITS_GAS_STATION            0
    PROP_WKDAY_EVE                    0
    PROP_WKDAY_DAY                    0
    PROP_WKND                         0
    FUEL_TXNS_L12                     0
    AVG_FUEL_VOL_L12                  0
    SHOP_TXNS_L12                     0
    SHOP_AVG_SPEND_L12                0
    FUEL_CARD_FLAG                    0
    DISTANCE_KM                       0
    dtype: int64



#### What's the problem with this?
>- We are imputing with mean BEFORE dealing with outliers. We probably shouldn't do that...
- Also we should probably only do a mean imputation if there is at least X% of data populated
- We could also consider imputting mean by groups (if height is missing can we impute different numbers for males & females?)

#### <b> Use sklearns Imputer pre-processing

- scikit-learn library provides a Imputer() pre-processing function
- this does both mean imputation and creates a missing value itendifier
- it is a flexible class that allows you to specify the value to replace (it can be something other than NaN) and the technique used to replace it (such as mean, median, or mode).
- the Imputer class operates directly on the NumPy array instead of the DataFrame.


```python
#Example below uses the Imputer class to replace missing values with the mean of each column then prints the number of NaN values in the transformed matrix.
from sklearn.preprocessing import Imputer
import numpy as np

# First we need to subset only numeric columns
#also you MUST drop target in this case as well (it creates problem if you don't as Imputer wont impute it!)
#and we might as well drop ID too as there's no point in imputting that (plus its never missing)
numeric_cols = data.drop(columns=['TARGET', 'ID']).select_dtypes(['number'])
```


```python
#check initial nulls
numeric_cols.isnull().sum()
```




    NUM_FMLY_MEMBERS              0
    NUM_CARS_HH                   0
    PARTNERS_SHOPPED              0
    AVG_WKLY_SPND_ALL_PARTNERS    0
    ENDING_PT_BALANCE             0
    CNT_VISITS_GAS_STATION        0
    PROP_WKDAY_EVE                0
    PROP_WKDAY_DAY                0
    PROP_WKND                     0
    FUEL_TXNS_L12                 0
    AVG_FUEL_VOL_L12              0
    SHOP_TXNS_L12                 0
    SHOP_AVG_SPEND_L12            0
    FUEL_CARD_FLAG                0
    DISTANCE_KM                   0
    dtype: int64




```python
#see list of column names for next step
numeric_cols.columns
```




    Index(['NUM_FMLY_MEMBERS', 'NUM_CARS_HH', 'PARTNERS_SHOPPED',
           'AVG_WKLY_SPND_ALL_PARTNERS', 'ENDING_PT_BALANCE',
           'CNT_VISITS_GAS_STATION', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12',
           'SHOP_AVG_SPEND_L12', 'FUEL_CARD_FLAG', 'DISTANCE_KM'],
          dtype='object')




```python
#the next 2 steps are pretty much the standard recipe for applying anything from SKLEARN (model fitting or preprocessing as in this case)
#instantiate imputer 
imp_mean=Imputer(missing_values="NaN", strategy="mean" )
imp_med=Imputer(missing_values="NaN", strategy="median" )
imp_mode=Imputer(missing_values="NaN", strategy="mode" )

#fit/apply imputer
numeric_cols["NUM_FMLY_MEMBERS"]=imp_mean.fit_transform(numeric_cols[["NUM_FMLY_MEMBERS"]]).ravel()

#we can apply different imput methods for each column. More on this later...
```


```python
numeric_cols.isnull().sum()
```




    NUM_FMLY_MEMBERS              0
    NUM_CARS_HH                   0
    PARTNERS_SHOPPED              0
    AVG_WKLY_SPND_ALL_PARTNERS    0
    ENDING_PT_BALANCE             0
    CNT_VISITS_GAS_STATION        0
    PROP_WKDAY_EVE                0
    PROP_WKDAY_DAY                0
    PROP_WKND                     0
    FUEL_TXNS_L12                 0
    AVG_FUEL_VOL_L12              0
    SHOP_TXNS_L12                 0
    SHOP_AVG_SPEND_L12            0
    FUEL_CARD_FLAG                0
    DISTANCE_KM                   0
    dtype: int64




```python
#### Next we want to create a missing data flag as a feature as well
data_prep_2=numeric_cols
for col in data_prep_2.columns:
    data_prep_2[col+"_missing"] = data_prep_2[col].isnull()
```


```python
data_prep_2.columns
```




    Index(['NUM_FMLY_MEMBERS', 'NUM_CARS_HH', 'PARTNERS_SHOPPED',
           'AVG_WKLY_SPND_ALL_PARTNERS', 'ENDING_PT_BALANCE',
           'CNT_VISITS_GAS_STATION', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12',
           'SHOP_AVG_SPEND_L12', 'FUEL_CARD_FLAG', 'DISTANCE_KM',
           'NUM_FMLY_MEMBERS_missing', 'NUM_CARS_HH_missing',
           'PARTNERS_SHOPPED_missing', 'AVG_WKLY_SPND_ALL_PARTNERS_missing',
           'ENDING_PT_BALANCE_missing', 'CNT_VISITS_GAS_STATION_missing',
           'PROP_WKDAY_EVE_missing', 'PROP_WKDAY_DAY_missing', 'PROP_WKND_missing',
           'FUEL_TXNS_L12_missing', 'AVG_FUEL_VOL_L12_missing',
           'SHOP_TXNS_L12_missing', 'SHOP_AVG_SPEND_L12_missing',
           'FUEL_CARD_FLAG_missing', 'DISTANCE_KM_missing'],
          dtype='object')



## PART 6: OUTLIER DETECTION & OTHER TRANSFORMATION


```python
#### OUTLIER DETECTION - Box Plots
```


```python
#let's look at feautre 'CNT_VISITS_GAS_STATION' first
i = 'CNT_VISITS_GAS_STATION'
df=data

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(df[i].min(), df[i].max()*1.1)
 
ax = df[i].plot(kind='kde')
 
plt.subplot(212)
plt.xlim(df[i].min(), df[i].max()*1.1)
sns.boxplot(x=df[i])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12322358>




    
![png](output_86_1.png)
    



```python
#One way to deal detect outliers is to IQR
```


```python
q75, q25 = np.percentile(data.CNT_VISITS_GAS_STATION.dropna(), [75 ,25])
iqr = q75 - q25
min_outlier = q25 - (iqr*1.5)
max_outlier = q75 + (iqr*1.5)
```


```python
print(min_outlier, '    ,   ', max_outlier)
```

    -14.5     ,    29.5



```python
#visualise min/max IQR on a plot
i = 'CNT_VISITS_GAS_STATION'
df=data
 
plt.subplot(212)
plt.xlim(df[i].min(), max_outlier*1.2)
sns.boxplot(x=df[i])
plt.axvline(x=min_outlier)
plt.axvline(x=max_outlier)
```




    <matplotlib.lines.Line2D at 0x174bcc50>




    
![png](output_90_1.png)
    



```python
#### Another idea would be to take the LOG first and then detect outliers.
```


```python
#dealing with outliers by transforming (ex: Log)
#note: log cannot deal with 0, as log(0) is infinity, so need to replace 0's with -1
data.loc[data.CNT_VISITS_GAS_STATION == 0, 'CNT_VISITS_GAS_STATION'] = -1

# Drop NA
data['CNT_VISITS_GAS_STATION'].dropna(inplace=True)
 
# Log Transform
data['Log_CNT_VISITS_GAS_STATION'] = np.log(data['CNT_VISITS_GAS_STATION'])

```


```python
print(data['Log_CNT_VISITS_GAS_STATION'].describe())
data['CNT_VISITS_GAS_STATION'].describe()

#notice how the log-scale only goes to 6.9 and we might not even have to deal with outliers...
```

    count    98714.000000
    mean         1.702211
    std          1.195683
    min          0.000000
    25%          0.693147
    50%          1.609438
    75%          2.564949
    max          6.919684
    Name: Log_CNT_VISITS_GAS_STATION, dtype: float64





    count    98714.000000
    mean        11.118323
    std         16.817438
    min          1.000000
    25%          2.000000
    50%          5.000000
    75%         13.000000
    max       1012.000000
    Name: CNT_VISITS_GAS_STATION, dtype: float64




```python
#if we wanted to try IQR again we could do so...
#instead we'll use a different technique called outlier capping

```

#### Outlier detection & CAPPING based on percentile-capping


```python
#provides a quick way to deal with numeric outliers
#capping can be done based on any criteria (99th percentile, above IQR etc.)
```


```python
#remember we need only numeric columns for now (deal with categorial later)
import numpy as np
numeric_cols = data.select_dtypes(['number']).drop(columns=['TARGET','ID'])
data_capped=numeric_cols.apply(lambda x: x.clip_upper(np.percentile(x.dropna(), 98)))

```

Compare distributionof original vs capped



```python
#compare distributionof original vs capped
numeric_cols.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
      <th>Log_CNT_VISITS_GAS_STATION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.894737</td>
      <td>1.728236</td>
      <td>2.963693</td>
      <td>46.167885</td>
      <td>4919.443433</td>
      <td>11.118323</td>
      <td>0.135107</td>
      <td>0.580036</td>
      <td>0.284857</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.037077</td>
      <td>10.033388</td>
      <td>1.702211</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.020786</td>
      <td>0.673730</td>
      <td>1.235105</td>
      <td>81.399290</td>
      <td>11529.172773</td>
      <td>16.817438</td>
      <td>0.228338</td>
      <td>0.343604</td>
      <td>0.307625</td>
      <td>15.616962</td>
      <td>693.713645</td>
      <td>9.324006</td>
      <td>5.314228</td>
      <td>0.188951</td>
      <td>3.689799</td>
      <td>1.195683</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.728236</td>
      <td>2.000000</td>
      <td>12.122863</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>40.700000</td>
      <td>2.000000</td>
      <td>3.360000</td>
      <td>0.000000</td>
      <td>9.812254</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.894737</td>
      <td>1.728236</td>
      <td>3.000000</td>
      <td>29.686880</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.617021</td>
      <td>0.209302</td>
      <td>6.000000</td>
      <td>146.875000</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.000000</td>
      <td>10.966584</td>
      <td>1.609438</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.126805</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>0.200000</td>
      <td>0.904762</td>
      <td>0.433209</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.000000</td>
      <td>11.713106</td>
      <td>2.564949</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>15342.317340</td>
      <td>693385.000000</td>
      <td>1012.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>649.000000</td>
      <td>37232.370000</td>
      <td>526.000000</td>
      <td>416.690000</td>
      <td>1.000000</td>
      <td>24.559018</td>
      <td>6.919684</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_capped.describe()
data_capped.columns
```




    Index(['NUM_FMLY_MEMBERS', 'NUM_CARS_HH', 'PARTNERS_SHOPPED',
           'AVG_WKLY_SPND_ALL_PARTNERS', 'ENDING_PT_BALANCE',
           'CNT_VISITS_GAS_STATION', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12',
           'SHOP_AVG_SPEND_L12', 'FUEL_CARD_FLAG', 'DISTANCE_KM',
           'Log_CNT_VISITS_GAS_STATION'],
          dtype='object')




```python
#next we will apply log transforms to certain features
log_cols=data_capped[['NUM_FMLY_MEMBERS', 'PARTNERS_SHOPPED','NUM_CARS_HH','PARTNERS_SHOPPED','AVG_WKLY_SPND_ALL_PARTNERS'\
                      ,'ENDING_PT_BALANCE','FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12','SHOP_AVG_SPEND_L12']]
log_cols.describe()
#data_capped_nonzero = data_capped.replace(0,-0.02)
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>PARTNERS_SHOPPED</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.865521</td>
      <td>2.949754</td>
      <td>1.718085</td>
      <td>2.949754</td>
      <td>44.338007</td>
      <td>4356.155338</td>
      <td>9.645121</td>
      <td>296.061562</td>
      <td>5.136858</td>
      <td>4.398752</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.925747</td>
      <td>1.188921</td>
      <td>0.619877</td>
      <td>1.188921</td>
      <td>43.648276</td>
      <td>6271.131855</td>
      <td>11.702096</td>
      <td>406.249807</td>
      <td>3.933727</td>
      <td>1.957448</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.728236</td>
      <td>2.000000</td>
      <td>12.122863</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>40.700000</td>
      <td>2.000000</td>
      <td>3.360000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.894737</td>
      <td>3.000000</td>
      <td>1.728236</td>
      <td>3.000000</td>
      <td>29.686880</td>
      <td>2011.000000</td>
      <td>6.000000</td>
      <td>146.875000</td>
      <td>5.660664</td>
      <td>4.674817</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.126805</td>
      <td>5097.000000</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>185.022685</td>
      <td>31639.480000</td>
      <td>55.000000</td>
      <td>2004.704800</td>
      <td>23.000000</td>
      <td>12.260000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#next replace 0 with small negative values otherwise log will fail
log_cols_non_zero = log_cols.replace(0,-0.02)
log_cols_non_zero.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>PARTNERS_SHOPPED</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.865521</td>
      <td>2.949751</td>
      <td>1.717730</td>
      <td>2.949751</td>
      <td>44.337945</td>
      <td>4356.155336</td>
      <td>9.645121</td>
      <td>296.061562</td>
      <td>5.136858</td>
      <td>4.398694</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.925747</td>
      <td>1.188929</td>
      <td>0.620866</td>
      <td>1.188929</td>
      <td>43.648339</td>
      <td>6271.131856</td>
      <td>11.702096</td>
      <td>406.249807</td>
      <td>3.933727</td>
      <td>1.957578</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-0.020000</td>
      <td>-0.020000</td>
      <td>-0.020000</td>
      <td>-0.020000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.020000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.728236</td>
      <td>2.000000</td>
      <td>12.122863</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>40.700000</td>
      <td>2.000000</td>
      <td>3.360000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.894737</td>
      <td>3.000000</td>
      <td>1.728236</td>
      <td>3.000000</td>
      <td>29.686880</td>
      <td>2011.000000</td>
      <td>6.000000</td>
      <td>146.875000</td>
      <td>5.660664</td>
      <td>4.674817</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.126805</td>
      <td>5097.000000</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>185.022685</td>
      <td>31639.480000</td>
      <td>55.000000</td>
      <td>2004.704800</td>
      <td>23.000000</td>
      <td>12.260000</td>
    </tr>
  </tbody>
</table>
</div>




```python
log_cols_non_zero['Log_NUM_FMLY_MEMBERS'] = np.log(log_cols_non_zero['NUM_FMLY_MEMBERS'])
```


```python
log_cols_non_zero['Log_NUM_FMLY_MEMBERS'].describe()
```




    count    98714.000000
    mean         0.992132
    std          0.369602
    min          0.000000
    25%          0.693147
    50%          1.062894
    75%          1.098612
    max          1.609438
    Name: Log_NUM_FMLY_MEMBERS, dtype: float64




in next section we see how to apply log transformation to many features at once


```python

```

## PART 7: PUTTING IT ALL TOGETHER & FITTING A MODEL

### HERE'S A SHORT STRATEGY WE'LL USE FOR DATA PREP
> - Step 1: Apply lower limit =0 and upper-limit capping at 99th percentile to certain numeric features where appropriate
> - Step 2: Apply log transformations to highly skewed numeric features
> - Step 3: Apply median imputation for all logged features & create imputed_flag as a feature
> - Step 4: Apply mean imputation for all other numeric features & create missing flag
> - Step 5: Create a missing flag column for all categorial missing feature

ONCE THAT'S COMPLETE WE'LL FIT SEVERAL SK-LEARN CLASSIFIERS
> - Step 6: Join everythig together from step 1-5
> - Step 7: Feed into SK LEARN

#### STEP 1: Apply lower limit=0 and upper limit 99th percentile outlier capping to raw numeric features


```python
#reimport data so we start fresh
data = pd.read_csv('PREMIUM_FUEL.csv')
```


```python
import numpy as np
numeric_cols = data.select_dtypes(['number']).drop(columns=['TARGET','ID'])
numeric_cols.columns
```




    Index(['NUM_FMLY_MEMBERS', 'NUM_CARS_HH', 'PARTNERS_SHOPPED',
           'AVG_WKLY_SPND_ALL_PARTNERS', 'ENDING_PT_BALANCE',
           'CNT_VISITS_GAS_STATION', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12',
           'SHOP_AVG_SPEND_L12', 'FUEL_CARD_FLAG', 'DISTANCE_KM'],
          dtype='object')




```python
numeric_cols.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>2.894737</td>
      <td>1.728236</td>
      <td>2.963693</td>
      <td>46.167885</td>
      <td>4919.443433</td>
      <td>11.118323</td>
      <td>0.135107</td>
      <td>0.580036</td>
      <td>0.284857</td>
      <td>10.249909</td>
      <td>327.959795</td>
      <td>5.660664</td>
      <td>4.674817</td>
      <td>0.037077</td>
      <td>10.033388</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.344243</td>
      <td>0.923183</td>
      <td>1.236660</td>
      <td>81.502148</td>
      <td>11529.172773</td>
      <td>16.817524</td>
      <td>0.228338</td>
      <td>0.343604</td>
      <td>0.307625</td>
      <td>16.846474</td>
      <td>747.989781</td>
      <td>13.335370</td>
      <td>7.600509</td>
      <td>0.188951</td>
      <td>3.689799</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-6324.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>12.087390</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>35.010000</td>
      <td>1.000000</td>
      <td>2.150000</td>
      <td>0.000000</td>
      <td>9.812254</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>29.583220</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.617021</td>
      <td>0.209302</td>
      <td>4.000000</td>
      <td>99.030000</td>
      <td>2.000000</td>
      <td>3.290000</td>
      <td>0.000000</td>
      <td>10.966584</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.245590</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>0.200000</td>
      <td>0.904762</td>
      <td>0.433209</td>
      <td>12.000000</td>
      <td>344.015000</td>
      <td>5.000000</td>
      <td>5.092500</td>
      <td>0.000000</td>
      <td>11.713106</td>
    </tr>
    <tr>
      <td>max</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>15342.317340</td>
      <td>693385.000000</td>
      <td>1012.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>649.000000</td>
      <td>37232.370000</td>
      <td>526.000000</td>
      <td>416.690000</td>
      <td>1.000000</td>
      <td>24.559018</td>
    </tr>
  </tbody>
</table>
</div>




```python
cap_numeric_cols = numeric_cols[['NUM_FMLY_MEMBERS', 'NUM_CARS_HH','PARTNERS_SHOPPED','AVG_WKLY_SPND_ALL_PARTNERS'\
                      ,'ENDING_PT_BALANCE', 'CNT_VISITS_GAS_STATION','FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12','SHOP_AVG_SPEND_L12']]

data_capped = cap_numeric_cols.apply(lambda x: x.clip_upper(np.percentile(x.dropna(), 98)))\
    .apply(lambda x: x.clip_lower(0))
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: FutureWarning: clip_upper(threshold) is deprecated, use clip(upper=threshold) instead
      after removing the cwd from sys.path.
    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: clip_lower(threshold) is deprecated, use clip(lower=threshold) instead
      """



```python
print(len(numeric_cols.columns))
print(len(cap_numeric_cols.columns))
```

    15
    10



```python
data_capped.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.877134</td>
      <td>1.709177</td>
      <td>2.949719</td>
      <td>44.334788</td>
      <td>4356.962265</td>
      <td>10.533314</td>
      <td>9.632351</td>
      <td>294.362763</td>
      <td>4.936157</td>
      <td>4.263892</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.288263</td>
      <td>0.849290</td>
      <td>1.190417</td>
      <td>43.707867</td>
      <td>6270.380024</td>
      <td>12.971966</td>
      <td>12.940057</td>
      <td>451.857962</td>
      <td>6.884891</td>
      <td>3.299975</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>12.087390</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>35.010000</td>
      <td>1.000000</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>29.583220</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>99.030000</td>
      <td>2.000000</td>
      <td>3.290000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.245590</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>12.000000</td>
      <td>344.015000</td>
      <td>5.000000</td>
      <td>5.092500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>185.092985</td>
      <td>31639.480000</td>
      <td>61.000000</td>
      <td>59.000000</td>
      <td>2166.515200</td>
      <td>35.000000</td>
      <td>17.684004</td>
    </tr>
  </tbody>
</table>
</div>



#### STEP 2: Apply log transformations to highly skewed numeric features


```python
#we will apply this to the same set of highly skewed features we applied capping to
data_cap_nozero=data_capped.apply(lambda x: x.replace(0, 0.02))
```


```python
data_cap_nozero.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.877134</td>
      <td>1.709844</td>
      <td>2.949722</td>
      <td>44.334850</td>
      <td>4356.962278</td>
      <td>10.533314</td>
      <td>9.632351</td>
      <td>294.362763</td>
      <td>4.936157</td>
      <td>4.264010</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.288263</td>
      <td>0.847955</td>
      <td>1.190410</td>
      <td>43.707804</td>
      <td>6270.380015</td>
      <td>12.971966</td>
      <td>12.940057</td>
      <td>451.857962</td>
      <td>6.884891</td>
      <td>3.299822</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>12.087390</td>
      <td>728.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>35.010000</td>
      <td>1.000000</td>
      <td>2.150000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>29.583220</td>
      <td>2011.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>99.030000</td>
      <td>2.000000</td>
      <td>3.290000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>62.245590</td>
      <td>5097.000000</td>
      <td>13.000000</td>
      <td>12.000000</td>
      <td>344.015000</td>
      <td>5.000000</td>
      <td>5.092500</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>185.092985</td>
      <td>31639.480000</td>
      <td>61.000000</td>
      <td>59.000000</td>
      <td>2166.515200</td>
      <td>35.000000</td>
      <td>17.684004</td>
    </tr>
  </tbody>
</table>
</div>




```python
#double check no zeros
(data_cap_nozero == 0).astype(int).sum(axis=0)
```




    NUM_FMLY_MEMBERS              0
    NUM_CARS_HH                   0
    PARTNERS_SHOPPED              0
    AVG_WKLY_SPND_ALL_PARTNERS    0
    ENDING_PT_BALANCE             0
    CNT_VISITS_GAS_STATION        0
    FUEL_TXNS_L12                 0
    AVG_FUEL_VOL_L12              0
    SHOP_TXNS_L12                 0
    SHOP_AVG_SPEND_L12            0
    dtype: int64




```python
#now apply log to each column using a quick lambda function
data_logged=data_cap_nozero.apply(lambda x: np.log(x))
```


```python
#note check the original vs log version of dataset
data_logged.describe()
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
      <th>NUM_FMLY_MEMBERS</th>
      <th>NUM_CARS_HH</th>
      <th>PARTNERS_SHOPPED</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS</th>
      <th>ENDING_PT_BALANCE</th>
      <th>CNT_VISITS_GAS_STATION</th>
      <th>FUEL_TXNS_L12</th>
      <th>AVG_FUEL_VOL_L12</th>
      <th>SHOP_TXNS_L12</th>
      <th>SHOP_AVG_SPEND_L12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>56924.000000</td>
      <td>52575.000000</td>
      <td>98466.000000</td>
      <td>98465.000000</td>
      <td>98714.000000</td>
      <td>98713.000000</td>
      <td>84831.000000</td>
      <td>84908.000000</td>
      <td>48259.000000</td>
      <td>48259.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.946210</td>
      <td>0.326701</td>
      <td>0.988306</td>
      <td>3.178924</td>
      <td>7.452191</td>
      <td>1.695525</td>
      <td>1.480498</td>
      <td>4.685536</td>
      <td>0.997553</td>
      <td>1.206481</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.489541</td>
      <td>0.897170</td>
      <td>0.458064</td>
      <td>1.339366</td>
      <td>1.610561</td>
      <td>1.180413</td>
      <td>1.263391</td>
      <td>1.477099</td>
      <td>1.011558</td>
      <td>0.750910</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.912023</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>2.492163</td>
      <td>6.590301</td>
      <td>0.693147</td>
      <td>0.000000</td>
      <td>3.555634</td>
      <td>0.000000</td>
      <td>0.765468</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>3.387207</td>
      <td>7.606387</td>
      <td>1.609438</td>
      <td>1.386294</td>
      <td>4.595423</td>
      <td>0.693147</td>
      <td>1.190888</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.386294</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>4.131088</td>
      <td>8.536407</td>
      <td>2.564949</td>
      <td>2.484907</td>
      <td>5.840685</td>
      <td>1.609438</td>
      <td>1.627769</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.791759</td>
      <td>1.386294</td>
      <td>1.791759</td>
      <td>5.220858</td>
      <td>10.362161</td>
      <td>4.110874</td>
      <td>4.077537</td>
      <td>7.680875</td>
      <td>3.555348</td>
      <td>2.872661</td>
    </tr>
  </tbody>
</table>
</div>




```python
#rename columns to differntiate log vs original
data_logged = data_logged.add_suffix('_LOG')
```


```python
data_logged.columns
```




    Index(['NUM_FMLY_MEMBERS_LOG', 'NUM_CARS_HH_LOG', 'PARTNERS_SHOPPED_LOG',
           'AVG_WKLY_SPND_ALL_PARTNERS_LOG', 'ENDING_PT_BALANCE_LOG',
           'CNT_VISITS_GAS_STATION_LOG', 'FUEL_TXNS_L12_LOG',
           'AVG_FUEL_VOL_L12_LOG', 'SHOP_TXNS_L12_LOG', 'SHOP_AVG_SPEND_L12_LOG'],
          dtype='object')



#### STEP 3: Apply median imputation for all logged features


```python
#see how many missing values
data_logged.isnull().sum()
```




    NUM_FMLY_MEMBERS_LOG              41790
    NUM_CARS_HH_LOG                   46139
    PARTNERS_SHOPPED_LOG                248
    AVG_WKLY_SPND_ALL_PARTNERS_LOG      249
    ENDING_PT_BALANCE_LOG                 0
    CNT_VISITS_GAS_STATION_LOG            1
    FUEL_TXNS_L12_LOG                 13883
    AVG_FUEL_VOL_L12_LOG              13806
    SHOP_TXNS_L12_LOG                 50455
    SHOP_AVG_SPEND_L12_LOG            50455
    dtype: int64




```python
data_to_impute = data_logged
```


```python
date_imputed = data_to_impute.apply(lambda x: x.fillna(x.median()))
```


```python
date_imputed.isnull().sum()
```




    NUM_FMLY_MEMBERS_LOG              0
    NUM_CARS_HH_LOG                   0
    PARTNERS_SHOPPED_LOG              0
    AVG_WKLY_SPND_ALL_PARTNERS_LOG    0
    ENDING_PT_BALANCE_LOG             0
    CNT_VISITS_GAS_STATION_LOG        0
    FUEL_TXNS_L12_LOG                 0
    AVG_FUEL_VOL_L12_LOG              0
    SHOP_TXNS_L12_LOG                 0
    SHOP_AVG_SPEND_L12_LOG            0
    dtype: int64




```python
date_imputed.describe()
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
      <th>NUM_FMLY_MEMBERS_LOG</th>
      <th>NUM_CARS_HH_LOG</th>
      <th>PARTNERS_SHOPPED_LOG</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS_LOG</th>
      <th>ENDING_PT_BALANCE_LOG</th>
      <th>CNT_VISITS_GAS_STATION_LOG</th>
      <th>FUEL_TXNS_L12_LOG</th>
      <th>AVG_FUEL_VOL_L12_LOG</th>
      <th>SHOP_TXNS_L12_LOG</th>
      <th>SHOP_AVG_SPEND_L12_LOG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.010729</td>
      <td>0.497978</td>
      <td>0.988584</td>
      <td>3.179450</td>
      <td>7.452191</td>
      <td>1.695524</td>
      <td>1.467249</td>
      <td>4.672933</td>
      <td>0.841964</td>
      <td>1.198511</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.379296</td>
      <td>0.679795</td>
      <td>0.457522</td>
      <td>1.337716</td>
      <td>1.610561</td>
      <td>1.180407</td>
      <td>1.171642</td>
      <td>1.370273</td>
      <td>0.723459</td>
      <td>0.525089</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>-3.912023</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-3.912023</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>2.495093</td>
      <td>6.590301</td>
      <td>0.693147</td>
      <td>0.693147</td>
      <td>3.706228</td>
      <td>0.693147</td>
      <td>1.190888</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>1.098612</td>
      <td>3.387207</td>
      <td>7.606387</td>
      <td>1.609438</td>
      <td>1.386294</td>
      <td>4.595423</td>
      <td>0.693147</td>
      <td>1.190888</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.098612</td>
      <td>0.693147</td>
      <td>1.386294</td>
      <td>4.129178</td>
      <td>8.536407</td>
      <td>2.564949</td>
      <td>2.302585</td>
      <td>5.627432</td>
      <td>0.693147</td>
      <td>1.190888</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.791759</td>
      <td>1.386294</td>
      <td>1.791759</td>
      <td>5.220858</td>
      <td>10.362161</td>
      <td>4.110874</td>
      <td>4.077537</td>
      <td>7.680875</td>
      <td>3.555348</td>
      <td>2.872661</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create flag columns to identify all imputed data
```


```python
impute_flag = data_capped.apply(lambda x: x.isnull())
impute_flag_columns = impute_flag.add_suffix('_IMP')
```


```python
impute_flag_columns.describe()
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
      <th>NUM_FMLY_MEMBERS_IMP</th>
      <th>NUM_CARS_HH_IMP</th>
      <th>PARTNERS_SHOPPED_IMP</th>
      <th>AVG_WKLY_SPND_ALL_PARTNERS_IMP</th>
      <th>ENDING_PT_BALANCE_IMP</th>
      <th>CNT_VISITS_GAS_STATION_IMP</th>
      <th>FUEL_TXNS_L12_IMP</th>
      <th>AVG_FUEL_VOL_L12_IMP</th>
      <th>SHOP_TXNS_L12_IMP</th>
      <th>SHOP_AVG_SPEND_L12_IMP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>56924</td>
      <td>52575</td>
      <td>98466</td>
      <td>98465</td>
      <td>98714</td>
      <td>98713</td>
      <td>84831</td>
      <td>84908</td>
      <td>50455</td>
      <td>50455</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check for missing values - all good
impute_flag_columns.isnull().sum()
```




    NUM_FMLY_MEMBERS_IMP              0
    NUM_CARS_HH_IMP                   0
    PARTNERS_SHOPPED_IMP              0
    AVG_WKLY_SPND_ALL_PARTNERS_IMP    0
    ENDING_PT_BALANCE_IMP             0
    CNT_VISITS_GAS_STATION_IMP        0
    FUEL_TXNS_L12_IMP                 0
    AVG_FUEL_VOL_L12_IMP              0
    SHOP_TXNS_L12_IMP                 0
    SHOP_AVG_SPEND_L12_IMP            0
    dtype: int64



### STEP 4: Apply mean imputation for all other numeric features


```python
numeric_cols = data.select_dtypes(['number']).drop(columns=['TARGET','ID'])
numeric_cols.columns
```




    Index(['NUM_FMLY_MEMBERS', 'NUM_CARS_HH', 'PARTNERS_SHOPPED',
           'AVG_WKLY_SPND_ALL_PARTNERS', 'ENDING_PT_BALANCE',
           'CNT_VISITS_GAS_STATION', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_TXNS_L12', 'AVG_FUEL_VOL_L12', 'SHOP_TXNS_L12',
           'SHOP_AVG_SPEND_L12', 'FUEL_CARD_FLAG', 'DISTANCE_KM'],
          dtype='object')




```python
#capped, logged, and imputed dataset so far
date_imputed.columns
```




    Index(['NUM_FMLY_MEMBERS_LOG', 'NUM_CARS_HH_LOG', 'PARTNERS_SHOPPED_LOG',
           'AVG_WKLY_SPND_ALL_PARTNERS_LOG', 'ENDING_PT_BALANCE_LOG',
           'CNT_VISITS_GAS_STATION_LOG', 'FUEL_TXNS_L12_LOG',
           'AVG_FUEL_VOL_L12_LOG', 'SHOP_TXNS_L12_LOG', 'SHOP_AVG_SPEND_L12_LOG'],
          dtype='object')




```python
#dataset that holds impute flags
impute_flag_columns.columns
```




    Index(['NUM_FMLY_MEMBERS_IMP', 'NUM_CARS_HH_IMP', 'PARTNERS_SHOPPED_IMP',
           'AVG_WKLY_SPND_ALL_PARTNERS_IMP', 'ENDING_PT_BALANCE_IMP',
           'CNT_VISITS_GAS_STATION_IMP', 'FUEL_TXNS_L12_IMP',
           'AVG_FUEL_VOL_L12_IMP', 'SHOP_TXNS_L12_IMP', 'SHOP_AVG_SPEND_L12_IMP'],
          dtype='object')




```python
#these columns are leftover:
data_num_remain = data[['FUEL_CARD_FLAG', 'DISTANCE_KM','PROP_WKDAY_EVE', 'PROP_WKDAY_DAY', 'PROP_WKND']]
```


```python
#for these guys we apply a mean impute & create a flag for missing...
data_to_impute = data_num_remain
```


```python
data_num_remain_imputed = data_to_impute.apply(lambda x: x.fillna(x.mean()))
```


```python
data_num_remain_imputed.describe()
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
      <th>FUEL_CARD_FLAG</th>
      <th>DISTANCE_KM</th>
      <th>PROP_WKDAY_EVE</th>
      <th>PROP_WKDAY_DAY</th>
      <th>PROP_WKND</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
      <td>98714.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.037077</td>
      <td>10.033388</td>
      <td>0.135107</td>
      <td>0.580036</td>
      <td>0.284857</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.188951</td>
      <td>3.689799</td>
      <td>0.228338</td>
      <td>0.343604</td>
      <td>0.307625</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>9.812254</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>10.966584</td>
      <td>0.000000</td>
      <td>0.617021</td>
      <td>0.209302</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>11.713106</td>
      <td>0.200000</td>
      <td>0.904762</td>
      <td>0.433209</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>24.559018</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#check if all missing vlaues are gone
data_num_remain_imputed.isnull().sum()
```




    FUEL_CARD_FLAG    0
    DISTANCE_KM       0
    PROP_WKDAY_EVE    0
    PROP_WKDAY_DAY    0
    PROP_WKND         0
    dtype: int64



#### Next we just need to create missing flags for these guys...


```python
data_num_remain_imp = data_num_remain.apply(lambda x: x.isnull())
data_num_remain_imp_flag = data_num_remain_imp.add_suffix('_IMP')
```


```python
print(data_num_remain_imp_flag.describe())
data_num_remain_imp_flag.isnull().sum()
```

           FUEL_CARD_FLAG_IMP DISTANCE_KM_IMP PROP_WKDAY_EVE_IMP  \
    count               98714           98714              98714   
    unique                  1               1                  1   
    top                 False           False              False   
    freq                98714           98714              98714   
    
           PROP_WKDAY_DAY_IMP PROP_WKND_IMP  
    count               98714         98714  
    unique                  1             1  
    top                 False         False  
    freq                98714         98714  





    FUEL_CARD_FLAG_IMP    0
    DISTANCE_KM_IMP       0
    PROP_WKDAY_EVE_IMP    0
    PROP_WKDAY_DAY_IMP    0
    PROP_WKND_IMP         0
    dtype: int64



## STEP 5: Create a missing flag column for all categorial missing feature


```python
cat_cols = data.select_dtypes(['object'])
cat_cols.columns

```




    Index(['CLIENT_GENDER', 'CLIENT_AGE_BAND', 'CLIENT_REGION',
           'REWARD_HIST_SEGMENT'],
          dtype='object')




```python
cat_cols_miss = cat_cols.apply(lambda x: x.isnull())
cat_cols_miss_flag = cat_cols_miss.add_suffix('_MISS')

```


```python
cat_cols_miss_flag.describe()
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
      <th>CLIENT_GENDER_MISS</th>
      <th>CLIENT_AGE_BAND_MISS</th>
      <th>CLIENT_REGION_MISS</th>
      <th>REWARD_HIST_SEGMENT_MISS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
      <td>98714</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>89002</td>
      <td>88604</td>
      <td>89547</td>
      <td>98164</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_cols_miss_flag.head(10)
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
      <th>CLIENT_GENDER_MISS</th>
      <th>CLIENT_AGE_BAND_MISS</th>
      <th>CLIENT_REGION_MISS</th>
      <th>REWARD_HIST_SEGMENT_MISS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### STEP 6: JOIN EVERYTHING TOGETHER

#### RECAP
> - The outputs of step 1/2/3 were TWO sepeate outputs: date_imputed & impute_flag_columns
> - Step 4 also resulted in TWO outputs: data_num_remain_imputed & data_num_remain_imp_flag
> - Lastly, step 5 resulted in ONE output: cat_cols_miss_flag

#### Our Strategy
> - Join all of these processed features together into 1 wide table and grab the TARGET and ID from origina dataframe
> - Feed this into a classifier


```python
#### We want to join these along the index
combine_1=pd.concat([date_imputed, impute_flag_columns, data_num_remain_imputed, data_num_remain_imp_flag, cat_cols_miss_flag], axis=1)
```


```python
combine_1.columns
```




    Index(['NUM_FMLY_MEMBERS_LOG', 'NUM_CARS_HH_LOG', 'PARTNERS_SHOPPED_LOG',
           'AVG_WKLY_SPND_ALL_PARTNERS_LOG', 'ENDING_PT_BALANCE_LOG',
           'CNT_VISITS_GAS_STATION_LOG', 'FUEL_TXNS_L12_LOG',
           'AVG_FUEL_VOL_L12_LOG', 'SHOP_TXNS_L12_LOG', 'SHOP_AVG_SPEND_L12_LOG',
           'NUM_FMLY_MEMBERS_IMP', 'NUM_CARS_HH_IMP', 'PARTNERS_SHOPPED_IMP',
           'AVG_WKLY_SPND_ALL_PARTNERS_IMP', 'ENDING_PT_BALANCE_IMP',
           'CNT_VISITS_GAS_STATION_IMP', 'FUEL_TXNS_L12_IMP',
           'AVG_FUEL_VOL_L12_IMP', 'SHOP_TXNS_L12_IMP', 'SHOP_AVG_SPEND_L12_IMP',
           'FUEL_CARD_FLAG', 'DISTANCE_KM', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_CARD_FLAG_IMP', 'DISTANCE_KM_IMP',
           'PROP_WKDAY_EVE_IMP', 'PROP_WKDAY_DAY_IMP', 'PROP_WKND_IMP',
           'CLIENT_GENDER_MISS', 'CLIENT_AGE_BAND_MISS', 'CLIENT_REGION_MISS',
           'REWARD_HIST_SEGMENT_MISS'],
          dtype='object')




```python
#check for nulls
combine_1.isnull().sum()
```




    NUM_FMLY_MEMBERS_LOG              0
    NUM_CARS_HH_LOG                   0
    PARTNERS_SHOPPED_LOG              0
    AVG_WKLY_SPND_ALL_PARTNERS_LOG    0
    ENDING_PT_BALANCE_LOG             0
    CNT_VISITS_GAS_STATION_LOG        0
    FUEL_TXNS_L12_LOG                 0
    AVG_FUEL_VOL_L12_LOG              0
    SHOP_TXNS_L12_LOG                 0
    SHOP_AVG_SPEND_L12_LOG            0
    NUM_FMLY_MEMBERS_IMP              0
    NUM_CARS_HH_IMP                   0
    PARTNERS_SHOPPED_IMP              0
    AVG_WKLY_SPND_ALL_PARTNERS_IMP    0
    ENDING_PT_BALANCE_IMP             0
    CNT_VISITS_GAS_STATION_IMP        0
    FUEL_TXNS_L12_IMP                 0
    AVG_FUEL_VOL_L12_IMP              0
    SHOP_TXNS_L12_IMP                 0
    SHOP_AVG_SPEND_L12_IMP            0
    FUEL_CARD_FLAG                    0
    DISTANCE_KM                       0
    PROP_WKDAY_EVE                    0
    PROP_WKDAY_DAY                    0
    PROP_WKND                         0
    FUEL_CARD_FLAG_IMP                0
    DISTANCE_KM_IMP                   0
    PROP_WKDAY_EVE_IMP                0
    PROP_WKDAY_DAY_IMP                0
    PROP_WKND_IMP                     0
    CLIENT_GENDER_MISS                0
    CLIENT_AGE_BAND_MISS              0
    CLIENT_REGION_MISS                0
    REWARD_HIST_SEGMENT_MISS          0
    dtype: int64




```python
#combine with labels
labels = data['TARGET']
combined=pd.concat([combine_1, labels], axis=1)
```


```python
combined.columns
```




    Index(['NUM_FMLY_MEMBERS_LOG', 'NUM_CARS_HH_LOG', 'PARTNERS_SHOPPED_LOG',
           'AVG_WKLY_SPND_ALL_PARTNERS_LOG', 'ENDING_PT_BALANCE_LOG',
           'CNT_VISITS_GAS_STATION_LOG', 'FUEL_TXNS_L12_LOG',
           'AVG_FUEL_VOL_L12_LOG', 'SHOP_TXNS_L12_LOG', 'SHOP_AVG_SPEND_L12_LOG',
           'NUM_FMLY_MEMBERS_IMP', 'NUM_CARS_HH_IMP', 'PARTNERS_SHOPPED_IMP',
           'AVG_WKLY_SPND_ALL_PARTNERS_IMP', 'ENDING_PT_BALANCE_IMP',
           'CNT_VISITS_GAS_STATION_IMP', 'FUEL_TXNS_L12_IMP',
           'AVG_FUEL_VOL_L12_IMP', 'SHOP_TXNS_L12_IMP', 'SHOP_AVG_SPEND_L12_IMP',
           'FUEL_CARD_FLAG', 'DISTANCE_KM', 'PROP_WKDAY_EVE', 'PROP_WKDAY_DAY',
           'PROP_WKND', 'FUEL_CARD_FLAG_IMP', 'DISTANCE_KM_IMP',
           'PROP_WKDAY_EVE_IMP', 'PROP_WKDAY_DAY_IMP', 'PROP_WKND_IMP',
           'CLIENT_GENDER_MISS', 'CLIENT_AGE_BAND_MISS', 'CLIENT_REGION_MISS',
           'REWARD_HIST_SEGMENT_MISS', 'TARGET'],
          dtype='object')



### STEP 7: FIT A CLASSIFIER



```python
# skelearn expects data to come in as Y and a feature vector of Xs

#we have all our features in a dataframe DF (without the target colum)
df=combine_1

#we put our label in a df Y
y=data['TARGET']

```


```python
#we create simple train and test set (we'll cover cross-validation etc. later)
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

```


```python
# fit a logistic regression
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=1e5)
model = logistic.fit(X_train, y_train)

```


```python
# use it to predict what woudl happen on unseen data aka...the test dataset aka hold out set
y_pred = logistic.predict(X_test)
```


```python
#get the confusion matrix
sklearn.metrics.confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
```




    array([[21588,  1768],
           [ 3082,  3177]], dtype=int64)




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
