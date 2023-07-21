```python
%matplotlib inline
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
```


```python
medical = pd.read_csv('./data/Medicare_Hospital_Spending_by_Claim (1).csv')
medical.tail(3)
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
      <th>Hospital Name</th>
      <th>Provider Number</th>
      <th>State</th>
      <th>Period</th>
      <th>Claim Type</th>
      <th>Avg Spending Per Episode (Hospital)</th>
      <th>Avg Spending Per Episode (State)</th>
      <th>Avg Spending Per Episode (Nation)</th>
      <th>Percent of Spending (Hospital)</th>
      <th>Percent of Spending (State)</th>
      <th>Percent of Spending (Nation)</th>
      <th>Measure Start Date</th>
      <th>Measure End Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>70595</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Durable Medical Equipment</td>
      <td>$61</td>
      <td>$114</td>
      <td>$101</td>
      <td>0.35%</td>
      <td>0.53%</td>
      <td>0.5%</td>
      <td>01/01/2014</td>
      <td>12/31/2014</td>
    </tr>
    <tr>
      <td>70596</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Carrier</td>
      <td>$1168</td>
      <td>$1231</td>
      <td>$1083</td>
      <td>6.65%</td>
      <td>5.73%</td>
      <td>5.41%</td>
      <td>01/01/2014</td>
      <td>12/31/2014</td>
    </tr>
    <tr>
      <td>70597</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>Complete Episode</td>
      <td>Total</td>
      <td>$17568</td>
      <td>$21484</td>
      <td>$20025</td>
      <td>100%</td>
      <td>100%</td>
      <td>100%</td>
      <td>01/01/2014</td>
      <td>12/31/2014</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Question 1
medical = medical.drop(['Measure Start Date', 'Measure End Date'], axis = 1)
medical.tail(3)
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
      <th>Hospital Name</th>
      <th>Provider Number</th>
      <th>State</th>
      <th>Period</th>
      <th>Claim Type</th>
      <th>Avg Spending Per Episode (Hospital)</th>
      <th>Avg Spending Per Episode (State)</th>
      <th>Avg Spending Per Episode (Nation)</th>
      <th>Percent of Spending (Hospital)</th>
      <th>Percent of Spending (State)</th>
      <th>Percent of Spending (Nation)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>70595</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Durable Medical Equipment</td>
      <td>$61</td>
      <td>$114</td>
      <td>$101</td>
      <td>0.35%</td>
      <td>0.53%</td>
      <td>0.5%</td>
    </tr>
    <tr>
      <td>70596</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Carrier</td>
      <td>$1168</td>
      <td>$1231</td>
      <td>$1083</td>
      <td>6.65%</td>
      <td>5.73%</td>
      <td>5.41%</td>
    </tr>
    <tr>
      <td>70597</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>Complete Episode</td>
      <td>Total</td>
      <td>$17568</td>
      <td>$21484</td>
      <td>$20025</td>
      <td>100%</td>
      <td>100%</td>
      <td>100%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Question 2
medical.columns = ['Hospital', 'Provider ID', 'State', 'Period', 'Claim Type', 'Avg Spending Hospital', 'Avg Spending State', 'Avg Spending Nation', 'Percent Spending Hospital', 'Percent Spending State', 'Percent Spending Nation']
medical.tail(3)
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
      <th>Hospital</th>
      <th>Provider ID</th>
      <th>State</th>
      <th>Period</th>
      <th>Claim Type</th>
      <th>Avg Spending Hospital</th>
      <th>Avg Spending State</th>
      <th>Avg Spending Nation</th>
      <th>Percent Spending Hospital</th>
      <th>Percent Spending State</th>
      <th>Percent Spending Nation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>70595</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Durable Medical Equipment</td>
      <td>$61</td>
      <td>$114</td>
      <td>$101</td>
      <td>0.35%</td>
      <td>0.53%</td>
      <td>0.5%</td>
    </tr>
    <tr>
      <td>70596</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>1 through 30 days After Discharge from Index H...</td>
      <td>Carrier</td>
      <td>$1168</td>
      <td>$1231</td>
      <td>$1083</td>
      <td>6.65%</td>
      <td>5.73%</td>
      <td>5.41%</td>
    </tr>
    <tr>
      <td>70597</td>
      <td>RESOLUTE HEALTH HOSPITAL</td>
      <td>670098</td>
      <td>TX</td>
      <td>Complete Episode</td>
      <td>Total</td>
      <td>$17568</td>
      <td>$21484</td>
      <td>$20025</td>
      <td>100%</td>
      <td>100%</td>
      <td>100%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Question 3
medical2 = medical[medical['State'] == 'TX']
medical2.head(3)
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
      <th>Hospital</th>
      <th>Provider ID</th>
      <th>State</th>
      <th>Period</th>
      <th>Claim Type</th>
      <th>Avg Spending Hospital</th>
      <th>Avg Spending State</th>
      <th>Avg Spending Nation</th>
      <th>Percent Spending Hospital</th>
      <th>Percent Spending State</th>
      <th>Percent Spending Nation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>58322</td>
      <td>PROVIDENCE MEMORIAL HOSPITAL</td>
      <td>450002</td>
      <td>TX</td>
      <td>1 to 3 days Prior to Index Hospital Admission</td>
      <td>Home Health Agency</td>
      <td>$52</td>
      <td>$23</td>
      <td>$13</td>
      <td>0.27%</td>
      <td>0.11%</td>
      <td>0.07%</td>
    </tr>
    <tr>
      <td>58323</td>
      <td>PROVIDENCE MEMORIAL HOSPITAL</td>
      <td>450002</td>
      <td>TX</td>
      <td>1 to 3 days Prior to Index Hospital Admission</td>
      <td>Hospice</td>
      <td>$0</td>
      <td>$1</td>
      <td>$1</td>
      <td>0%</td>
      <td>0.01%</td>
      <td>0%</td>
    </tr>
    <tr>
      <td>58324</td>
      <td>PROVIDENCE MEMORIAL HOSPITAL</td>
      <td>450002</td>
      <td>TX</td>
      <td>1 to 3 days Prior to Index Hospital Admission</td>
      <td>Inpatient</td>
      <td>$2</td>
      <td>$5</td>
      <td>$5</td>
      <td>0.01%</td>
      <td>0.02%</td>
      <td>0.03%</td>
    </tr>
  </tbody>
</table>
</div>




```python
medical2.dtypes
```




    Hospital                     object
    Provider ID                   int64
    State                        object
    Period                       object
    Claim Type                   object
    Avg Spending Hospital        object
    Avg Spending State           object
    Avg Spending Nation          object
    Percent Spending Hospital    object
    Percent Spending State       object
    Percent Spending Nation      object
    dtype: object




```python
medical2[['Percent Spending Hospital', 'Percent Spending State', 'Percent Spending Nation']] = medical2[['Percent Spending Hospital', 'Percent Spending State', 'Percent Spending Nation'
                    ]].replace('%','',regex=True).astype(float)
medical2[['Avg Spending Hospital', 'Avg Spending State', 'Avg Spending Nation']] = medical2[['Avg Spending Hospital', 'Avg Spending State', 'Avg Spending Nation'
                      ]].replace('\$','',regex = True).astype(float)
```


```python
medical2.dtypes
```




    Hospital                      object
    Provider ID                    int64
    State                         object
    Period                        object
    Claim Type                    object
    Avg Spending Hospital        float64
    Avg Spending State           float64
    Avg Spending Nation          float64
    Percent Spending Hospital    float64
    Percent Spending State       float64
    Percent Spending Nation      float64
    dtype: object




```python
# Question 4
medical2[['Avg Spending Hospital','Hospital','Claim Type']].groupby(["Claim Type",'Hospital']).mean()
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
      <th></th>
      <th>Avg Spending Hospital</th>
    </tr>
    <tr>
      <th>Claim Type</th>
      <th>Hospital</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Carrier</td>
      <td>ABILENE REGIONAL MEDICAL CENTER</td>
      <td>988.666667</td>
    </tr>
    <tr>
      <td>ANSON GENERAL HOSPITAL</td>
      <td>341.000000</td>
    </tr>
    <tr>
      <td>ARISE AUSTIN MEDICAL CENTER</td>
      <td>1105.666667</td>
    </tr>
    <tr>
      <td>BAPTIST BEAUMONT HOSPITAL</td>
      <td>935.666667</td>
    </tr>
    <tr>
      <td>BAPTIST EMERGENCY HOSPITAL</td>
      <td>468.333333</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td rowspan="5" valign="top">Total</td>
      <td>WILLIAM P. CLEMENTS JR. UNIVERSITY HOSPITAL</td>
      <td>21371.000000</td>
    </tr>
    <tr>
      <td>WILSON N. JONES REGIONAL MEDICAL CENTER</td>
      <td>21467.000000</td>
    </tr>
    <tr>
      <td>WISE REGIONAL HEALTH SYSTEM</td>
      <td>22756.000000</td>
    </tr>
    <tr>
      <td>WOMANS HOSPITAL OF TEXAS,THE</td>
      <td>10803.000000</td>
    </tr>
    <tr>
      <td>WOODLAND HEIGHTS MEDICAL CENTER</td>
      <td>18199.000000</td>
    </tr>
  </tbody>
</table>
<p>2352 rows × 1 columns</p>
</div>




```python
#Question 5
medical2[['Percent Spending Hospital','Hospital','Claim Type']].groupby(["Claim Type",'Hospital']).agg(['max', 'mean'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">Percent Spending Hospital</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>max</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>Claim Type</th>
      <th>Hospital</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Carrier</td>
      <td>ABILENE REGIONAL MEDICAL CENTER</td>
      <td>7.12</td>
      <td>4.750000</td>
    </tr>
    <tr>
      <td>ANSON GENERAL HOSPITAL</td>
      <td>4.77</td>
      <td>2.960000</td>
    </tr>
    <tr>
      <td>ARISE AUSTIN MEDICAL CENTER</td>
      <td>10.45</td>
      <td>5.576667</td>
    </tr>
    <tr>
      <td>BAPTIST BEAUMONT HOSPITAL</td>
      <td>5.83</td>
      <td>4.763333</td>
    </tr>
    <tr>
      <td>BAPTIST EMERGENCY HOSPITAL</td>
      <td>9.76</td>
      <td>5.796667</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td rowspan="5" valign="top">Total</td>
      <td>WILLIAM P. CLEMENTS JR. UNIVERSITY HOSPITAL</td>
      <td>100.00</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>WILSON N. JONES REGIONAL MEDICAL CENTER</td>
      <td>100.00</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>WISE REGIONAL HEALTH SYSTEM</td>
      <td>100.00</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>WOMANS HOSPITAL OF TEXAS,THE</td>
      <td>100.00</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <td>WOODLAND HEIGHTS MEDICAL CENTER</td>
      <td>100.00</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
<p>2352 rows × 2 columns</p>
</div>




```python
# Question 6
# An analytical model is meant to predict or explain the behaviour of something
# An analytical model is something that takes in an input processes it and provides an output

```


```python
# Question 7
# In a supervised learning the features are accompanied by labels to categorize data
# In unsupervised learning the features are not accompanied by labels and categories need to be defined

```


```python
# Question 8
# Classification uses predefined labels in which objects are assigned
# Clustering uses the similarities between objects to group accordingly
```


```python
# Question 9
# The reason it is called Ordinary Least Squares is because the 
# regression minimizes the sum of the squares between the observed and predicted values 
# of the dependent variable to estimate their relationship compared to other regression models
# It does this by using a line of best fit that minimizes the sum of squares

```


```python
# Question 10
# 1. Understand that the data comes from people, and can do potential harm when analysed the wrong way
# 2. Prepare for possible privacy breaches to minimize harm
# 3. Identify possible vectors of reidentification in your data.
# 4. Share data as specified in research protocols, but proactively address concerns of potential harm 
# 5. Understand the development of your data, and that their may be multiple interpretations
# 6. Engage colleagues and discuss difficult ethical decisions
# 7. ESTABLISH CODES OF CONDUCT
# 8. Prepare for audits and design you data and model accordingly
# 9. Keep in mind the societal outcomes of your data and how it may affect others
# 10.Understand the bigger picture and importance of your data in a greater context

```
