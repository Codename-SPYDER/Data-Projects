```python
import numpy as np
import pandas as pd
```


```python
Boeing = pd.read_csv('./Basic_Methods-(Data)/Lab1/BoeingStock.csv')

```


```python
IBM = pd.read_csv('./Basic_Methods-(Data)/Lab1/IBMStock.csv')
GE = pd.read_csv('./Basic_Methods-(Data)/Lab1/GEStock.csv')
ProctorGamble = pd.read_csv('./Basic_Methods-(Data)/Lab1/ProcterGambleStock.csv')
CocaCola = pd.read_csv('./Basic_Methods-(Data)/Lab1/CocaColaStock.csv')

```


```python
Boeing.tail(5)
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
      <th>Date</th>
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>475</td>
      <td>8/1/09</td>
      <td>45.994286</td>
    </tr>
    <tr>
      <td>476</td>
      <td>9/1/09</td>
      <td>51.362857</td>
    </tr>
    <tr>
      <td>477</td>
      <td>10/1/09</td>
      <td>51.159091</td>
    </tr>
    <tr>
      <td>478</td>
      <td>11/1/09</td>
      <td>50.696500</td>
    </tr>
    <tr>
      <td>479</td>
      <td>12/1/09</td>
      <td>55.028636</td>
    </tr>
  </tbody>
</table>
</div>




```python
Boeing['Date'] = pd.to_datetime(Boeing['Date'])
Boeing.set_index('Date', inplace=True)

IBM['Date'] = pd.to_datetime(IBM['Date'])
IBM.set_index('Date', inplace=True)

GE['Date'] = pd.to_datetime(GE['Date'])
GE.set_index('Date', inplace=True)

ProctorGamble['Date'] = pd.to_datetime(ProctorGamble['Date'])
ProctorGamble.set_index('Date', inplace=True)

CocaCola['Date'] = pd.to_datetime(CocaCola['Date'])
CocaCola.set_index('Date', inplace=True)




```


```python
Boeing.tail(5)
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
      <th>StockPrice</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2009-08-01</td>
      <td>45.994286</td>
    </tr>
    <tr>
      <td>2009-09-01</td>
      <td>51.362857</td>
    </tr>
    <tr>
      <td>2009-10-01</td>
      <td>51.159091</td>
    </tr>
    <tr>
      <td>2009-11-01</td>
      <td>50.696500</td>
    </tr>
    <tr>
      <td>2009-12-01</td>
      <td>55.028636</td>
    </tr>
  </tbody>
</table>
</div>



<ins>**Warm-Up/ Basic Statistics Questions**




```python
Boeing.info()
Boeing.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 480 entries, 1970-01-01 to 2009-12-01
    Data columns (total 1 columns):
    StockPrice    480 non-null float64
    dtypes: float64(1)
    memory usage: 7.5 KB





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
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>46.592934</td>
    </tr>
    <tr>
      <td>std</td>
      <td>19.891837</td>
    </tr>
    <tr>
      <td>min</td>
      <td>12.736364</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>34.642274</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>44.883398</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>57.214486</td>
    </tr>
    <tr>
      <td>max</td>
      <td>107.280000</td>
    </tr>
  </tbody>
</table>
</div>




```python
IBM.info()
IBM.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 480 entries, 1970-01-01 to 2009-12-01
    Data columns (total 1 columns):
    StockPrice    480 non-null float64
    dtypes: float64(1)
    memory usage: 7.5 KB





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
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>144.375030</td>
    </tr>
    <tr>
      <td>std</td>
      <td>87.822078</td>
    </tr>
    <tr>
      <td>min</td>
      <td>43.395000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>88.343929</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>112.114595</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>165.407284</td>
    </tr>
    <tr>
      <td>max</td>
      <td>438.901579</td>
    </tr>
  </tbody>
</table>
</div>




```python
GE.info()
GE.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 480 entries, 1970-01-01 to 2009-12-01
    Data columns (total 1 columns):
    StockPrice    480 non-null float64
    dtypes: float64(1)
    memory usage: 7.5 KB





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
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>59.303504</td>
    </tr>
    <tr>
      <td>std</td>
      <td>23.992551</td>
    </tr>
    <tr>
      <td>min</td>
      <td>9.293636</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>44.214405</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>55.812045</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>72.226201</td>
    </tr>
    <tr>
      <td>max</td>
      <td>156.843684</td>
    </tr>
  </tbody>
</table>
</div>




```python
ProctorGamble.info()
ProctorGamble.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 480 entries, 1970-01-01 to 2009-12-01
    Data columns (total 1 columns):
    StockPrice    480 non-null float64
    dtypes: float64(1)
    memory usage: 7.5 KB





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
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>77.704516</td>
    </tr>
    <tr>
      <td>std</td>
      <td>18.194140</td>
    </tr>
    <tr>
      <td>min</td>
      <td>46.884545</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>62.478663</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>78.336077</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>89.468375</td>
    </tr>
    <tr>
      <td>max</td>
      <td>149.620000</td>
    </tr>
  </tbody>
</table>
</div>




```python
CocaCola.info()
CocaCola.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 480 entries, 1970-01-01 to 2009-12-01
    Data columns (total 1 columns):
    StockPrice    480 non-null float64
    dtypes: float64(1)
    memory usage: 7.5 KB





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
      <th>StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>60.029730</td>
    </tr>
    <tr>
      <td>std</td>
      <td>25.166291</td>
    </tr>
    <tr>
      <td>min</td>
      <td>30.057143</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>42.755595</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>51.436988</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>69.617192</td>
    </tr>
    <tr>
      <td>max</td>
      <td>146.584286</td>
    </tr>
  </tbody>
</table>
</div>




```python
Boeing_5subset = Boeing.last('5Y')
```


```python
Boeing_5subset.median()
```




    StockPrice    69.675667
    dtype: float64



<ins>**Basic Plotting Questions (Part 1)**


```python

import matplotlib.pyplot as plt
```


```python
CocaCola.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11a0efc90>




    
![png](output_16_1.png)
    


<ins>**Basic Plotting Questions (Part 2)**


```python
CocaCola['P&G_StockPrice'] = ProctorGamble['StockPrice']
```


```python
CocaCola.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11aa82c50>




    
![png](output_19_1.png)
    


<ins>**Data Visualization 1995 - 2005**


```python
CocaCola['Boeing_StockPrice'] = Boeing['StockPrice']
```


```python
CocaCola['GE_StockPrice'] = GE['StockPrice']
```


```python
CocaCola['IBM_StockPrice'] = IBM['StockPrice']
```


```python
CocaCola.rename(columns = {'StockPrice':'CC_StockPrice'}, inplace = True)
```


```python
CocaCola.tail(5)
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
      <th>CC_StockPrice</th>
      <th>P&amp;G_StockPrice</th>
      <th>Boeing_StockPrice</th>
      <th>GE_StockPrice</th>
      <th>IBM_StockPrice</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2009-08-01</td>
      <td>49.150952</td>
      <td>53.098095</td>
      <td>45.994286</td>
      <td>14.023333</td>
      <td>118.430952</td>
    </tr>
    <tr>
      <td>2009-09-01</td>
      <td>51.588571</td>
      <td>55.764762</td>
      <td>51.362857</td>
      <td>15.591905</td>
      <td>119.055714</td>
    </tr>
    <tr>
      <td>2009-10-01</td>
      <td>54.090000</td>
      <td>57.518182</td>
      <td>51.159091</td>
      <td>15.797727</td>
      <td>122.239546</td>
    </tr>
    <tr>
      <td>2009-11-01</td>
      <td>55.908000</td>
      <td>61.297000</td>
      <td>50.696500</td>
      <td>15.508000</td>
      <td>125.273500</td>
    </tr>
    <tr>
      <td>2009-12-01</td>
      <td>57.790909</td>
      <td>62.052727</td>
      <td>55.028636</td>
      <td>15.754545</td>
      <td>128.896364</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_stock = CocaCola['1995':'2005']
```


```python
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,8)
```


```python
all_stock.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11ac07f90>




    
![png](output_28_1.png)
    



```python
Asian_Crisis = CocaCola['1997']
```


```python
Asian_Crisis.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11af40dd0>




    
![png](output_30_1.png)
    



```python
all_stock['2003':'2005'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b135c10>




    
![png](output_31_1.png)
    


<ins>**Monthly Trend Analysis**


```python
months = CocaCola.index.month
monthly_avg = CocaCola.groupby(months).mean()

```


```python
monthly_avg.tail(12)
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
      <th>CC_StockPrice</th>
      <th>P&amp;G_StockPrice</th>
      <th>Boeing_StockPrice</th>
      <th>GE_StockPrice</th>
      <th>IBM_StockPrice</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>60.368487</td>
      <td>79.617984</td>
      <td>46.510974</td>
      <td>62.045106</td>
      <td>150.238423</td>
    </tr>
    <tr>
      <td>2</td>
      <td>60.734754</td>
      <td>79.025755</td>
      <td>46.892233</td>
      <td>62.520805</td>
      <td>152.693993</td>
    </tr>
    <tr>
      <td>3</td>
      <td>62.071354</td>
      <td>77.347607</td>
      <td>46.882076</td>
      <td>63.150548</td>
      <td>152.432690</td>
    </tr>
    <tr>
      <td>4</td>
      <td>62.688882</td>
      <td>77.686708</td>
      <td>47.046860</td>
      <td>64.480092</td>
      <td>152.116824</td>
    </tr>
    <tr>
      <td>5</td>
      <td>61.443581</td>
      <td>77.859578</td>
      <td>48.137160</td>
      <td>60.871351</td>
      <td>151.502194</td>
    </tr>
    <tr>
      <td>6</td>
      <td>60.812084</td>
      <td>77.392751</td>
      <td>47.385255</td>
      <td>56.468439</td>
      <td>139.090676</td>
    </tr>
    <tr>
      <td>7</td>
      <td>58.983460</td>
      <td>76.645559</td>
      <td>46.553602</td>
      <td>56.733493</td>
      <td>139.067018</td>
    </tr>
    <tr>
      <td>8</td>
      <td>58.880139</td>
      <td>76.822663</td>
      <td>46.863107</td>
      <td>56.503149</td>
      <td>140.145475</td>
    </tr>
    <tr>
      <td>9</td>
      <td>57.600238</td>
      <td>76.623845</td>
      <td>46.304854</td>
      <td>56.239131</td>
      <td>139.088527</td>
    </tr>
    <tr>
      <td>10</td>
      <td>57.938868</td>
      <td>76.679035</td>
      <td>45.216035</td>
      <td>56.238968</td>
      <td>137.346553</td>
    </tr>
    <tr>
      <td>11</td>
      <td>59.102683</td>
      <td>78.456104</td>
      <td>45.149903</td>
      <td>57.288795</td>
      <td>138.018682</td>
    </tr>
    <tr>
      <td>12</td>
      <td>59.732227</td>
      <td>78.296608</td>
      <td>46.173146</td>
      <td>59.102174</td>
      <td>140.759310</td>
    </tr>
  </tbody>
</table>
</div>




```python
CocaCola.describe()

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
      <th>CC_StockPrice</th>
      <th>P&amp;G_StockPrice</th>
      <th>Boeing_StockPrice</th>
      <th>GE_StockPrice</th>
      <th>IBM_StockPrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>60.029730</td>
      <td>77.704516</td>
      <td>46.592934</td>
      <td>59.303504</td>
      <td>144.375030</td>
    </tr>
    <tr>
      <td>std</td>
      <td>25.166291</td>
      <td>18.194140</td>
      <td>19.891837</td>
      <td>23.992551</td>
      <td>87.822078</td>
    </tr>
    <tr>
      <td>min</td>
      <td>30.057143</td>
      <td>46.884545</td>
      <td>12.736364</td>
      <td>9.293636</td>
      <td>43.395000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>42.755595</td>
      <td>62.478663</td>
      <td>34.642274</td>
      <td>44.214405</td>
      <td>88.343929</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>51.436988</td>
      <td>78.336077</td>
      <td>44.883398</td>
      <td>55.812045</td>
      <td>112.114595</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>69.617192</td>
      <td>89.468375</td>
      <td>57.214486</td>
      <td>72.226201</td>
      <td>165.407284</td>
    </tr>
    <tr>
      <td>max</td>
      <td>146.584286</td>
      <td>149.620000</td>
      <td>107.280000</td>
      <td>156.843684</td>
      <td>438.901579</td>
    </tr>
  </tbody>
</table>
</div>




```python
higher_avg = monthly_avg.copy()
```


```python
print(higher_avg)
```

          CC_StockPrice  P&G_StockPrice  Boeing_StockPrice  GE_StockPrice  \
    Date                                                                    
    1         60.368487       79.617984          46.510974      62.045106   
    2         60.734754       79.025755          46.892233      62.520805   
    3         62.071354       77.347607          46.882076      63.150548   
    4         62.688882       77.686708          47.046860      64.480092   
    5         61.443581       77.859578          48.137160      60.871351   
    6         60.812084       77.392751          47.385255      56.468439   
    7         58.983460       76.645559          46.553602      56.733493   
    8         58.880139       76.822663          46.863107      56.503149   
    9         57.600238       76.623845          46.304854      56.239131   
    10        57.938868       76.679035          45.216035      56.238968   
    11        59.102683       78.456104          45.149903      57.288795   
    12        59.732227       78.296608          46.173146      59.102174   
    
          IBM_StockPrice  
    Date                  
    1         150.238423  
    2         152.693993  
    3         152.432690  
    4         152.116824  
    5         151.502194  
    6         139.090676  
    7         139.067018  
    8         140.145475  
    9         139.088527  
    10        137.346553  
    11        138.018682  
    12        140.759310  



```python
higher_avg.loc[higher_avg.CC_StockPrice <= 60.03, "CC_StockPrice"] = 0
higher_avg.loc[higher_avg.Boeing_StockPrice <= 46.60, "Boeing_StockPrice"] = 0
higher_avg.loc[higher_avg.GE_StockPrice <= 59.30, "GE_StockPrice"] = 0
higher_avg.loc[higher_avg.IBM_StockPrice <= 144.38, "IBM_StockPrice"] = 0

```


```python
higher_avg.loc[higher_avg['P&G_StockPrice'] <= 77.70, 'P&G_StockPrice'] = 0
```


```python
higher_avg
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
      <th>CC_StockPrice</th>
      <th>P&amp;G_StockPrice</th>
      <th>Boeing_StockPrice</th>
      <th>GE_StockPrice</th>
      <th>IBM_StockPrice</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>60.368487</td>
      <td>79.617984</td>
      <td>0.000000</td>
      <td>62.045106</td>
      <td>150.238423</td>
    </tr>
    <tr>
      <td>2</td>
      <td>60.734754</td>
      <td>79.025755</td>
      <td>46.892233</td>
      <td>62.520805</td>
      <td>152.693993</td>
    </tr>
    <tr>
      <td>3</td>
      <td>62.071354</td>
      <td>0.000000</td>
      <td>46.882076</td>
      <td>63.150548</td>
      <td>152.432690</td>
    </tr>
    <tr>
      <td>4</td>
      <td>62.688882</td>
      <td>0.000000</td>
      <td>47.046860</td>
      <td>64.480092</td>
      <td>152.116824</td>
    </tr>
    <tr>
      <td>5</td>
      <td>61.443581</td>
      <td>77.859578</td>
      <td>48.137160</td>
      <td>60.871351</td>
      <td>151.502194</td>
    </tr>
    <tr>
      <td>6</td>
      <td>60.812084</td>
      <td>0.000000</td>
      <td>47.385255</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>46.863107</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.000000</td>
      <td>78.456104</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.000000</td>
      <td>78.296608</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
higher_avg.plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11b66b410>




    
![png](output_41_1.png)
    



```python

```
