***
## **IMPORTING LIBRARIES**
***


```python
## import all required libraries
import numpy as np ## for linear algebra
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt ## for visualisation
%matplotlib inline
import seaborn as sns ## for visualisation
import warnings ## for filterout warnings
warnings.filterwarnings("ignore")
import datetime as dt
```

***
## **LOADING THE DATA**
***


```python
# Loading the data
sales_train = pd.read_csv('train.csv',parse_dates=['date'] ,index_col=['date'])
sales_test = pd.read_csv('test.csv',parse_dates=['date'] ,index_col=['date'])
```


```python
sales_df=sales_train.copy()
```

***
## **EXAMINING THE DATA**
***


```python
#sales_train = sales_train.set_index('date')
```


```python
# check the first couple of rows in the data
sales_train.head()
```


```python
sales_test.head()
```


```python
sales.shape
```


```python
# check train info (data types and number of rows)
sales.info()
```


```python
sales.describe()
```


```python
# checking for null values for train and test
sales.isnull().sum()
```


```python
# How many stores and items are there?
sales['store'].nunique(), sales_test['store'].nunique(),sales['item'].nunique(), sales_test['item'].nunique()
```

***
## **EXPLORATORY DATA ANALYSIS**
***


```python
sales = sales_train.loc[sales_train['store']==1]
```


```python
sales=sales_train.loc[sales_train['item']==1]
```


```python
sales.shape
```


```python
# Plot showing the sales data trend, we can see that the sales ahve been trending upwords for the past 5 years with highs and lows at the mid year and beginning of year mark.
plt.figure(figsize=(20,10))
plt.plot(sales.sales.resample('w').sum(),label="sales")
plt.title("Store 1 Item Sales for 2015-2018")
```


```python
# to calcualte the 10 day moving average to smooth out the data points - 21.28 
sales['sales'].mean = sales['sales'].rolling(window=10).mean()
```


```python
#sales_train['sales'].mean.plot()
plt.figure(figsize=(20,10))
plt.plot(sales['sales'].mean.resample('W').sum(),label="sales")
plt.title("Store 1 Item MA10 Sales for 2015-2018")
```

## create dataset for every shop and visualize sales trend

shop1 = sales_train[sales_train.store==1]['sales'].sort_index(ascending=True)
shop2 = sales_train[sales_train.store==2]['sales'].sort_index(ascending=True)
shop3 = sales_train[sales_train.store==3]['sales'].sort_index(ascending=True)
shop4 = sales_train[sales_train.store==4]['sales'].sort_index(ascending=True)
shop5 = sales_train[sales_train.store==5]['sales'].sort_index(ascending=True)
shop6 = sales_train[sales_train.store==6]['sales'].sort_index(ascending=True)
shop7 = sales_train[sales_train.store==7]['sales'].sort_index(ascending=True)
shop8 = sales_train[sales_train.store==8]['sales'].sort_index(ascending=True)
shop9 = sales_train[sales_train.store==9]['sales'].sort_index(ascending=True)
shop10 = sales_train[sales_train.store==10]['sales'].sort_index(ascending=True)


fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(10,figsize=(20,70))
shop1.resample('w').sum().plot(ax=ax1,title='Store 1 Sales')
shop2.resample('w').sum().plot(ax=ax2,title='Store 2 Sales')
shop3.resample('w').sum().plot(ax=ax3,title='Store 3 Sales')
shop4.resample('w').sum().plot(ax=ax4,title='Store 4 Sales')
shop5.resample('w').sum().plot(ax=ax5,title='Store 5 Sales')
shop6.resample('w').sum().plot(ax=ax6,title='Store 6 Sales')
shop7.resample('w').sum().plot(ax=ax7,title='Store 7 Sales')
shop8.resample('w').sum().plot(ax=ax8,title='Store 8 Sales')
shop9.resample('w').sum().plot(ax=ax9,title='Store 9 Sales')
shop10.resample('w').sum().plot(ax=ax10,title='Store 10 Sales')
plt.show()


```python
# To plot the Quantile-Quantile Plot (QQ Plot), used to determine whether a data set is distributed a certain way and usually showcases how the data fits a normal distribution
import scipy.stats
import pylab
```


```python
scipy.stats.probplot(sales.sales, plot = pylab)
```

The QQ plot takes all the values a variable can take, and arranges them in accending order, the y-axis expresses the price with the highest being at the top and lowest at the bottom. 
The x-axis represents how many standard deviations away from the mean these values are. The red diagonal line represents what the data points shoudl follow if they are noramlly distributed. 
In our case our data is not noramlly distributed since there are more values around the zero mark then we should, there for our data is not normally distributed and we cannot use normal distribution to make forcasts.

***

***


```python
del sales['item'], sales['store']
```


```python
sales.describe()
```

***
### **WHITE NOISE**


```python
wn = np.random.normal(loc=sales_train_df.sales.mean(), scale = sales_train_df.sales.std(), size = len(sales_train_df))
```


```python
sales_train_df['wn'] = wn
```


```python
sales_train_df.describe()
```


```python
plt.figure(figsize=(20,10))
plt.plot(sales_train_df.wn.resample('w').sum(),label="sales")
plt.title("White Noise Time-Series")
```


```python
plt.figure(figsize=(20,10))
plt.plot(sales_train_df.sales.resample('w').sum(),label="sales")
plt.ylim(50000,300000)
plt.title("Sales 2015-2018")
```


***


### **STATIONARITY OR NON-STATIONARITY**


```python
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(sales['sales'], autolag = 'AIC')
print("1. ADF: ", dftest[0])
print("2. P-value: ", dftest[1])
print("3. Num of Lags: ", dftest[2])
print("4. Num of Oberservations Used for ADF Regression and Critical Values Calculations: ", dftest[3])
print("5. Critical Values:")
for key , val in dftest[4].items():
    print("\t", key, ": ", val)

```

***
### **SEASONALITY**



```python
import statsmodels.api as sm
```


```python
y = sales['sales'].resample('w').mean() 

result = sm.tsa.seasonal_decompose(y, model='additive')
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(20, 20)
```


```python
z = sales['sales'].resample('w').mean() 

result = sm.tsa.seasonal_decompose(z, model='multiplicative')
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(20, 20)
```

***
## **ACF**



```python
#The code below takes 5 mins to run
```


```python
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
pacf=plot_pacf(sales['sales'], lags=25)
acf=plot_acf(sales['sales'], lags=25)
```


```python
train = sales.iloc[0:14608] ## train 
test = sales.iloc[14608:] ## test
```


```python
from pmdarima import auto_arima # --> pip install this
import warnings
warnings.filterwarnings("ignore")
```


```python
stepwise_fit = auto_arima(sales['sales'], trace=True, suppress_warnings=True)
stepwise_fit.summary()
```


```python
from statsmodels.tsa.arima_model import ARIMA
```


```python
print(sales.shape)
#train=sales.iloc[:1461]
#test=sales.iloc[1461:]
print(train.shape,test.shape)
```


```python
model=ARIMA(train['sales'],order=(5,1,2))
model=model.fit()
model.summary()
```


```python
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,typ='levels')
print(pred)
pred.index=sales.index[start:end+1]
print(pred)
```


```python
pred.plot(legend=True)
test['sales'].plot(legend=True)
```


```python
test['sales'].mean()
```


```python
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(pred, test['sales']))
print(rmse)
```


```python
model2=ARIMA(sales['sales'], order=(5,1,2))
model2=model2.fit()
sales.tail()
```


```python
index_future_dates=pd.date_range(start='2017-12-31', end='2018-01-31')
print(index_future_dates)
pred=model2.predict(start=len(sales),end=len(sales)+31,typ='levels').rename('ARIMA Predictions')
#print(comp_pred)
pred.index=index_future_dates
print(pred.index)
print(pred[0:10])
```

## Anvil Uplink


```python
a=input('yyyy-mm-dd: ')
b=input('yyyy-mm-dd: ')

#2018-12-31
#2019-01-31

```

    yyyy-mm-dd: 2018-12-31
    yyyy-mm-dd: 2019-01-31



```python
user_date_range = pd.date_range(start=a,end=b)
pred.index = user_date_range
print(pred.index)
print(pred[0:10])
```

    DatetimeIndex(['2018-12-31', '2019-01-01', '2019-01-02', '2019-01-03',
                   '2019-01-04', '2019-01-05', '2019-01-06', '2019-01-07',
                   '2019-01-08', '2019-01-09', '2019-01-10', '2019-01-11',
                   '2019-01-12', '2019-01-13', '2019-01-14', '2019-01-15',
                   '2019-01-16', '2019-01-17', '2019-01-18', '2019-01-19',
                   '2019-01-20', '2019-01-21', '2019-01-22', '2019-01-23',
                   '2019-01-24', '2019-01-25', '2019-01-26', '2019-01-27',
                   '2019-01-28', '2019-01-29', '2019-01-30', '2019-01-31'],
                  dtype='datetime64[ns]', freq='D')
    2018-12-31    20.009661
    2019-01-01    18.943766
    2019-01-02    17.814788
    2019-01-03    18.797962
    2019-01-04    20.404586
    2019-01-05    21.718020
    2019-01-06    21.660654
    2019-01-07    21.074279
    2019-01-08    20.351041
    2019-01-09    19.964455
    Freq: D, Name: ARIMA Predictions, dtype: float64



```python
model.predict(start=a,end=b)
```


```python
import anvil.server

anvil.server.connect("KWZLDYZCM63NAOSZZLL3NEQC-K4TPWXJQR3RK3YTD")
```


```python
import anvil
@anvil.server.callable
def sale_prediction(a,b):
    
    d_range = pd.date_range(start = a, end = b)
    pred.index = d_range
    return pred[0:10]
```


```python
import anvil
@anvil.server.callable
def sale_prediction(d_range):
    
    
```
