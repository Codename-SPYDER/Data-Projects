```python
#Importing the raw data from csv

import pandas as pd
import numpy as np
data = pd.read_csv('Covid_data.csv')
df = pd.DataFrame (data)
df.columns = ['Day-Month', 'Demand 2019', 'HOEP 2019', 'Date 2020', 'Demand 2020', 'HOEP 2020', 'COVID Cases', '2019-2020 DD', '2019-2020 PD']

data.head()
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
      <th>Day-Month</th>
      <th>Demand 2019</th>
      <th>HOEP 2019</th>
      <th>Date 2020</th>
      <th>Demand 2020</th>
      <th>HOEP 2020</th>
      <th>COVID Cases</th>
      <th>2019-2020 DD</th>
      <th>2019-2020 PD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>01-Jan</td>
      <td>407273</td>
      <td>$ 6.99</td>
      <td>2020-01-01</td>
      <td>338635.0</td>
      <td>$ 0.55</td>
      <td>NaN</td>
      <td>-16.9%</td>
      <td>-92.12%</td>
    </tr>
    <tr>
      <td>1</td>
      <td>02-Jan</td>
      <td>450409</td>
      <td>$ 27.66</td>
      <td>2020-01-02</td>
      <td>362851.0</td>
      <td>$ 0.01</td>
      <td>NaN</td>
      <td>-19.4%</td>
      <td>-99.96%</td>
    </tr>
    <tr>
      <td>2</td>
      <td>03-Jan</td>
      <td>454382</td>
      <td>$ 13.56</td>
      <td>2020-01-03</td>
      <td>369581.0</td>
      <td>$ 16.59</td>
      <td>NaN</td>
      <td>-18.7%</td>
      <td>22.31%</td>
    </tr>
    <tr>
      <td>3</td>
      <td>04-Jan</td>
      <td>435563</td>
      <td>$ 9.50</td>
      <td>2020-01-04</td>
      <td>361837.0</td>
      <td>$ 15.05</td>
      <td>NaN</td>
      <td>-16.9%</td>
      <td>58.44%</td>
    </tr>
    <tr>
      <td>4</td>
      <td>05-Jan</td>
      <td>417771</td>
      <td>$ 13.41</td>
      <td>2020-01-05</td>
      <td>359681.0</td>
      <td>$ 5.69</td>
      <td>NaN</td>
      <td>-13.9%</td>
      <td>-57.60%</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Covid19 data limitation, some days there are no reported COVID19 cases, code is used to fill NAN with zero 

df['COVID Cases'] = df['COVID Cases'].fillna(0)
df.head()

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
      <th>Day-Month</th>
      <th>Demand 2019</th>
      <th>HOEP 2019</th>
      <th>Date 2020</th>
      <th>Demand 2020</th>
      <th>HOEP 2020</th>
      <th>COVID Cases</th>
      <th>2019-2020 DD</th>
      <th>2019-2020 PD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>01-Jan</td>
      <td>407273</td>
      <td>$ 6.99</td>
      <td>2020-01-01</td>
      <td>338635.0</td>
      <td>$ 0.55</td>
      <td>0.0</td>
      <td>-16.9%</td>
      <td>-92.12%</td>
    </tr>
    <tr>
      <td>1</td>
      <td>02-Jan</td>
      <td>450409</td>
      <td>$ 27.66</td>
      <td>2020-01-02</td>
      <td>362851.0</td>
      <td>$ 0.01</td>
      <td>0.0</td>
      <td>-19.4%</td>
      <td>-99.96%</td>
    </tr>
    <tr>
      <td>2</td>
      <td>03-Jan</td>
      <td>454382</td>
      <td>$ 13.56</td>
      <td>2020-01-03</td>
      <td>369581.0</td>
      <td>$ 16.59</td>
      <td>0.0</td>
      <td>-18.7%</td>
      <td>22.31%</td>
    </tr>
    <tr>
      <td>3</td>
      <td>04-Jan</td>
      <td>435563</td>
      <td>$ 9.50</td>
      <td>2020-01-04</td>
      <td>361837.0</td>
      <td>$ 15.05</td>
      <td>0.0</td>
      <td>-16.9%</td>
      <td>58.44%</td>
    </tr>
    <tr>
      <td>4</td>
      <td>05-Jan</td>
      <td>417771</td>
      <td>$ 13.41</td>
      <td>2020-01-05</td>
      <td>359681.0</td>
      <td>$ 5.69</td>
      <td>0.0</td>
      <td>-13.9%</td>
      <td>-57.60%</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['2019-2020 DD', '2019-2020 PD']] = df[['2019-2020 DD', '2019-2020 PD'
                    ]].replace('%','', regex=True).astype(float)
df.head()
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
      <th>Day-Month</th>
      <th>Demand 2019</th>
      <th>HOEP 2019</th>
      <th>Date 2020</th>
      <th>Demand 2020</th>
      <th>HOEP 2020</th>
      <th>COVID Cases</th>
      <th>2019-2020 DD</th>
      <th>2019-2020 PD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>01-Jan</td>
      <td>407273</td>
      <td>6.99</td>
      <td>2020-01-01</td>
      <td>338635.0</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>-16.9</td>
      <td>-92.12</td>
    </tr>
    <tr>
      <td>1</td>
      <td>02-Jan</td>
      <td>450409</td>
      <td>27.66</td>
      <td>2020-01-02</td>
      <td>362851.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>-19.4</td>
      <td>-99.96</td>
    </tr>
    <tr>
      <td>2</td>
      <td>03-Jan</td>
      <td>454382</td>
      <td>13.56</td>
      <td>2020-01-03</td>
      <td>369581.0</td>
      <td>16.59</td>
      <td>0.0</td>
      <td>-18.7</td>
      <td>22.31</td>
    </tr>
    <tr>
      <td>3</td>
      <td>04-Jan</td>
      <td>435563</td>
      <td>9.50</td>
      <td>2020-01-04</td>
      <td>361837.0</td>
      <td>15.05</td>
      <td>0.0</td>
      <td>-16.9</td>
      <td>58.44</td>
    </tr>
    <tr>
      <td>4</td>
      <td>05-Jan</td>
      <td>417771</td>
      <td>13.41</td>
      <td>2020-01-05</td>
      <td>359681.0</td>
      <td>5.69</td>
      <td>0.0</td>
      <td>-13.9</td>
      <td>-57.60</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes

```




    Day-Month        object
    Demand 2019       int64
    HOEP 2019        object
    Date 2020        object
    Demand 2020     float64
    HOEP 2020        object
    COVID Cases     float64
    2019-2020 DD    float64
    2019-2020 PD    float64
    dtype: object




```python
df[['HOEP 2019', 'HOEP 2020']] = df[['HOEP 2019', 'HOEP 2020'
                      ]].replace('\$','', regex = True)
```


```python
df[['HOEP 2019', 'HOEP 2020']] = df[['HOEP 2019', 'HOEP 2020'
                      ]].replace(' ','', regex = True)
```


```python
df[['HOEP 2019', 'HOEP 2020']] = df[['HOEP 2019', 'HOEP 2020'
                      ]].replace(' ','', regex = True).astype(float)
```


```python
df.dtypes
df1 = df[['HOEP 2019', 'HOEP 2020']]
df2 = df[['Demand 2019', 'Demand 2020']]
```


```python
df1.plot()
df2.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1b0a9590>




    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



```python
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates

sns.regplot(x="COVID Cases", y="Demand 2020", data=data);
```


    
![png](output_9_0.png)
    



```python
sns.residplot(x="COVID Cases", y="HOEP 2020", data=data.query("dataset == 'I'"),
              scatter_kws={"s": 80});
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-59-4c819b2eb985> in <module>
    ----> 1 sns.residplot(x="COVID Cases", y="HOEP 2020", data=data.query("dataset == 'I'"),
          2               scatter_kws={"s": 80});


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in query(self, expr, inplace, **kwargs)
       3182         kwargs["level"] = kwargs.pop("level", 0) + 1
       3183         kwargs["target"] = None
    -> 3184         res = self.eval(expr, **kwargs)
       3185 
       3186         try:


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in eval(self, expr, inplace, **kwargs)
       3293         if resolvers is None:
       3294             index_resolvers = self._get_index_resolvers()
    -> 3295             column_resolvers = self._get_space_character_free_column_resolvers()
       3296             resolvers = column_resolvers, index_resolvers
       3297         if "target" not in kwargs:


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py in _get_space_character_free_column_resolvers(self)
        482         from pandas.core.computation.common import _remove_spaces_column_name
        483 
    --> 484         return {_remove_spaces_column_name(k): v for k, v in self.items()}
        485 
        486     @property


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py in <dictcomp>(.0)
        482         from pandas.core.computation.common import _remove_spaces_column_name
        483 
    --> 484         return {_remove_spaces_column_name(k): v for k, v in self.items()}
        485 
        486     @property


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in items(self)
        834         if self.columns.is_unique and hasattr(self, "_item_cache"):
        835             for k in self.columns:
    --> 836                 yield k, self._get_item_cache(k)
        837         else:
        838             for i, k in enumerate(self.columns):


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/generic.py in _get_item_cache(self, item)
       3268         res = cache.get(item)
       3269         if res is None:
    -> 3270             values = self._data.get(item)
       3271             res = self._box_item_values(item, values)
       3272             cache[item] = res


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/internals/managers.py in get(self, item)
        958                         raise ValueError("cannot label index with a null key")
        959 
    --> 960             return self.iget(loc)
        961         else:
        962 


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/internals/managers.py in iget(self, i)
        976         """
        977         block = self.blocks[self._blknos[i]]
    --> 978         values = block.iget(self._blklocs[i])
        979         if values.ndim != 1:
        980             return values


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/internals/blocks.py in iget(self, i)
        368 
        369     def iget(self, i):
    --> 370         return self.values[i]
        371 
        372     def set(self, locs, values):


    IndexError: index 2 is out of bounds for axis 0 with size 2



```python
#COVID19 cases Year to date linestyle using Seaborn

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates

plt.style.use('seaborn')
    
df['COVID19 New cases'] = df['COVID19 New cases'].fillna(0)
    
data['Date 2020'] = pd.to_datetime(data['Date 2020'])
data.sort_values('Date 2020', inplace=True)
Date = data['Date 2020']
Covid19 = data['COVID19 New cases']
plt.plot_date(Date, Covid19, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)

plt.title('COVID19 Cases Year to date')
plt.xlabel('Year to date 2020')
plt.ylabel('Cases')

plt.gca().axes.get_xaxis().set_visible(False)
plt.tight_layout()

plt.show()
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2896             try:
    -> 2897                 return self._engine.get_loc(key)
       2898             except KeyError:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'COVID19 New cases'

    
    During handling of the above exception, another exception occurred:


    KeyError                                  Traceback (most recent call last)

    <ipython-input-60-97ef4eb3ff89> in <module>
         10 plt.style.use('seaborn')
         11 
    ---> 12 df['COVID19 New cases'] = df['COVID19 New cases'].fillna(0)
         13 
         14 data['Date 2020'] = pd.to_datetime(data['Date 2020'])


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py in __getitem__(self, key)
       2978             if self.columns.nlevels > 1:
       2979                 return self._getitem_multilevel(key)
    -> 2980             indexer = self.columns.get_loc(key)
       2981             if is_integer(indexer):
       2982                 indexer = [indexer]


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexes/base.py in get_loc(self, key, method, tolerance)
       2897                 return self._engine.get_loc(key)
       2898             except KeyError:
    -> 2899                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2900         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2901         if indexer.ndim > 1 or indexer.size > 1:


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/index.pyx in pandas._libs.index.IndexEngine.get_loc()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()


    KeyError: 'COVID19 New cases'



```python
#COVID19 cases Year to date Barchart using Seaborn

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates
    
plt.style.use('seaborn')
  
data = pd.read_csv(r'C:\\Users\\loret\\Documents\\York University\\Group Project\\Consolidated Data.csv')
df = pd.DataFrame (data, columns= ['Date 2020','Ontario Demand 2020','HOEP 2020','COVID19 New cases'])
    
df['COVID19 New cases'] = df['COVID19 New cases'].fillna(0)
    
data['Date 2020'] = pd.to_datetime(data['Date 2020'])
data.sort_values('Date 2020', inplace=True)
Date = data['Date 2020']
Covid19 = data['COVID19 New cases']
plt.bar(Date, Covid19, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
    
plt.gca().axes.get_xaxis().set_visible(False)
 
plt.title('COVID19 Cases Year to date')
plt.xlabel('Year to date 2020')
plt.ylabel('Cases')

plt.tight_layout()
   
plt.show()
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-61-167d8b3a9eb2> in <module>
         10 plt.style.use('seaborn')
         11 
    ---> 12 data = pd.read_csv(r'C:\\Users\\loret\\Documents\\York University\\Group Project\\Consolidated Data.csv')
         13 df = pd.DataFrame (data, columns= ['Date 2020','Ontario Demand 2020','HOEP 2020','COVID19 New cases'])
         14 


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/io/parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)
        683         )
        684 
    --> 685         return _read(filepath_or_buffer, kwds)
        686 
        687     parser_f.__name__ = name


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/io/parsers.py in _read(filepath_or_buffer, kwds)
        455 
        456     # Create the parser.
    --> 457     parser = TextFileReader(fp_or_buf, **kwds)
        458 
        459     if chunksize or iterator:


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/io/parsers.py in __init__(self, f, engine, **kwds)
        893             self.options["has_index_names"] = kwds["has_index_names"]
        894 
    --> 895         self._make_engine(self.engine)
        896 
        897     def close(self):


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/io/parsers.py in _make_engine(self, engine)
       1133     def _make_engine(self, engine="c"):
       1134         if engine == "c":
    -> 1135             self._engine = CParserWrapper(self.f, **self.options)
       1136         else:
       1137             if engine == "python":


    ~/opt/anaconda3/lib/python3.7/site-packages/pandas/io/parsers.py in __init__(self, src, **kwds)
       1915         kwds["usecols"] = self.usecols
       1916 
    -> 1917         self._reader = parsers.TextReader(src, **kwds)
       1918         self.unnamed_cols = self._reader.unnamed_cols
       1919 


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.__cinit__()


    pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._setup_parser_source()


    FileNotFoundError: [Errno 2] File b'C:\\\\Users\\\\loret\\\\Documents\\\\York University\\\\Group Project\\\\Consolidated Data.csv' does not exist: b'C:\\\\Users\\\\loret\\\\Documents\\\\York University\\\\Group Project\\\\Consolidated Data.csv'



```python
#Ontario Power Demand Year To date linestyle using Seaborn

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates
import pandas as pd
plt.style.use('seaborn')
 
data = pd.read_csv(r'C:\\Users\\loret\\Documents\\York University\\Group Project\\Consolidated Data.csv')
df = pd.DataFrame (data, columns= ['Date 2020','Ontario Demand 2020','HOEP 2020','COVID19 New cases'])
  
data['Date 2020'] = pd.to_datetime(data['Date 2020'])
data.sort_values('Date 2020', inplace=True)
Date = data['Date 2020']
Demand = data['Ontario Demand 2020']
plt.plot_date(Date, Demand, linestyle='solid')
plt.gcf().autofmt_xdate()
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
plt.gca().xaxis.set_major_formatter(date_format)
plt.tight_layout()

plt.gca().axes.get_xaxis().set_visible(False)

plt.title('Ontario Power Demand 2020')
plt.xlabel('Year to date 2020')
plt.ylabel('Demand')
 
    
plt.show()
```


    
![png](output_13_0.png)
    



```python
# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame (data, columns= ['Date 2020','Ontario Demand 2020','HOEP 2020','COVID19 New cases'])
df['COVID19 New cases'] = df['COVID19 New cases'].fillna(0)
 
df = sns.scatterplot(x = df["Date 2020"].values, y = df["Ontario Demand 2020"].values)



plt.xlabel("Year to Date 2020", size=11)
plt.ylabel("Ontario Demand 2020", size=11)
plt.title("Ontario Demand Year to Date 2020", size=12)
```




    Text(0.5, 1.0, 'Ontario Demand Year to Date')




    
![png](output_14_1.png)
    



```python
# library & dataset
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame (data, columns= ['Date 2020','Ontario Demand 2020','HOEP 2020','COVID19 New cases'])
df['COVID19 New cases'] = df['COVID19 New cases'].fillna(0)
 
df = sns.scatterplot(x = df["Date 2020"].values, y = df["HOEP 2020"].values)



plt.xlabel("Year to Date 2020", size=11)
plt.ylabel("HOEP 2020", size=11)
plt.title("HOEP", size=12)
```




    Text(0.5, 1.0, 'HOEP')




    
![png](output_15_1.png)
    



```python

```
