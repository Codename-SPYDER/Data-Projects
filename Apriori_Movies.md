```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
%matplotlib inline
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```


```python

import requests, zipfile, io
r = requests.get('http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

#For python 3+, sub the StringIO module with the io module and use BytesIO instead of StringIO: 
#Here are release notes that mention this change.
    #import requests, zipfile, StringIO
    #r = requests.get(zip_file_url)
    #z = zipfile.ZipFile(io.BytesIO(r.content))
    #z.extractall("/path/to/destination_directory")
```


```python
ls -a ml-latest-small
```

    [34m.[m[m/           README.txt   movies.csv   tags.csv
    [34m..[m[m/          links.csv    ratings.csv



```python
movies = pd.read_csv('ml-latest-small/movies.csv')
tags = pd.read_csv('ml-latest-small/tags.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
```


```python
movies.head()
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
movies.movieId.value_counts()
```




    86014     1
    1282      1
    3347      1
    1298      1
    25870     1
             ..
    60072     1
    4775      1
    50601     1
    131749    1
    83969     1
    Name: movieId, Length: 9742, dtype: int64




```python
movies.isnull().sum()
```




    movieId    0
    title      0
    genres     0
    dtype: int64




```python
tags
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
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>60756</td>
      <td>funny</td>
      <td>1445714994</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>60756</td>
      <td>Highly quotable</td>
      <td>1445714996</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>60756</td>
      <td>will ferrell</td>
      <td>1445714992</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>89774</td>
      <td>Boxing story</td>
      <td>1445715207</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>89774</td>
      <td>MMA</td>
      <td>1445715200</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3678</td>
      <td>606</td>
      <td>7382</td>
      <td>for katie</td>
      <td>1171234019</td>
    </tr>
    <tr>
      <td>3679</td>
      <td>606</td>
      <td>7936</td>
      <td>austere</td>
      <td>1173392334</td>
    </tr>
    <tr>
      <td>3680</td>
      <td>610</td>
      <td>3265</td>
      <td>gun fu</td>
      <td>1493843984</td>
    </tr>
    <tr>
      <td>3681</td>
      <td>610</td>
      <td>3265</td>
      <td>heroic bloodshed</td>
      <td>1493843978</td>
    </tr>
    <tr>
      <td>3682</td>
      <td>610</td>
      <td>168248</td>
      <td>Heroic Bloodshed</td>
      <td>1493844270</td>
    </tr>
  </tbody>
</table>
<p>3683 rows × 4 columns</p>
</div>




```python
tags.isnull().sum()
```




    userId       0
    movieId      0
    tag          0
    timestamp    0
    dtype: int64




```python
ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.movieId.value_counts()
```




    356       329
    318       317
    296       307
    593       279
    2571      278
             ... 
    5986        1
    100304      1
    34800       1
    83976       1
    8196        1
    Name: movieId, Length: 9724, dtype: int64




```python
ratings.isnull().sum()
```




    userId       0
    movieId      0
    rating       0
    timestamp    0
    dtype: int64




```python
ratings.loc[(ratings['userId'] == 336) & (ratings['movieId'] == 1)]
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>51837</td>
      <td>336</td>
      <td>1</td>
      <td>4.0</td>
      <td>1122227329</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.loc[(ratings['userId'] ==474) & (ratings['movieId'] == 1)]
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>73092</td>
      <td>474</td>
      <td>1</td>
      <td>4.0</td>
      <td>978575760</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.loc[ratings.userId.isin([336, 474]) & (ratings['movieId'] == 1)]
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>51837</td>
      <td>336</td>
      <td>1</td>
      <td>4.0</td>
      <td>1122227329</td>
    </tr>
    <tr>
      <td>73092</td>
      <td>474</td>
      <td>1</td>
      <td>4.0</td>
      <td>978575760</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags.sort_values(by='movieId', ascending=True)

# Tags correspong to movie dataset
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
      <th>userId</th>
      <th>movieId</th>
      <th>tag</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2886</td>
      <td>567</td>
      <td>1</td>
      <td>fun</td>
      <td>1525286013</td>
    </tr>
    <tr>
      <td>981</td>
      <td>474</td>
      <td>1</td>
      <td>pixar</td>
      <td>1137206825</td>
    </tr>
    <tr>
      <td>629</td>
      <td>336</td>
      <td>1</td>
      <td>pixar</td>
      <td>1139045764</td>
    </tr>
    <tr>
      <td>35</td>
      <td>62</td>
      <td>2</td>
      <td>Robin Williams</td>
      <td>1528843907</td>
    </tr>
    <tr>
      <td>34</td>
      <td>62</td>
      <td>2</td>
      <td>magic board game</td>
      <td>1528843932</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>402</td>
      <td>62</td>
      <td>187595</td>
      <td>star wars</td>
      <td>1528934552</td>
    </tr>
    <tr>
      <td>528</td>
      <td>184</td>
      <td>193565</td>
      <td>comedy</td>
      <td>1537098587</td>
    </tr>
    <tr>
      <td>527</td>
      <td>184</td>
      <td>193565</td>
      <td>anime</td>
      <td>1537098582</td>
    </tr>
    <tr>
      <td>530</td>
      <td>184</td>
      <td>193565</td>
      <td>remaster</td>
      <td>1537098592</td>
    </tr>
    <tr>
      <td>529</td>
      <td>184</td>
      <td>193565</td>
      <td>gintama</td>
      <td>1537098603</td>
    </tr>
  </tbody>
</table>
<p>3683 rows × 4 columns</p>
</div>




```python
master = movies.merge(tags,on ='movieId',how = 'inner')
master2 = master.merge(ratings,on = ['movieId', 'userId'],how = 'inner')
#on: 
#Column or index level names to join on. Must be found in both the left and right DataFrame and/or Series objects.

#how:
#inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.

```


```python
master2
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>timestamp_x</th>
      <th>rating</th>
      <th>timestamp_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>1139045764</td>
      <td>4.0</td>
      <td>1122227329</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>474</td>
      <td>pixar</td>
      <td>1137206825</td>
      <td>4.0</td>
      <td>978575760</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>567</td>
      <td>fun</td>
      <td>1525286013</td>
      <td>3.5</td>
      <td>1525286001</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>1528843929</td>
      <td>4.0</td>
      <td>1528843890</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>magic board game</td>
      <td>1528843932</td>
      <td>4.0</td>
      <td>1528843890</td>
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
    </tr>
    <tr>
      <td>3471</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>star wars</td>
      <td>1528934552</td>
      <td>4.0</td>
      <td>1528934550</td>
    </tr>
    <tr>
      <td>3472</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>anime</td>
      <td>1537098582</td>
      <td>3.5</td>
      <td>1537098554</td>
    </tr>
    <tr>
      <td>3473</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>1537098587</td>
      <td>3.5</td>
      <td>1537098554</td>
    </tr>
    <tr>
      <td>3474</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>gintama</td>
      <td>1537098603</td>
      <td>3.5</td>
      <td>1537098554</td>
    </tr>
    <tr>
      <td>3475</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>remaster</td>
      <td>1537098592</td>
      <td>3.5</td>
      <td>1537098554</td>
    </tr>
  </tbody>
</table>
<p>3476 rows × 8 columns</p>
</div>




```python
master2.drop(columns=['timestamp_x','timestamp_y'],inplace=True)
```


```python
master2
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>474</td>
      <td>pixar</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>567</td>
      <td>fun</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>magic board game</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3471</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>star wars</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3472</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>anime</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3473</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3474</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>gintama</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3475</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>remaster</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>3476 rows × 6 columns</p>
</div>




```python
len(master2.userId.unique()) 
```




    54




```python
len(master2.movieId.unique()) 
```




    1464




```python
master2.sort_values(by='movieId', ascending=True, inplace = True)
```


```python
master2
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>474</td>
      <td>pixar</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>567</td>
      <td>fun</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>magic board game</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3471</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>star wars</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3473</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3474</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>gintama</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3472</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>anime</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3475</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>remaster</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>3476 rows × 6 columns</p>
</div>




```python
master2.groupby('movieId')['rating'].mean().round(1)
```




    movieId
    1         3.8
    2         3.8
    3         2.5
    5         1.5
    7         3.0
             ... 
    183611    4.0
    184471    3.5
    187593    4.0
    187595    4.0
    193565    3.5
    Name: rating, Length: 1464, dtype: float64




```python
master3 = master2.drop_duplicates(subset=['movieId'])
```


```python
master3
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>289</td>
      <td>moldy</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>474</td>
      <td>pregnancy</td>
      <td>1.5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
      <td>474</td>
      <td>remake</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>3461</td>
      <td>183611</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>62</td>
      <td>Comedy</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3464</td>
      <td>184471</td>
      <td>Tomb Raider (2018)</td>
      <td>Action|Adventure|Fantasy</td>
      <td>62</td>
      <td>adventure</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3467</td>
      <td>187593</td>
      <td>Deadpool 2 (2018)</td>
      <td>Action|Comedy|Sci-Fi</td>
      <td>62</td>
      <td>Josh Brolin</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3470</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>Emilia Clarke</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3473</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>1464 rows × 6 columns</p>
</div>




```python
master3['ratings'] = master2.groupby('movieId')['rating'].transform(np.mean).round(1)
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>rating</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>4.0</td>
      <td>3.8</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>4.0</td>
      <td>3.8</td>
    </tr>
    <tr>
      <td>7</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>289</td>
      <td>moldy</td>
      <td>2.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>9</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>474</td>
      <td>pregnancy</td>
      <td>1.5</td>
      <td>1.5</td>
    </tr>
    <tr>
      <td>11</td>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
      <td>474</td>
      <td>remake</td>
      <td>3.0</td>
      <td>3.0</td>
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
    </tr>
    <tr>
      <td>3461</td>
      <td>183611</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>62</td>
      <td>Comedy</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3464</td>
      <td>184471</td>
      <td>Tomb Raider (2018)</td>
      <td>Action|Adventure|Fantasy</td>
      <td>62</td>
      <td>adventure</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>3467</td>
      <td>187593</td>
      <td>Deadpool 2 (2018)</td>
      <td>Action|Comedy|Sci-Fi</td>
      <td>62</td>
      <td>Josh Brolin</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3470</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>Emilia Clarke</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3473</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>1464 rows × 7 columns</p>
</div>




```python
master3.reset_index(drop=True, inplace = True)
```


```python
master3.columns
```




    Index(['movieId', 'title', 'genres', 'userId', 'tag', 'rating', 'ratings'], dtype='object')




```python
master3
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>rating</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>4.0</td>
      <td>3.8</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>4.0</td>
      <td>3.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>289</td>
      <td>moldy</td>
      <td>2.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>474</td>
      <td>pregnancy</td>
      <td>1.5</td>
      <td>1.5</td>
    </tr>
    <tr>
      <td>4</td>
      <td>7</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
      <td>474</td>
      <td>remake</td>
      <td>3.0</td>
      <td>3.0</td>
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
    </tr>
    <tr>
      <td>1459</td>
      <td>183611</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>62</td>
      <td>Comedy</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1460</td>
      <td>184471</td>
      <td>Tomb Raider (2018)</td>
      <td>Action|Adventure|Fantasy</td>
      <td>62</td>
      <td>adventure</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>1461</td>
      <td>187593</td>
      <td>Deadpool 2 (2018)</td>
      <td>Action|Comedy|Sci-Fi</td>
      <td>62</td>
      <td>Josh Brolin</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1462</td>
      <td>187595</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>Emilia Clarke</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>1463</td>
      <td>193565</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
      <td>3.5</td>
    </tr>
  </tbody>
</table>
<p>1464 rows × 7 columns</p>
</div>




```python
master3['MovieId'] = master3.index
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3.columns
```




    Index(['movieId', 'title', 'genres', 'userId', 'tag', 'rating', 'ratings', 'MovieId'], dtype='object')




```python
master3.drop(columns=['movieId','rating'],inplace=True)
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/pandas/core/frame.py:4102: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      errors=errors,



```python
master3
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
      <th>title</th>
      <th>genres</th>
      <th>userId</th>
      <th>tag</th>
      <th>ratings</th>
      <th>MovieId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>336</td>
      <td>pixar</td>
      <td>3.8</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>62</td>
      <td>fantasy</td>
      <td>3.8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>289</td>
      <td>moldy</td>
      <td>2.5</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>474</td>
      <td>pregnancy</td>
      <td>1.5</td>
      <td>3</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Sabrina (1995)</td>
      <td>Comedy|Romance</td>
      <td>474</td>
      <td>remake</td>
      <td>3.0</td>
      <td>4</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1459</td>
      <td>Game Night (2018)</td>
      <td>Action|Comedy|Crime|Horror</td>
      <td>62</td>
      <td>Comedy</td>
      <td>4.0</td>
      <td>1459</td>
    </tr>
    <tr>
      <td>1460</td>
      <td>Tomb Raider (2018)</td>
      <td>Action|Adventure|Fantasy</td>
      <td>62</td>
      <td>adventure</td>
      <td>3.5</td>
      <td>1460</td>
    </tr>
    <tr>
      <td>1461</td>
      <td>Deadpool 2 (2018)</td>
      <td>Action|Comedy|Sci-Fi</td>
      <td>62</td>
      <td>Josh Brolin</td>
      <td>4.0</td>
      <td>1461</td>
    </tr>
    <tr>
      <td>1462</td>
      <td>Solo: A Star Wars Story (2018)</td>
      <td>Action|Adventure|Children|Sci-Fi</td>
      <td>62</td>
      <td>Emilia Clarke</td>
      <td>4.0</td>
      <td>1462</td>
    </tr>
    <tr>
      <td>1463</td>
      <td>Gintama: The Movie (2010)</td>
      <td>Action|Animation|Comedy|Sci-Fi</td>
      <td>184</td>
      <td>comedy</td>
      <td>3.5</td>
      <td>1463</td>
    </tr>
  </tbody>
</table>
<p>1464 rows × 6 columns</p>
</div>




```python
master3["ratings"] = master3["ratings"].astype(str)
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3["important_features"] = master3["genres"] + ' '+ master3["tag"] + ' ' + master3["ratings"]
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3['important_features']
```




    0       Adventure|Animation|Children|Comedy|Fantasy pi...
    1                  Adventure|Children|Fantasy fantasy 3.8
    2                                Comedy|Romance moldy 2.5
    3                                    Comedy pregnancy 1.5
    4                               Comedy|Romance remake 3.0
                                  ...                        
    1459                Action|Comedy|Crime|Horror Comedy 4.0
    1460               Action|Adventure|Fantasy adventure 3.5
    1461                 Action|Comedy|Sci-Fi Josh Brolin 4.0
    1462    Action|Adventure|Children|Sci-Fi Emilia Clarke...
    1463            Action|Animation|Comedy|Sci-Fi comedy 3.5
    Name: important_features, Length: 1464, dtype: object




```python
master3['important_features'] = master3['important_features'].str.replace('|',',')
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3['important_features'] = master3['important_features'].str.replace(' ',',')
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3["important_features"] = master3['important_features'] + ','+ master3["title"]
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
master3['important_features']
```




    0       Adventure,Animation,Children,Comedy,Fantasy,pi...
    1       Adventure,Children,Fantasy,fantasy,3.8,Jumanji...
    2        Comedy,Romance,moldy,2.5,Grumpier Old Men (1995)
    3       Comedy,pregnancy,1.5,Father of the Bride Part ...
    4                Comedy,Romance,remake,3.0,Sabrina (1995)
                                  ...                        
    1459    Action,Comedy,Crime,Horror,Comedy,4.0,Game Nig...
    1460    Action,Adventure,Fantasy,adventure,3.5,Tomb Ra...
    1461    Action,Comedy,Sci-Fi,Josh,Brolin,4.0,Deadpool ...
    1462    Action,Adventure,Children,Sci-Fi,Emilia,Clarke...
    1463    Action,Animation,Comedy,Sci-Fi,comedy,3.5,Gint...
    Name: important_features, Length: 1464, dtype: object




```python
master3.important_features.iloc[8]
```




    'Drama,Romance,Jane,Austen,5.0,Sense and Sensibility (1995)'




```python
master3["important_features"] = master3["important_features"].astype(str)
```

    /Users/siddiqkhan/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
cm = CountVectorizer().fit_transform(master3['important_features'])
```


```python
cs = cosine_similarity(cm)

print(cs)
```

    [[1.         0.58925565 0.25197632 ... 0.11785113 0.28867513 0.28867513]
     [0.58925565 1.         0.13363062 ... 0.         0.20412415 0.        ]
     [0.25197632 0.13363062 1.         ... 0.13363062 0.         0.21821789]
     ...
     [0.11785113 0.         0.13363062 ... 1.         0.40824829 0.51031036]
     [0.28867513 0.20412415 0.         ... 0.40824829 1.         0.25      ]
     [0.28867513 0.         0.21821789 ... 0.51031036 0.25       1.        ]]



```python
cs.shape
```




    (1464, 1464)




```python
# Title of the movie the user likes
Title = 'Jumanji (1995)'

# Find movies id
movie_id = master3[master3.title == Title]['MovieId'].values[0]
```


```python
movie_id
```




    1




```python
# Create list of enumerations for similarity score [(movie_id, similarity score)]
# movie_id corresponds to cs row and movie_id value is where enumeration begins
# movie_id is where enumeration starts
scores = list(enumerate(cs[movie_id]))
```


```python
# Sort List based on similarity score
sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)
sorted_scores = sorted_scores[1:]
```


```python
# Print sorted scores
print(sorted_scores)
```

    [(0, 0.5892556509887895), (1342, 0.5345224838248487), (485, 0.4714045207910316), (628, 0.4714045207910316), (1460, 0.4714045207910316), (189, 0.4472135954999579), (236, 0.4472135954999579), (763, 0.4472135954999579), (592, 0.4330127018922194), (617, 0.4330127018922194), (714, 0.4330127018922194), (1023, 0.4330127018922194), (1390, 0.42640143271122083), (14, 0.41666666666666663), (150, 0.408248290463863), (465, 0.408248290463863), (969, 0.408248290463863), (139, 0.40089186286863654), (332, 0.40089186286863654), (585, 0.40089186286863654), (678, 0.40089186286863654), (480, 0.39223227027636803), (1180, 0.39223227027636803), (1200, 0.39223227027636803), (1301, 0.39223227027636803), (1187, 0.3779644730092272), (231, 0.3749999999999999), (233, 0.3749999999999999), (343, 0.3749999999999999), (484, 0.3749999999999999), (1190, 0.3749999999999999), (1186, 0.36514837167011066), (1384, 0.36514837167011066), (214, 0.35355339059327373), (938, 0.35355339059327373), (1303, 0.35355339059327373), (1457, 0.35355339059327373), (608, 0.34299717028501764), (1253, 0.33541019662496846), (1286, 0.33541019662496846), (1299, 0.33541019662496846), (1366, 0.33541019662496846), (950, 0.3333333333333333), (96, 0.3198010745334156), (262, 0.3198010745334156), (276, 0.3198010745334156), (880, 0.3198010745334156), (1201, 0.3198010745334156), (973, 0.3162277660168379), (126, 0.30618621784789724), (138, 0.30618621784789724), (475, 0.30618621784789724), (479, 0.30618621784789724), (607, 0.30618621784789724), (820, 0.30618621784789724), (864, 0.30618621784789724), (1125, 0.30618621784789724), (1375, 0.30618621784789724), (1427, 0.30618621784789724), (1456, 0.30618621784789724), (137, 0.294174202707276), (232, 0.294174202707276), (609, 0.294174202707276), (1209, 0.294174202707276), (1249, 0.294174202707276), (17, 0.2886751345948129), (65, 0.2886751345948129), (330, 0.2886751345948129), (476, 0.2886751345948129), (1315, 0.2886751345948129), (1439, 0.2886751345948129), (888, 0.28347335475692037), (812, 0.2721655269759087), (19, 0.26726124191242434), (41, 0.26726124191242434), (98, 0.26726124191242434), (360, 0.26726124191242434), (504, 0.26726124191242434), (575, 0.26726124191242434), (621, 0.26726124191242434), (1058, 0.26726124191242434), (1252, 0.26726124191242434), (1326, 0.26726124191242434), (1182, 0.2651650429449553), (723, 0.25724787771376323), (886, 0.25), (1408, 0.25), (42, 0.24999999999999994), (77, 0.24999999999999994), (131, 0.24999999999999994), (132, 0.24999999999999994), (225, 0.24999999999999994), (235, 0.24999999999999994), (350, 0.24999999999999994), (517, 0.24999999999999994), (573, 0.24999999999999994), (645, 0.24999999999999994), (736, 0.24999999999999994), (756, 0.24999999999999994), (809, 0.24999999999999994), (908, 0.24999999999999994), (920, 0.24999999999999994), (953, 0.24999999999999994), (954, 0.24999999999999994), (34, 0.2357022603955158), (43, 0.2357022603955158), (156, 0.2357022603955158), (620, 0.2357022603955158), (626, 0.2357022603955158), (779, 0.2357022603955158), (986, 0.2357022603955158), (1123, 0.2357022603955158), (1172, 0.2357022603955158), (1235, 0.2357022603955158), (1263, 0.2357022603955158), (1268, 0.2357022603955158), (1313, 0.2357022603955158), (1360, 0.2357022603955158), (1423, 0.2357022603955158), (206, 0.22360679774997896), (486, 0.22360679774997896), (1081, 0.22360679774997896), (1165, 0.22360679774997896), (1267, 0.22360679774997896), (1278, 0.22360679774997896), (1302, 0.22360679774997896), (1305, 0.22360679774997896), (1401, 0.22360679774997896), (970, 0.2165063509461097), (91, 0.21320071635561041), (1266, 0.21320071635561041), (1310, 0.21320071635561041), (530, 0.2041241452319315), (971, 0.2041241452319315), (1269, 0.2041241452319315), (1340, 0.2041241452319315), (1369, 0.2041241452319315), (1462, 0.2041241452319315), (821, 0.20044593143431827), (548, 0.19611613513818402), (581, 0.19611613513818402), (788, 0.19611613513818402), (1059, 0.19364916731037082), (813, 0.1889822365046136), (295, 0.18257418583505533), (1000, 0.18257418583505533), (6, 0.17677669529663687), (12, 0.17677669529663687), (22, 0.17677669529663687), (474, 0.17677669529663687), (696, 0.17677669529663687), (4, 0.15811388300841894), (7, 0.15811388300841894), (20, 0.15811388300841894), (50, 0.15811388300841894), (613, 0.15811388300841894), (1407, 0.15811388300841894), (15, 0.14433756729740646), (21, 0.14433756729740646), (29, 0.14433756729740646), (37, 0.14433756729740646), (46, 0.14433756729740646), (49, 0.14433756729740646), (57, 0.14433756729740646), (62, 0.14433756729740646), (130, 0.14433756729740646), (142, 0.14433756729740646), (144, 0.14433756729740646), (537, 0.14433756729740646), (591, 0.14433756729740646), (695, 0.14433756729740646), (804, 0.14433756729740646), (1043, 0.14433756729740646), (1207, 0.14433756729740646), (1228, 0.14433756729740646), (1239, 0.14433756729740646), (1274, 0.14433756729740646), (1429, 0.14433756729740646), (2, 0.13363062095621217), (9, 0.13363062095621217), (11, 0.13363062095621217), (13, 0.13363062095621217), (27, 0.13363062095621217), (35, 0.13363062095621217), (40, 0.13363062095621217), (44, 0.13363062095621217), (47, 0.13363062095621217), (51, 0.13363062095621217), (54, 0.13363062095621217), (154, 0.13363062095621217), (155, 0.13363062095621217), (205, 0.13363062095621217), (229, 0.13363062095621217), (234, 0.13363062095621217), (437, 0.13363062095621217), (560, 0.13363062095621217), (605, 0.13363062095621217), (848, 0.13363062095621217), (862, 0.13363062095621217), (905, 0.13363062095621217), (967, 0.13363062095621217), (1052, 0.13363062095621217), (1112, 0.13363062095621217), (1122, 0.13363062095621217), (1224, 0.13363062095621217), (1361, 0.13363062095621217), (1276, 0.13130643285972254), (1184, 0.1270001270001905), (5, 0.12499999999999997), (8, 0.12499999999999997), (18, 0.12499999999999997), (23, 0.12499999999999997), (25, 0.12499999999999997), (26, 0.12499999999999997), (31, 0.12499999999999997), (68, 0.12499999999999997), (76, 0.12499999999999997), (81, 0.12499999999999997), (82, 0.12499999999999997), (129, 0.12499999999999997), (153, 0.12499999999999997), (230, 0.12499999999999997), (278, 0.12499999999999997), (319, 0.12499999999999997), (431, 0.12499999999999997), (467, 0.12499999999999997), (510, 0.12499999999999997), (538, 0.12499999999999997), (562, 0.12499999999999997), (588, 0.12499999999999997), (590, 0.12499999999999997), (606, 0.12499999999999997), (637, 0.12499999999999997), (646, 0.12499999999999997), (697, 0.12499999999999997), (771, 0.12499999999999997), (865, 0.12499999999999997), (997, 0.12499999999999997), (1047, 0.12499999999999997), (1070, 0.12499999999999997), (1349, 0.12499999999999997), (1365, 0.12499999999999997), (1382, 0.12499999999999997), (1389, 0.12499999999999997), (1415, 0.12499999999999997), (1422, 0.12499999999999997), (1428, 0.12499999999999997), (3, 0.1178511301977579), (10, 0.1178511301977579), (53, 0.1178511301977579), (79, 0.1178511301977579), (94, 0.1178511301977579), (108, 0.1178511301977579), (127, 0.1178511301977579), (134, 0.1178511301977579), (194, 0.1178511301977579), (220, 0.1178511301977579), (254, 0.1178511301977579), (281, 0.1178511301977579), (321, 0.1178511301977579), (328, 0.1178511301977579), (472, 0.1178511301977579), (483, 0.1178511301977579), (672, 0.1178511301977579), (707, 0.1178511301977579), (731, 0.1178511301977579), (823, 0.1178511301977579), (840, 0.1178511301977579), (1053, 0.1178511301977579), (1137, 0.1178511301977579), (1219, 0.1178511301977579), (1322, 0.1178511301977579), (1336, 0.1178511301977579), (1372, 0.1178511301977579), (1392, 0.1178511301977579), (1417, 0.1178511301977579), (24, 0.11180339887498948), (32, 0.11180339887498948), (160, 0.11180339887498948), (165, 0.11180339887498948), (204, 0.11180339887498948), (213, 0.11180339887498948), (344, 0.11180339887498948), (413, 0.11180339887498948), (416, 0.11180339887498948), (473, 0.11180339887498948), (603, 0.11180339887498948), (710, 0.11180339887498948), (793, 0.11180339887498948), (945, 0.11180339887498948), (1091, 0.11180339887498948), (1140, 0.11180339887498948), (1293, 0.11180339887498948), (1323, 0.11180339887498948), (1324, 0.11180339887498948), (1352, 0.11180339887498948), (1357, 0.11180339887498948), (1398, 0.11180339887498948), (1411, 0.11180339887498948), (1416, 0.11180339887498948), (1424, 0.11180339887498948), (146, 0.10660035817780521), (264, 0.10660035817780521), (326, 0.10660035817780521), (337, 0.10660035817780521), (355, 0.10660035817780521), (395, 0.10660035817780521), (529, 0.10660035817780521), (547, 0.10660035817780521), (828, 0.10660035817780521), (1273, 0.10660035817780521), (1356, 0.10660035817780521), (1406, 0.10660035817780521), (1413, 0.10660035817780521), (1414, 0.10660035817780521), (1419, 0.10660035817780521), (16, 0.10206207261596575), (179, 0.10206207261596575), (221, 0.10206207261596575), (237, 0.10206207261596575), (363, 0.10206207261596575), (388, 0.10206207261596575), (559, 0.10206207261596575), (1300, 0.10206207261596575), (1379, 0.10206207261596575), (63, 0.09805806756909201), (275, 0.09805806756909201), (313, 0.09805806756909201), (382, 0.09805806756909201), (459, 0.09805806756909201), (316, 0.0944911182523068), (286, 0.09128709291752767), (362, 0.09128709291752767), (895, 0.09128709291752767), (1217, 0.09128709291752767), (1281, 0.09128709291752767), (152, 0.08838834764831843), (419, 0.08838834764831843), (460, 0.08838834764831843), (1092, 0.08574929257125441), (1329, 0.08574929257125441), (975, 0.08333333333333333), (277, 0.062499999999999986), (279, 0.06154574548966637), (28, 0.0), (30, 0.0), (33, 0.0), (36, 0.0), (38, 0.0), (39, 0.0), (45, 0.0), (48, 0.0), (52, 0.0), (55, 0.0), (56, 0.0), (58, 0.0), (59, 0.0), (60, 0.0), (61, 0.0), (64, 0.0), (66, 0.0), (67, 0.0), (69, 0.0), (70, 0.0), (71, 0.0), (72, 0.0), (73, 0.0), (74, 0.0), (75, 0.0), (78, 0.0), (80, 0.0), (83, 0.0), (84, 0.0), (85, 0.0), (86, 0.0), (87, 0.0), (88, 0.0), (89, 0.0), (90, 0.0), (92, 0.0), (93, 0.0), (95, 0.0), (97, 0.0), (99, 0.0), (100, 0.0), (101, 0.0), (102, 0.0), (103, 0.0), (104, 0.0), (105, 0.0), (106, 0.0), (107, 0.0), (109, 0.0), (110, 0.0), (111, 0.0), (112, 0.0), (113, 0.0), (114, 0.0), (115, 0.0), (116, 0.0), (117, 0.0), (118, 0.0), (119, 0.0), (120, 0.0), (121, 0.0), (122, 0.0), (123, 0.0), (124, 0.0), (125, 0.0), (128, 0.0), (133, 0.0), (135, 0.0), (136, 0.0), (140, 0.0), (141, 0.0), (143, 0.0), (145, 0.0), (147, 0.0), (148, 0.0), (149, 0.0), (151, 0.0), (157, 0.0), (158, 0.0), (159, 0.0), (161, 0.0), (162, 0.0), (163, 0.0), (164, 0.0), (166, 0.0), (167, 0.0), (168, 0.0), (169, 0.0), (170, 0.0), (171, 0.0), (172, 0.0), (173, 0.0), (174, 0.0), (175, 0.0), (176, 0.0), (177, 0.0), (178, 0.0), (180, 0.0), (181, 0.0), (182, 0.0), (183, 0.0), (184, 0.0), (185, 0.0), (186, 0.0), (187, 0.0), (188, 0.0), (190, 0.0), (191, 0.0), (192, 0.0), (193, 0.0), (195, 0.0), (196, 0.0), (197, 0.0), (198, 0.0), (199, 0.0), (200, 0.0), (201, 0.0), (202, 0.0), (203, 0.0), (207, 0.0), (208, 0.0), (209, 0.0), (210, 0.0), (211, 0.0), (212, 0.0), (215, 0.0), (216, 0.0), (217, 0.0), (218, 0.0), (219, 0.0), (222, 0.0), (223, 0.0), (224, 0.0), (226, 0.0), (227, 0.0), (228, 0.0), (238, 0.0), (239, 0.0), (240, 0.0), (241, 0.0), (242, 0.0), (243, 0.0), (244, 0.0), (245, 0.0), (246, 0.0), (247, 0.0), (248, 0.0), (249, 0.0), (250, 0.0), (251, 0.0), (252, 0.0), (253, 0.0), (255, 0.0), (256, 0.0), (257, 0.0), (258, 0.0), (259, 0.0), (260, 0.0), (261, 0.0), (263, 0.0), (265, 0.0), (266, 0.0), (267, 0.0), (268, 0.0), (269, 0.0), (270, 0.0), (271, 0.0), (272, 0.0), (273, 0.0), (274, 0.0), (280, 0.0), (282, 0.0), (283, 0.0), (284, 0.0), (285, 0.0), (287, 0.0), (288, 0.0), (289, 0.0), (290, 0.0), (291, 0.0), (292, 0.0), (293, 0.0), (294, 0.0), (296, 0.0), (297, 0.0), (298, 0.0), (299, 0.0), (300, 0.0), (301, 0.0), (302, 0.0), (303, 0.0), (304, 0.0), (305, 0.0), (306, 0.0), (307, 0.0), (308, 0.0), (309, 0.0), (310, 0.0), (311, 0.0), (312, 0.0), (314, 0.0), (315, 0.0), (317, 0.0), (318, 0.0), (320, 0.0), (322, 0.0), (323, 0.0), (324, 0.0), (325, 0.0), (327, 0.0), (329, 0.0), (331, 0.0), (333, 0.0), (334, 0.0), (335, 0.0), (336, 0.0), (338, 0.0), (339, 0.0), (340, 0.0), (341, 0.0), (342, 0.0), (345, 0.0), (346, 0.0), (347, 0.0), (348, 0.0), (349, 0.0), (351, 0.0), (352, 0.0), (353, 0.0), (354, 0.0), (356, 0.0), (357, 0.0), (358, 0.0), (359, 0.0), (361, 0.0), (364, 0.0), (365, 0.0), (366, 0.0), (367, 0.0), (368, 0.0), (369, 0.0), (370, 0.0), (371, 0.0), (372, 0.0), (373, 0.0), (374, 0.0), (375, 0.0), (376, 0.0), (377, 0.0), (378, 0.0), (379, 0.0), (380, 0.0), (381, 0.0), (383, 0.0), (384, 0.0), (385, 0.0), (386, 0.0), (387, 0.0), (389, 0.0), (390, 0.0), (391, 0.0), (392, 0.0), (393, 0.0), (394, 0.0), (396, 0.0), (397, 0.0), (398, 0.0), (399, 0.0), (400, 0.0), (401, 0.0), (402, 0.0), (403, 0.0), (404, 0.0), (405, 0.0), (406, 0.0), (407, 0.0), (408, 0.0), (409, 0.0), (410, 0.0), (411, 0.0), (412, 0.0), (414, 0.0), (415, 0.0), (417, 0.0), (418, 0.0), (420, 0.0), (421, 0.0), (422, 0.0), (423, 0.0), (424, 0.0), (425, 0.0), (426, 0.0), (427, 0.0), (428, 0.0), (429, 0.0), (430, 0.0), (432, 0.0), (433, 0.0), (434, 0.0), (435, 0.0), (436, 0.0), (438, 0.0), (439, 0.0), (440, 0.0), (441, 0.0), (442, 0.0), (443, 0.0), (444, 0.0), (445, 0.0), (446, 0.0), (447, 0.0), (448, 0.0), (449, 0.0), (450, 0.0), (451, 0.0), (452, 0.0), (453, 0.0), (454, 0.0), (455, 0.0), (456, 0.0), (457, 0.0), (458, 0.0), (461, 0.0), (462, 0.0), (463, 0.0), (464, 0.0), (466, 0.0), (468, 0.0), (469, 0.0), (470, 0.0), (471, 0.0), (477, 0.0), (478, 0.0), (481, 0.0), (482, 0.0), (487, 0.0), (488, 0.0), (489, 0.0), (490, 0.0), (491, 0.0), (492, 0.0), (493, 0.0), (494, 0.0), (495, 0.0), (496, 0.0), (497, 0.0), (498, 0.0), (499, 0.0), (500, 0.0), (501, 0.0), (502, 0.0), (503, 0.0), (505, 0.0), (506, 0.0), (507, 0.0), (508, 0.0), (509, 0.0), (511, 0.0), (512, 0.0), (513, 0.0), (514, 0.0), (515, 0.0), (516, 0.0), (518, 0.0), (519, 0.0), (520, 0.0), (521, 0.0), (522, 0.0), (523, 0.0), (524, 0.0), (525, 0.0), (526, 0.0), (527, 0.0), (528, 0.0), (531, 0.0), (532, 0.0), (533, 0.0), (534, 0.0), (535, 0.0), (536, 0.0), (539, 0.0), (540, 0.0), (541, 0.0), (542, 0.0), (543, 0.0), (544, 0.0), (545, 0.0), (546, 0.0), (549, 0.0), (550, 0.0), (551, 0.0), (552, 0.0), (553, 0.0), (554, 0.0), (555, 0.0), (556, 0.0), (557, 0.0), (558, 0.0), (561, 0.0), (563, 0.0), (564, 0.0), (565, 0.0), (566, 0.0), (567, 0.0), (568, 0.0), (569, 0.0), (570, 0.0), (571, 0.0), (572, 0.0), (574, 0.0), (576, 0.0), (577, 0.0), (578, 0.0), (579, 0.0), (580, 0.0), (582, 0.0), (583, 0.0), (584, 0.0), (586, 0.0), (587, 0.0), (589, 0.0), (593, 0.0), (594, 0.0), (595, 0.0), (596, 0.0), (597, 0.0), (598, 0.0), (599, 0.0), (600, 0.0), (601, 0.0), (602, 0.0), (604, 0.0), (610, 0.0), (611, 0.0), (612, 0.0), (614, 0.0), (615, 0.0), (616, 0.0), (618, 0.0), (619, 0.0), (622, 0.0), (623, 0.0), (624, 0.0), (625, 0.0), (627, 0.0), (629, 0.0), (630, 0.0), (631, 0.0), (632, 0.0), (633, 0.0), (634, 0.0), (635, 0.0), (636, 0.0), (638, 0.0), (639, 0.0), (640, 0.0), (641, 0.0), (642, 0.0), (643, 0.0), (644, 0.0), (647, 0.0), (648, 0.0), (649, 0.0), (650, 0.0), (651, 0.0), (652, 0.0), (653, 0.0), (654, 0.0), (655, 0.0), (656, 0.0), (657, 0.0), (658, 0.0), (659, 0.0), (660, 0.0), (661, 0.0), (662, 0.0), (663, 0.0), (664, 0.0), (665, 0.0), (666, 0.0), (667, 0.0), (668, 0.0), (669, 0.0), (670, 0.0), (671, 0.0), (673, 0.0), (674, 0.0), (675, 0.0), (676, 0.0), (677, 0.0), (679, 0.0), (680, 0.0), (681, 0.0), (682, 0.0), (683, 0.0), (684, 0.0), (685, 0.0), (686, 0.0), (687, 0.0), (688, 0.0), (689, 0.0), (690, 0.0), (691, 0.0), (692, 0.0), (693, 0.0), (694, 0.0), (698, 0.0), (699, 0.0), (700, 0.0), (701, 0.0), (702, 0.0), (703, 0.0), (704, 0.0), (705, 0.0), (706, 0.0), (708, 0.0), (709, 0.0), (711, 0.0), (712, 0.0), (713, 0.0), (715, 0.0), (716, 0.0), (717, 0.0), (718, 0.0), (719, 0.0), (720, 0.0), (721, 0.0), (722, 0.0), (724, 0.0), (725, 0.0), (726, 0.0), (727, 0.0), (728, 0.0), (729, 0.0), (730, 0.0), (732, 0.0), (733, 0.0), (734, 0.0), (735, 0.0), (737, 0.0), (738, 0.0), (739, 0.0), (740, 0.0), (741, 0.0), (742, 0.0), (743, 0.0), (744, 0.0), (745, 0.0), (746, 0.0), (747, 0.0), (748, 0.0), (749, 0.0), (750, 0.0), (751, 0.0), (752, 0.0), (753, 0.0), (754, 0.0), (755, 0.0), (757, 0.0), (758, 0.0), (759, 0.0), (760, 0.0), (761, 0.0), (762, 0.0), (764, 0.0), (765, 0.0), (766, 0.0), (767, 0.0), (768, 0.0), (769, 0.0), (770, 0.0), (772, 0.0), (773, 0.0), (774, 0.0), (775, 0.0), (776, 0.0), (777, 0.0), (778, 0.0), (780, 0.0), (781, 0.0), (782, 0.0), (783, 0.0), (784, 0.0), (785, 0.0), (786, 0.0), (787, 0.0), (789, 0.0), (790, 0.0), (791, 0.0), (792, 0.0), (794, 0.0), (795, 0.0), (796, 0.0), (797, 0.0), (798, 0.0), (799, 0.0), (800, 0.0), (801, 0.0), (802, 0.0), (803, 0.0), (805, 0.0), (806, 0.0), (807, 0.0), (808, 0.0), (810, 0.0), (811, 0.0), (814, 0.0), (815, 0.0), (816, 0.0), (817, 0.0), (818, 0.0), (819, 0.0), (822, 0.0), (824, 0.0), (825, 0.0), (826, 0.0), (827, 0.0), (829, 0.0), (830, 0.0), (831, 0.0), (832, 0.0), (833, 0.0), (834, 0.0), (835, 0.0), (836, 0.0), (837, 0.0), (838, 0.0), (839, 0.0), (841, 0.0), (842, 0.0), (843, 0.0), (844, 0.0), (845, 0.0), (846, 0.0), (847, 0.0), (849, 0.0), (850, 0.0), (851, 0.0), (852, 0.0), (853, 0.0), (854, 0.0), (855, 0.0), (856, 0.0), (857, 0.0), (858, 0.0), (859, 0.0), (860, 0.0), (861, 0.0), (863, 0.0), (866, 0.0), (867, 0.0), (868, 0.0), (869, 0.0), (870, 0.0), (871, 0.0), (872, 0.0), (873, 0.0), (874, 0.0), (875, 0.0), (876, 0.0), (877, 0.0), (878, 0.0), (879, 0.0), (881, 0.0), (882, 0.0), (883, 0.0), (884, 0.0), (885, 0.0), (887, 0.0), (889, 0.0), (890, 0.0), (891, 0.0), (892, 0.0), (893, 0.0), (894, 0.0), (896, 0.0), (897, 0.0), (898, 0.0), (899, 0.0), (900, 0.0), (901, 0.0), (902, 0.0), (903, 0.0), (904, 0.0), (906, 0.0), (907, 0.0), (909, 0.0), (910, 0.0), (911, 0.0), (912, 0.0), (913, 0.0), (914, 0.0), (915, 0.0), (916, 0.0), (917, 0.0), (918, 0.0), (919, 0.0), (921, 0.0), (922, 0.0), (923, 0.0), (924, 0.0), (925, 0.0), (926, 0.0), (927, 0.0), (928, 0.0), (929, 0.0), (930, 0.0), (931, 0.0), (932, 0.0), (933, 0.0), (934, 0.0), (935, 0.0), (936, 0.0), (937, 0.0), (939, 0.0), (940, 0.0), (941, 0.0), (942, 0.0), (943, 0.0), (944, 0.0), (946, 0.0), (947, 0.0), (948, 0.0), (949, 0.0), (951, 0.0), (952, 0.0), (955, 0.0), (956, 0.0), (957, 0.0), (958, 0.0), (959, 0.0), (960, 0.0), (961, 0.0), (962, 0.0), (963, 0.0), (964, 0.0), (965, 0.0), (966, 0.0), (968, 0.0), (972, 0.0), (974, 0.0), (976, 0.0), (977, 0.0), (978, 0.0), (979, 0.0), (980, 0.0), (981, 0.0), (982, 0.0), (983, 0.0), (984, 0.0), (985, 0.0), (987, 0.0), (988, 0.0), (989, 0.0), (990, 0.0), (991, 0.0), (992, 0.0), (993, 0.0), (994, 0.0), (995, 0.0), (996, 0.0), (998, 0.0), (999, 0.0), (1001, 0.0), (1002, 0.0), (1003, 0.0), (1004, 0.0), (1005, 0.0), (1006, 0.0), (1007, 0.0), (1008, 0.0), (1009, 0.0), (1010, 0.0), (1011, 0.0), (1012, 0.0), (1013, 0.0), (1014, 0.0), (1015, 0.0), (1016, 0.0), (1017, 0.0), (1018, 0.0), (1019, 0.0), (1020, 0.0), (1021, 0.0), (1022, 0.0), (1024, 0.0), (1025, 0.0), (1026, 0.0), (1027, 0.0), (1028, 0.0), (1029, 0.0), (1030, 0.0), (1031, 0.0), (1032, 0.0), (1033, 0.0), (1034, 0.0), (1035, 0.0), (1036, 0.0), (1037, 0.0), (1038, 0.0), (1039, 0.0), (1040, 0.0), (1041, 0.0), (1042, 0.0), (1044, 0.0), (1045, 0.0), (1046, 0.0), (1048, 0.0), (1049, 0.0), (1050, 0.0), (1051, 0.0), (1054, 0.0), (1055, 0.0), (1056, 0.0), (1057, 0.0), (1060, 0.0), (1061, 0.0), (1062, 0.0), (1063, 0.0), (1064, 0.0), (1065, 0.0), (1066, 0.0), (1067, 0.0), (1068, 0.0), (1069, 0.0), (1071, 0.0), (1072, 0.0), (1073, 0.0), (1074, 0.0), (1075, 0.0), (1076, 0.0), (1077, 0.0), (1078, 0.0), (1079, 0.0), (1080, 0.0), (1082, 0.0), (1083, 0.0), (1084, 0.0), (1085, 0.0), (1086, 0.0), (1087, 0.0), (1088, 0.0), (1089, 0.0), (1090, 0.0), (1093, 0.0), (1094, 0.0), (1095, 0.0), (1096, 0.0), (1097, 0.0), (1098, 0.0), (1099, 0.0), (1100, 0.0), (1101, 0.0), (1102, 0.0), (1103, 0.0), (1104, 0.0), (1105, 0.0), (1106, 0.0), (1107, 0.0), (1108, 0.0), (1109, 0.0), (1110, 0.0), (1111, 0.0), (1113, 0.0), (1114, 0.0), (1115, 0.0), (1116, 0.0), (1117, 0.0), (1118, 0.0), (1119, 0.0), (1120, 0.0), (1121, 0.0), (1124, 0.0), (1126, 0.0), (1127, 0.0), (1128, 0.0), (1129, 0.0), (1130, 0.0), (1131, 0.0), (1132, 0.0), (1133, 0.0), (1134, 0.0), (1135, 0.0), (1136, 0.0), (1138, 0.0), (1139, 0.0), (1141, 0.0), (1142, 0.0), (1143, 0.0), (1144, 0.0), (1145, 0.0), (1146, 0.0), (1147, 0.0), (1148, 0.0), (1149, 0.0), (1150, 0.0), (1151, 0.0), (1152, 0.0), (1153, 0.0), (1154, 0.0), (1155, 0.0), (1156, 0.0), (1157, 0.0), (1158, 0.0), (1159, 0.0), (1160, 0.0), (1161, 0.0), (1162, 0.0), (1163, 0.0), (1164, 0.0), (1166, 0.0), (1167, 0.0), (1168, 0.0), (1169, 0.0), (1170, 0.0), (1171, 0.0), (1173, 0.0), (1174, 0.0), (1175, 0.0), (1176, 0.0), (1177, 0.0), (1178, 0.0), (1179, 0.0), (1181, 0.0), (1183, 0.0), (1185, 0.0), (1188, 0.0), (1189, 0.0), (1191, 0.0), (1192, 0.0), (1193, 0.0), (1194, 0.0), (1195, 0.0), (1196, 0.0), (1197, 0.0), (1198, 0.0), (1199, 0.0), (1202, 0.0), (1203, 0.0), (1204, 0.0), (1205, 0.0), (1206, 0.0), (1208, 0.0), (1210, 0.0), (1211, 0.0), (1212, 0.0), (1213, 0.0), (1214, 0.0), (1215, 0.0), (1216, 0.0), (1218, 0.0), (1220, 0.0), (1221, 0.0), (1222, 0.0), (1223, 0.0), (1225, 0.0), (1226, 0.0), (1227, 0.0), (1229, 0.0), (1230, 0.0), (1231, 0.0), (1232, 0.0), (1233, 0.0), (1234, 0.0), (1236, 0.0), (1237, 0.0), (1238, 0.0), (1240, 0.0), (1241, 0.0), (1242, 0.0), (1243, 0.0), (1244, 0.0), (1245, 0.0), (1246, 0.0), (1247, 0.0), (1248, 0.0), (1250, 0.0), (1251, 0.0), (1254, 0.0), (1255, 0.0), (1256, 0.0), (1257, 0.0), (1258, 0.0), (1259, 0.0), (1260, 0.0), (1261, 0.0), (1262, 0.0), (1264, 0.0), (1265, 0.0), (1270, 0.0), (1271, 0.0), (1272, 0.0), (1275, 0.0), (1277, 0.0), (1279, 0.0), (1280, 0.0), (1282, 0.0), (1283, 0.0), (1284, 0.0), (1285, 0.0), (1287, 0.0), (1288, 0.0), (1289, 0.0), (1290, 0.0), (1291, 0.0), (1292, 0.0), (1294, 0.0), (1295, 0.0), (1296, 0.0), (1297, 0.0), (1298, 0.0), (1304, 0.0), (1306, 0.0), (1307, 0.0), (1308, 0.0), (1309, 0.0), (1311, 0.0), (1312, 0.0), (1314, 0.0), (1316, 0.0), (1317, 0.0), (1318, 0.0), (1319, 0.0), (1320, 0.0), (1321, 0.0), (1325, 0.0), (1327, 0.0), (1328, 0.0), (1330, 0.0), (1331, 0.0), (1332, 0.0), (1333, 0.0), (1334, 0.0), (1335, 0.0), (1337, 0.0), (1338, 0.0), (1339, 0.0), (1341, 0.0), (1343, 0.0), (1344, 0.0), (1345, 0.0), (1346, 0.0), (1347, 0.0), (1348, 0.0), (1350, 0.0), (1351, 0.0), (1353, 0.0), (1354, 0.0), (1355, 0.0), (1358, 0.0), (1359, 0.0), (1362, 0.0), (1363, 0.0), (1364, 0.0), (1367, 0.0), (1368, 0.0), (1370, 0.0), (1371, 0.0), (1373, 0.0), (1374, 0.0), (1376, 0.0), (1377, 0.0), (1378, 0.0), (1380, 0.0), (1381, 0.0), (1383, 0.0), (1385, 0.0), (1386, 0.0), (1387, 0.0), (1388, 0.0), (1391, 0.0), (1393, 0.0), (1394, 0.0), (1395, 0.0), (1396, 0.0), (1397, 0.0), (1399, 0.0), (1400, 0.0), (1402, 0.0), (1403, 0.0), (1404, 0.0), (1405, 0.0), (1409, 0.0), (1410, 0.0), (1412, 0.0), (1418, 0.0), (1420, 0.0), (1421, 0.0), (1425, 0.0), (1426, 0.0), (1430, 0.0), (1431, 0.0), (1432, 0.0), (1433, 0.0), (1434, 0.0), (1435, 0.0), (1436, 0.0), (1437, 0.0), (1438, 0.0), (1440, 0.0), (1441, 0.0), (1442, 0.0), (1443, 0.0), (1444, 0.0), (1445, 0.0), (1446, 0.0), (1447, 0.0), (1448, 0.0), (1449, 0.0), (1450, 0.0), (1451, 0.0), (1452, 0.0), (1453, 0.0), (1454, 0.0), (1455, 0.0), (1458, 0.0), (1459, 0.0), (1461, 0.0), (1463, 0.0)]



```python
#Create a loop for first 7 movie recommendations
j = 0
print('The 7 most recommended movies to', Title, 'are:\n')
for item in sorted_scores:
    movie_title = master3[master3.MovieId ==item[0]]['title'].values[0]
    print(j+1, movie_title)
    j = j + 1
    if j > 6:
        break
    
```

    The 7 most recommended movies to Jumanji (1995) are:
    
    1 Toy Story (1995)
    2 Sintel (2010)
    3 Watership Down (1978)
    4 Toy Story 2 (1999)
    5 Tomb Raider (2018)
    6 Wizard of Oz, The (1939)
    7 Alice in Wonderland (1951)



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
merge_list = master.groupby(by = ["userId"])["title"].apply(list).reset_index()
merge_list.head()
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
      <th>userId</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>[Step Brothers (2008), Step Brothers (2008), S...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>7</td>
      <td>[Departed, The (2006)]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>18</td>
      <td>[Carlito's Way (1993), Carlito's Way (1993), C...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>21</td>
      <td>[My Best Friend's Wedding (1997), My Best Frie...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>49</td>
      <td>[Interstellar (2014), Interstellar (2014), Int...</td>
    </tr>
  </tbody>
</table>
</div>




```python
merge_list = merge_list["title"].tolist()
merge_list[0:2]
```




    [['Step Brothers (2008)',
      'Step Brothers (2008)',
      'Step Brothers (2008)',
      'Warrior (2011)',
      'Warrior (2011)',
      'Warrior (2011)',
      'Wolf of Wall Street, The (2013)',
      'Wolf of Wall Street, The (2013)',
      'Wolf of Wall Street, The (2013)'],
     ['Departed, The (2006)']]




```python
merge_list
```




    [['Step Brothers (2008)',
      'Step Brothers (2008)',
      'Step Brothers (2008)',
      'Warrior (2011)',
      'Warrior (2011)',
      'Warrior (2011)',
      'Wolf of Wall Street, The (2013)',
      'Wolf of Wall Street, The (2013)',
      'Wolf of Wall Street, The (2013)'],
     ['Departed, The (2006)'],
     ["Carlito's Way (1993)",
      "Carlito's Way (1993)",
      "Carlito's Way (1993)",
      'Godfather: Part II, The (1974)',
      'Godfather: Part II, The (1974)',
      'Pianist, The (2002)',
      'Pianist, The (2002)',
      'Lucky Number Slevin (2006)',
      'Fracture (2007)',
      'Fracture (2007)',
      'Fracture (2007)',
      'Upside Down: The Creation Records Story (2010)',
      'Upside Down: The Creation Records Story (2010)',
      'Upside Down: The Creation Records Story (2010)',
      'Just Eat It: A Food Waste Story (2014)',
      'Just Eat It: A Food Waste Story (2014)'],
     ["My Best Friend's Wedding (1997)",
      "My Best Friend's Wedding (1997)",
      'Big Eyes (2014)',
      'The Interview (2014)'],
     ['Interstellar (2014)', 'Interstellar (2014)', 'Interstellar (2014)'],
     ['Jumanji (1995)',
      'Jumanji (1995)',
      'Jumanji (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Braveheart (1995)',
      'Addams Family Values (1993)',
      'Addams Family Values (1993)',
      'Addams Family Values (1993)',
      'Addams Family Values (1993)',
      'Addams Family Values (1993)',
      'Addams Family Values (1993)',
      'Godfather: Part III, The (1990)',
      'Godfather: Part III, The (1990)',
      'Godfather: Part III, The (1990)',
      'Godfather: Part III, The (1990)',
      'Godfather: Part III, The (1990)',
      'Addams Family, The (1991)',
      'Addams Family, The (1991)',
      'Addams Family, The (1991)',
      'Addams Family, The (1991)',
      'Addams Family, The (1991)',
      'Addams Family, The (1991)',
      'Home Alone 2: Lost in New York (1992)',
      'Home Alone 2: Lost in New York (1992)',
      'Home Alone 2: Lost in New York (1992)',
      'Home Alone 2: Lost in New York (1992)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Toy Story 2 (1999)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Gladiator (2000)',
      'Enemy at the Gates (2001)',
      'Enemy at the Gates (2001)',
      'Enemy at the Gates (2001)',
      'Enemy at the Gates (2001)',
      'Enemy at the Gates (2001)',
      'Insomnia (2002)',
      'Insomnia (2002)',
      'Insomnia (2002)',
      'Insomnia (2002)',
      'Final Destination 2 (2003)',
      'Final Destination 2 (2003)',
      'Hulk (2003)',
      'Hulk (2003)',
      'Hulk (2003)',
      'Hulk (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'League of Extraordinary Gentlemen, The (a.k.a. LXG) (2003)',
      'Lara Croft Tomb Raider: The Cradle of Life (2003)',
      'Lara Croft Tomb Raider: The Cradle of Life (2003)',
      'Lara Croft Tomb Raider: The Cradle of Life (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Lord of the Rings: The Return of the King, The (2003)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Animatrix, The (2003)',
      'Animatrix, The (2003)',
      'Animatrix, The (2003)',
      'Animatrix, The (2003)',
      'Animatrix, The (2003)',
      'Animatrix, The (2003)',
      "Lemony Snicket's A Series of Unfortunate Events (2004)",
      "Lemony Snicket's A Series of Unfortunate Events (2004)",
      "Lemony Snicket's A Series of Unfortunate Events (2004)",
      "Lemony Snicket's A Series of Unfortunate Events (2004)",
      "Lemony Snicket's A Series of Unfortunate Events (2004)",
      'Spanglish (2004)',
      'Spanglish (2004)',
      'Spanglish (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      'Layer Cake (2004)',
      "Howl's Moving Castle (Hauru no ugoku shiro) (2004)",
      "Howl's Moving Castle (Hauru no ugoku shiro) (2004)",
      "Howl's Moving Castle (Hauru no ugoku shiro) (2004)",
      "Howl's Moving Castle (Hauru no ugoku shiro) (2004)",
      'Kingdom of Heaven (2005)',
      'Kingdom of Heaven (2005)',
      'Kingdom of Heaven (2005)',
      'Kingdom of Heaven (2005)',
      'Fantastic Four (2005)',
      'Fantastic Four (2005)',
      'Fantastic Four (2005)',
      'Fantastic Four (2005)',
      'Fantastic Four (2005)',
      'Fantastic Four (2005)',
      'Corpse Bride (2005)',
      'Corpse Bride (2005)',
      'Corpse Bride (2005)',
      'Corpse Bride (2005)',
      'Corpse Bride (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Kiss Kiss Bang Bang (2005)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Da Vinci Code, The (2006)',
      'Babel (2006)',
      'Babel (2006)',
      'Babel (2006)',
      'Babel (2006)',
      'Night at the Museum (2006)',
      'Night at the Museum (2006)',
      'Stranger than Fiction (2006)',
      'Stranger than Fiction (2006)',
      'Stranger than Fiction (2006)',
      'Stranger than Fiction (2006)',
      'Stranger than Fiction (2006)',
      'Stranger than Fiction (2006)',
      'Good Year, A (2006)',
      'Blood Diamond (2006)',
      'Blood Diamond (2006)',
      'Blood Diamond (2006)',
      'Blood Diamond (2006)',
      'Blood Diamond (2006)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Fantastic Four: Rise of the Silver Surfer (2007)',
      'Definitely, Maybe (2008)',
      'Definitely, Maybe (2008)',
      'Chronicles of Narnia: Prince Caspian, The (2008)',
      'Chronicles of Narnia: Prince Caspian, The (2008)',
      'Chronicles of Narnia: Prince Caspian, The (2008)',
      'Chronicles of Narnia: Prince Caspian, The (2008)',
      'Hancock (2008)',
      'Hancock (2008)',
      'Hancock (2008)',
      'Get Smart (2008)',
      'Get Smart (2008)',
      'Step Brothers (2008)',
      'Step Brothers (2008)',
      'Step Brothers (2008)',
      'Pineapple Express (2008)',
      'Pineapple Express (2008)',
      'Pineapple Express (2008)',
      'Pineapple Express (2008)',
      'Twilight (2008)',
      'Twilight (2008)',
      'Twilight (2008)',
      'Twilight (2008)',
      'Twilight (2008)',
      'Brothers Bloom, The (2008)',
      'Brothers Bloom, The (2008)',
      'Brothers Bloom, The (2008)',
      'Brothers Bloom, The (2008)',
      'Brothers Bloom, The (2008)',
      'Brothers Bloom, The (2008)',
      'Zombieland (2009)',
      'Zombieland (2009)',
      'Zombieland (2009)',
      'Zombieland (2009)',
      'Zombieland (2009)',
      'Zombieland (2009)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Green Lantern (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Friends with Benefits (2011)',
      'Taken 2 (2012)',
      'Taken 2 (2012)',
      'Taken 2 (2012)',
      'Taken 2 (2012)',
      'Taken 2 (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Django Unchained (2012)',
      'Man of Steel (2013)',
      'Man of Steel (2013)',
      'What If (2013)',
      'What If (2013)',
      'Anchorman 2: The Legend Continues (2013)',
      'Anchorman 2: The Legend Continues (2013)',
      'Anchorman 2: The Legend Continues (2013)',
      'Anchorman 2: The Legend Continues (2013)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'Divergent (2014)',
      'A Million Ways to Die in the West (2014)',
      'A Million Ways to Die in the West (2014)',
      'A Million Ways to Die in the West (2014)',
      'A Million Ways to Die in the West (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'Maze Runner, The (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'Wild Tales (2014)',
      'Wild Tales (2014)',
      'Wild Tales (2014)',
      'Wild Tales (2014)',
      'Wild Tales (2014)',
      'Wild Tales (2014)',
      'The Interview (2014)',
      'The Interview (2014)',
      'The Interview (2014)',
      'The Interview (2014)',
      'The Interview (2014)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hateful Eight (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'The Hunger Games: Mockingjay - Part 2 (2015)',
      'Self/less (2015)',
      'Self/less (2015)',
      'Self/less (2015)',
      'Self/less (2015)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Suicide Squad (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'Batman v Superman: Dawn of Justice (2016)',
      'The Revenant (2015)',
      'The Revenant (2015)',
      'The Revenant (2015)',
      'The Revenant (2015)',
      'Captain Fantastic (2016)',
      'Captain Fantastic (2016)',
      'Captain Fantastic (2016)',
      'Captain Fantastic (2016)',
      'Captain Fantastic (2016)',
      'Captain Fantastic (2016)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'John Wick: Chapter Two (2017)',
      'Black Mirror: White Christmas (2014)',
      'Black Mirror: White Christmas (2014)',
      'Black Mirror: White Christmas (2014)',
      'Black Mirror: White Christmas (2014)',
      'Black Mirror: White Christmas (2014)',
      'Jumanji: Welcome to the Jungle (2017)',
      'Jumanji: Welcome to the Jungle (2017)',
      'Jumanji: Welcome to the Jungle (2017)',
      'Jumanji: Welcome to the Jungle (2017)',
      'Game Night (2018)',
      'Game Night (2018)',
      'Game Night (2018)',
      'Tomb Raider (2018)',
      'Tomb Raider (2018)',
      'Tomb Raider (2018)',
      'Deadpool 2 (2018)',
      'Deadpool 2 (2018)',
      'Deadpool 2 (2018)',
      'Solo: A Star Wars Story (2018)',
      'Solo: A Star Wars Story (2018)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)'],
     ["Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      'Hobbit: The Desolation of Smaug, The (2013)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Croods, The (2013)',
      'Croods, The (2013)',
      'Croods, The (2013)',
      'John Wick (2014)',
      'John Wick (2014)',
      'John Wick (2014)',
      'Big Hero 6 (2014)',
      'Big Hero 6 (2014)',
      'Big Hero 6 (2014)',
      'Taken 3 (2015)',
      'Taken 3 (2015)',
      'Taken 3 (2015)'],
     ['Postman, The (1997)',
      'Postman, The (1997)',
      'Very Bad Things (1998)',
      'Very Bad Things (1998)',
      'Dogma (1999)',
      'Dogma (1999)',
      'Dogma (1999)',
      'Dogma (1999)',
      'Going Places (Valseuses, Les) (1974)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'Battle Royale (Batoru rowaiaru) (2000)',
      'The Butterfly Effect (2004)',
      'The Butterfly Effect (2004)',
      'The Butterfly Effect (2004)',
      'The Butterfly Effect (2004)',
      'Saw (2004)',
      'Saw (2004)',
      'Saw (2004)',
      'Saw (2004)',
      'Saw (2004)',
      'Saw (2004)',
      'Saw (2004)',
      "Love Me If You Dare (Jeux d'enfants) (2003)",
      'Lady Vengeance (Sympathy for Lady Vengeance) (Chinjeolhan geumjassi) (2005)',
      'Funny Games U.S. (2007)',
      'Vicky Cristina Barcelona (2008)',
      'Zack and Miri Make a Porno (2008)',
      'Zack and Miri Make a Porno (2008)',
      'Observe and Report (2009)',
      'Movie 43 (2013)',
      'Movie 43 (2013)',
      'Movie 43 (2013)',
      'Movie 43 (2013)',
      'Movie 43 (2013)',
      'Everybody Wants Some (2016)',
      'Sausage Party (2016)',
      'Sausage Party (2016)',
      'Sausage Party (2016)',
      'Sausage Party (2016)',
      'Sausage Party (2016)'],
     ['Virgin Suicides, The (1999)',
      'Identity (2003)',
      'Down with Love (2003)',
      'Down with Love (2003)'],
     ['Forbidden Kingdom, The (2008)', 'Forbidden Kingdom, The (2008)'],
     ['Meet the Robinsons (2007)'],
     ['Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Bourne Ultimatum, The (2007)',
      'Bourne Ultimatum, The (2007)',
      'Bourne Ultimatum, The (2007)',
      'Bourne Ultimatum, The (2007)'],
     ['Happy Gilmore (1996)'],
     ['Big Hero 6 (2014)', 'Big Hero 6 (2014)', 'Big Hero 6 (2014)'],
     ['Following (1998)',
      'Following (1998)',
      'Following (1998)',
      'Following (1998)',
      'Following (1998)',
      'Following (1998)',
      'X-Men (2000)',
      'X-Men (2000)',
      'X-Men (2000)',
      'X-Men (2000)',
      'X-Men (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Memento (2000)',
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      'Insomnia (2002)',
      'Insomnia (2002)',
      'Insomnia (2002)',
      'Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku no Tobira) (2001)',
      'Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku no Tobira) (2001)',
      'Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku no Tobira) (2001)',
      'Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997)',
      'Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997)',
      'Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997)',
      'Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997)',
      'Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni) (1997)',
      'Gintama: The Movie (2010)',
      'Gintama: The Movie (2010)',
      'Gintama: The Movie (2010)',
      'Gintama: The Movie (2010)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Big Fish (2003)',
      'Big Fish (2003)',
      'Big Fish (2003)',
      'Big Fish (2003)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Life of Pi (2012)',
      'Life of Pi (2012)',
      'Life of Pi (2012)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)'],
     ['Billabong Odyssey (2003)',
      'Billabong Odyssey (2003)',
      'Prestige, The (2006)'],
     ['The DUFF (2015)', 'The DUFF (2015)'],
     ['X-Men Origins: Wolverine (2009)'],
     ['Proof (1991)'],
     ['Grumpier Old Men (1995)',
      'Grumpier Old Men (1995)',
      "Schindler's List (1993)",
      'Top Gun (1986)'],
     ['Ratatouille (2007)',
      'Ratatouille (2007)',
      'Invincible Iron Man, The (2007)',
      'The Hobbit: The Battle of the Five Armies (2014)'],
     ['Lost in Translation (2003)'],
     ['Traffic (2000)',
      'Beautiful Mind, A (2001)',
      '28 Days Later (2002)',
      '21 Grams (2003)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Serenity (2005)',
      'Elite Squad (Tropa de Elite) (2007)',
      'Tucker & Dale vs Evil (2010)'],
     ['Trainspotting (1996)',
      'Trainspotting (1996)',
      'Trainspotting (1996)',
      'Jesus of Montreal (Jésus de Montréal) (1989)',
      'Marat/Sade (1966)',
      'Murder on a Sunday Morning (Un coupable idéal) (2001)',
      'In the Realms of the Unreal (2004)',
      'In the Realms of the Unreal (2004)',
      'In the Realms of the Unreal (2004)',
      'Deliver Us from Evil (2006)',
      'Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)',
      'Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Haunted World of El Superbeasto, The (2009)',
      'Restrepo (2010)',
      'True Grit (2010)',
      'True Grit (2010)',
      'True Grit (2010)',
      'True Grit (2010)',
      'True Grit (2010)',
      'Starsuckers (2009)',
      'Starsuckers (2009)',
      'Starsuckers (2009)',
      'Hands Over the City (Le mani sulla città) (1963)',
      'Hands Over the City (Le mani sulla città) (1963)',
      'Hands Over the City (Le mani sulla città) (1963)',
      'Godzilla (2014)',
      'Good Copy Bad Copy (2007)',
      'Good Copy Bad Copy (2007)',
      'Good Copy Bad Copy (2007)',
      'A Story of Children and Film (2013)',
      'A Story of Children and Film (2013)',
      'A Story of Children and Film (2013)'],
     ['Lion King, The (1994)', 'Lion King, The (1994)', 'Lion King, The (1994)'],
     ['This Is Spinal Tap (1984)',
      "There's Something About Mary (1998)",
      'Dick Tracy (1990)',
      'High Fidelity (2000)',
      'Almost Famous (2000)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Hot Fuzz (2007)'],
     ['Toy Story (1995)',
      'Batman Forever (1995)',
      'Three Musketeers, The (1993)',
      'Dead Poets Society (1989)',
      'Sin City (2005)',
      'Cinderella Man (2005)',
      'Wedding Crashers (2005)',
      'Lord of War (2005)',
      'Corpse Bride (2005)',
      'In Her Shoes (2005)'],
     ['Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)'],
     ["St. Elmo's Fire (1985)",
      'Waiting... (2005)',
      'Breakfast on Pluto (2005)',
      'Reefer Madness: The Movie Musical (2005)',
      'Burn After Reading (2008)'],
     ['Clueless (1995)',
      'Clueless (1995)',
      'Clueless (1995)',
      'Clueless (1995)',
      'Clueless (1995)',
      "William Shakespeare's Romeo + Juliet (1996)",
      "William Shakespeare's Romeo + Juliet (1996)",
      "William Shakespeare's Romeo + Juliet (1996)",
      "William Shakespeare's Romeo + Juliet (1996)",
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      'A.I. Artificial Intelligence (2001)',
      'A.I. Artificial Intelligence (2001)',
      'A.I. Artificial Intelligence (2001)',
      'A.I. Artificial Intelligence (2001)',
      'Road to Perdition (2002)',
      'Road to Perdition (2002)',
      'Jezebel (1938)',
      'Jezebel (1938)',
      '13 Going on 30 (2004)',
      'Mean Girls (2004)',
      'Mean Girls (2004)',
      'Mean Girls (2004)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      '3:10 to Yuma (2007)',
      'Gone Baby Gone (2007)',
      'Gone Baby Gone (2007)',
      'Gone Baby Gone (2007)',
      'The Hunger Games (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)',
      'Dark Knight Rises, The (2012)'],
     ['Zero Dark Thirty (2012)',
      'Zero Dark Thirty (2012)',
      'Zero Dark Thirty (2012)',
      'Zero Dark Thirty (2012)',
      'Zero Dark Thirty (2012)'],
     ['Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Seven (a.k.a. Se7en) (1995)',
      'Seven (a.k.a. Se7en) (1995)',
      'Usual Suspects, The (1995)',
      'Usual Suspects, The (1995)',
      'Usual Suspects, The (1995)',
      'Usual Suspects, The (1995)',
      'Usual Suspects, The (1995)',
      'Basketball Diaries, The (1995)',
      'Basketball Diaries, The (1995)',
      'Basketball Diaries, The (1995)',
      'Basketball Diaries, The (1995)',
      'Kids (1995)',
      'Kids (1995)',
      'Kids (1995)',
      'Clerks (1994)',
      'Clerks (1994)',
      'Clerks (1994)',
      'Clerks (1994)',
      'Clerks (1994)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Natural Born Killers (1994)',
      'Natural Born Killers (1994)',
      'Natural Born Killers (1994)',
      'Natural Born Killers (1994)',
      'Natural Born Killers (1994)',
      'Natural Born Killers (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Lion King, The (1994)',
      "Schindler's List (1993)",
      'Blade Runner (1982)',
      'Terminator 2: Judgment Day (1991)',
      'Terminator 2: Judgment Day (1991)',
      'Terminator 2: Judgment Day (1991)',
      'Terminator 2: Judgment Day (1991)',
      'Terminator 2: Judgment Day (1991)',
      'Terminator 2: Judgment Day (1991)',
      'Fargo (1996)',
      'Fargo (1996)',
      'Fargo (1996)',
      'Fargo (1996)',
      'Primal Fear (1996)',
      'Primal Fear (1996)',
      'Primal Fear (1996)',
      'Primal Fear (1996)',
      'Primal Fear (1996)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'People vs. Larry Flynt, The (1996)',
      'Monty Python and the Holy Grail (1975)',
      'Monty Python and the Holy Grail (1975)',
      "One Flew Over the Cuckoo's Nest (1975)",
      "One Flew Over the Cuckoo's Nest (1975)",
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Aliens (1986)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Terminator, The (1984)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Shining, The (1980)',
      'Cape Fear (1991)',
      'Cape Fear (1991)',
      'Cape Fear (1991)',
      'Cape Fear (1991)',
      'Game, The (1997)',
      'Game, The (1997)',
      'Game, The (1997)',
      'Game, The (1997)',
      'Game, The (1997)',
      'Game, The (1997)',
      'Good Will Hunting (1997)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      "There's Something About Mary (1998)",
      "There's Something About Mary (1998)",
      "There's Something About Mary (1998)",
      "Rosemary's Baby (1968)",
      "Rosemary's Baby (1968)",
      "Rosemary's Baby (1968)",
      "Rosemary's Baby (1968)",
      "Rosemary's Baby (1968)",
      'Night at the Roxbury, A (1998)',
      'American History X (1998)',
      '8MM (1999)',
      'Matrix, The (1999)',
      'Matrix, The (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'South Park: Bigger, Longer and Uncut (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Girl, Interrupted (1999)',
      'Girl, Interrupted (1999)',
      'Girl, Interrupted (1999)',
      'Girl, Interrupted (1999)',
      'Girl, Interrupted (1999)',
      'Girl, Interrupted (1999)',
      'Misery (1990)',
      'Misery (1990)',
      'Misery (1990)',
      'Misery (1990)',
      'Misery (1990)',
      'Misery (1990)',
      'Misery (1990)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Best in Show (2000)',
      'Meet the Parents (2000)',
      'Meet the Parents (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Scarface (1983)',
      'Zoolander (2001)',
      'Zoolander (2001)',
      'Zoolander (2001)',
      'Zoolander (2001)',
      'Zoolander (2001)',
      'Zoolander (2001)',
      'Donnie Darko (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Rashomon (Rashômon) (1950)',
      'Rashomon (Rashômon) (1950)',
      'City of God (Cidade de Deus) (2002)',
      'City of God (Cidade de Deus) (2002)',
      'Old School (2003)',
      'Old School (2003)',
      'House of 1000 Corpses (2003)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'Anchorman: The Legend of Ron Burgundy (2004)',
      'The Machinist (2004)',
      'The Machinist (2004)',
      'The Machinist (2004)',
      'The Machinist (2004)',
      'The Machinist (2004)',
      'The Machinist (2004)',
      'Gia (1998)',
      'Gia (1998)',
      'Meet the Fockers (2004)',
      'Meet the Fockers (2004)',
      "Devil's Rejects, The (2005)",
      'Jarhead (2005)',
      'Jarhead (2005)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Departed, The (2006)',
      'Bug (2007)',
      'Bug (2007)',
      'Bug (2007)',
      'Bug (2007)',
      'Bug (2007)',
      'Bug (2007)',
      'Step Brothers (2008)',
      'Step Brothers (2008)',
      'Coraline (2009)',
      'Coraline (2009)',
      'Coraline (2009)',
      'Coraline (2009)',
      'Coraline (2009)',
      'Coraline (2009)',
      'Inglourious Basterds (2009)',
      'Inglourious Basterds (2009)',
      'Inglourious Basterds (2009)',
      'Inglourious Basterds (2009)',
      'Inglourious Basterds (2009)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Shutter Island (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      '127 Hours (2010)',
      'Black Swan (2010)',
      'Black Swan (2010)',
      'Black Swan (2010)',
      'Black Swan (2010)',
      'Black Swan (2010)',
      'Black Swan (2010)',
      'Now You See Me (2013)',
      'Now You See Me (2013)',
      'Now You See Me (2013)',
      'Now You See Me (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Prisoners (2013)',
      'Dallas Buyers Club (2013)',
      'Dallas Buyers Club (2013)',
      'Dallas Buyers Club (2013)',
      'Hobbit: The Desolation of Smaug, The (2013)',
      'Hobbit: The Desolation of Smaug, The (2013)',
      'Hobbit: The Desolation of Smaug, The (2013)',
      'Interstellar (2014)',
      'Interstellar (2014)',
      'Interstellar (2014)',
      'Babadook, The (2014)',
      'Babadook, The (2014)',
      'Babadook, The (2014)',
      'Babadook, The (2014)',
      'Gone Girl (2014)',
      'Gone Girl (2014)'],
     ['Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Fight Club (1999)',
      'Dark Knight, The (2008)',
      'Dark Knight, The (2008)'],
     ['Lord of the Rings: The Two Towers, The (2002)',
      'Hobbit: An Unexpected Journey, The (2012)',
      'Hobbit: An Unexpected Journey, The (2012)'],
     ['Who Killed Chea Vichea? (2010)',
      'Who Killed Chea Vichea? (2010)',
      'Who Killed Chea Vichea? (2010)',
      'Who Killed Chea Vichea? (2010)',
      'Who Killed Chea Vichea? (2010)'],
     ['Toy Story (1995)',
      'Jumanji (1995)',
      'Father of the Bride Part II (1995)',
      'Father of the Bride Part II (1995)',
      'Sabrina (1995)',
      'American President, The (1995)',
      'American President, The (1995)',
      'Nixon (1995)',
      'Nixon (1995)',
      'Casino (1995)',
      'Sense and Sensibility (1995)',
      'Get Shorty (1995)',
      'Copycat (1995)',
      'Leaving Las Vegas (1995)',
      'Othello (1995)',
      'Persuasion (1995)',
      'Persuasion (1995)',
      'City of Lost Children, The (Cité des enfants perdus, La) (1995)',
      'Dangerous Minds (1995)',
      'Dangerous Minds (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Babe (1995)',
      'Babe (1995)',
      'Dead Man Walking (1995)',
      'Dead Man Walking (1995)',
      'It Takes Two (1995)',
      'Clueless (1995)',
      'Clueless (1995)',
      'Cry, the Beloved Country (1995)',
      'Cry, the Beloved Country (1995)',
      'Richard III (1995)',
      'Restoration (1995)',
      'To Die For (1995)',
      'How to Make an American Quilt (1995)',
      'Seven (a.k.a. Se7en) (1995)',
      'Usual Suspects, The (1995)',
      'Mighty Aphrodite (1995)',
      'Mighty Aphrodite (1995)',
      'Postman, The (Postino, Il) (1994)',
      "Mr. Holland's Opus (1995)",
      'Mary Reilly (1996)',
      'In the Bleak Midwinter (1995)',
      'Bottle Rocket (1996)',
      'Happy Gilmore (1996)',
      'Muppet Treasure Island (1996)',
      'Braveheart (1995)',
      'Taxi Driver (1976)',
      'Anne Frank Remembered (1995)',
      'Boomerang (1992)',
      'Up Close and Personal (1996)',
      'Apollo 13 (1995)',
      'Apollo 13 (1995)',
      'Apollo 13 (1995)',
      'Batman Forever (1995)',
      'Congo (1995)',
      'Crimson Tide (1995)',
      'Crumb (1994)',
      'Net, The (1995)',
      'Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)',
      'Before Sunrise (1995)',
      'Billy Madison (1995)',
      'Circle of Friends (1995)',
      'Clerks (1994)',
      'Don Juan DeMarco (1995)',
      'Don Juan DeMarco (1995)',
      'Dolores Claiborne (1995)',
      'Eat Drink Man Woman (Yin shi nan nu) (1994)',
      'Ed Wood (1994)',
      'Forget Paris (1995)',
      'Forget Paris (1995)',
      'Forget Paris (1995)',
      'Hoop Dreams (1994)',
      'Heavenly Creatures (1994)',
      'Immortal Beloved (1994)',
      'I.Q. (1994)',
      'Just Cause (1995)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Star Wars: Episode IV - A New Hope (1977)',
      'Little Women (1994)',
      'Little Princess, A (1995)',
      'Little Princess, A (1995)',
      'Little Princess, A (1995)',
      'Madness of King George, The (1994)',
      'Madness of King George, The (1994)',
      'Miracle on 34th Street (1994)',
      'My Family (1995)',
      'Murder in the First (1995)',
      'Nell (1994)',
      'Nell (1994)',
      'Once Were Warriors (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Pulp Fiction (1994)',
      'Quiz Show (1994)',
      'Three Colors: Blue (Trois couleurs: Bleu) (1993)',
      'Three Colors: White (Trzy kolory: Bialy) (1994)',
      'Stargate (1994)',
      'Santa Clause, The (1994)',
      'Shawshank Redemption, The (1994)',
      'Shawshank Redemption, The (1994)',
      'Shawshank Redemption, The (1994)',
      'To Live (Huozhe) (1994)',
      'Star Trek: Generations (1994)',
      "What's Eating Gilbert Grape (1993)",
      'Virtuosity (1995)',
      'While You Were Sleeping (1995)',
      "Muriel's Wedding (1994)",
      "Muriel's Wedding (1994)",
      'Adventures of Priscilla, Queen of the Desert, The (1994)',
      'Adventures of Priscilla, Queen of the Desert, The (1994)',
      'Adventures of Priscilla, Queen of the Desert, The (1994)',
      'Clear and Present Danger (1994)',
      'Client, The (1994)',
      'Corrina, Corrina (1994)',
      'Forrest Gump (1994)',
      'Forrest Gump (1994)',
      'Four Weddings and a Funeral (1994)',
      'It Could Happen to You (1994)',
      'Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993)',
      'Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993)',
      'Lion King, The (1994)',
      'Paper, The (1994)',
      'Speed (1994)',
      'True Lies (1994)',
      'When a Man Loves a Woman (1994)',
      'Age of Innocence, The (1993)',
      'Black Beauty (1994)',
      'Blue Sky (1994)',
      'Blue Sky (1994)',
      'Dave (1993)',
      'Firm, The (1993)',
      'Fugitive, The (1993)',
      'Hudsucker Proxy, The (1994)',
      'In the Line of Fire (1993)',
      'In the Name of the Father (1993)',
      "What's Love Got to Do with It? (1993)",
      'Jurassic Park (1993)',
      'M. Butterfly (1993)',
      'M. Butterfly (1993)',
      'Much Ado About Nothing (1993)',
      'Mrs. Doubtfire (1993)',
      'Mrs. Doubtfire (1993)',
      'Mrs. Doubtfire (1993)',
      'Philadelphia (1993)',
      'Radioland Murders (1994)',
      'Radioland Murders (1994)',
      'Remains of the Day, The (1993)',
      'Remains of the Day, The (1993)',
      'Renaissance Man (1994)',
      'Romper Stomper (1992)',
      'Romper Stomper (1992)',
      'Romper Stomper (1992)',
      'Rudy (1993)',
      "Schindler's List (1993)",
      'Searching for Bobby Fischer (1993)',
      'Secret Garden, The (1993)',
      'Shadowlands (1993)',
      'Short Cuts (1993)',
      'Six Degrees of Separation (1993)',
      'Sleepless in Seattle (1993)',
      'Blade Runner (1982)',
      'So I Married an Axe Murderer (1993)',
      'Nightmare Before Christmas, The (1993)',
      'Nightmare Before Christmas, The (1993)',
      'War Room, The (1993)',
      'Welcome to the Dollhouse (1995)',
      'Home Alone (1990)',
      'Ghost (1990)',
      'Aladdin (1992)',
      'Terminator 2: Judgment Day (1991)',
      'Dances with Wolves (1990)',
      'Dances with Wolves (1990)',
      'Batman (1989)',
      'Silence of the Lambs, The (1991)',
      'Snow White and the Seven Dwarfs (1937)',
      'Beauty and the Beast (1991)',
      'Pinocchio (1940)',
      'Pretty Woman (1990)',
      'Fargo (1996)',
      'Aristocats, The (1970)',
      'Primal Fear (1996)',
      'Jack and Sarah (1995)',
      'Courage Under Fire (1996)',
      'Mission: Impossible (1996)',
      'Song of the Little Road (Pather Panchali) (1955)',
      'World of Apu, The (Apur Sansar) (1959)',
      'Mystery Science Theater 3000: The Movie (1996)',
      'Space Jam (1996)',
      'Truth About Cats & Dogs, The (1996)',
      'Wallace & Gromit: The Best of Aardman Animation (1996)',
      'Cold Comfort Farm (1995)',
      'Rock, The (1996)',
      'Twister (1996)',
      'Wallace & Gromit: A Close Shave (1995)',
      'Arrival, The (1996)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Trainspotting (1996)',
      'Independence Day (a.k.a. ID4) (1996)',
      'Lone Star (1996)',
      'Time to Kill, A (1996)',
      'Very Brady Sequel, A (1996)',
      'First Wives Club, The (1996)',
      'Ransom (1996)',
      'Emma (1996)',
      'Tin Cup (1996)',
      'Godfather, The (1972)',
      'Twelfth Night (1996)',
      'Philadelphia Story, The (1940)',
      "Singin' in the Rain (1952)",
      'American in Paris, An (1951)',
      "Breakfast at Tiffany's (1961)",
      'Vertigo (1958)',
      'Rear Window (1954)',
      'It Happened One Night (1934)',
      'Gaslight (1944)',
      'Gay Divorcee, The (1934)',
      'North by Northwest (1959)',
      'Apartment, The (1960)',
      'Some Like It Hot (1959)',
      'Charade (1963)',
      'Casablanca (1942)',
      'Maltese Falcon, The (1941)',
      'My Fair Lady (1964)',
      'Sabrina (1954)',
      'Roman Holiday (1953)',
      'Roman Holiday (1953)',
      'Meet Me in St. Louis (1944)',
      'Wizard of Oz, The (1939)',
      'Wizard of Oz, The (1939)',
      'Gone with the Wind (1939)',
      'My Favorite Year (1982)',
      'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)',
      'Citizen Kane (1941)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      'All About Eve (1950)',
      'Women, The (1939)',
      'Rebecca (1940)',
      'Foreign Correspondent (1940)',
      'Foreign Correspondent (1940)',
      'Foreign Correspondent (1940)',
      'Notorious (1946)',
      'Spellbound (1945)',
      'Father of the Bride (1950)',
      'Ninotchka (1939)',
      'Ninotchka (1939)',
      'Gigi (1958)',
      'Adventures of Robin Hood, The (1938)',
      'Mark of Zorro, The (1940)',
      'Ghost and Mrs. Muir, The (1947)',
      'Lost Horizon (1937)',
      'Top Hat (1935)',
      'My Man Godfrey (1936)',
      'My Man Godfrey (1936)',
      'My Man Godfrey (1936)',
      'Giant (1956)',
      'Thin Man, The (1934)',
      'His Girl Friday (1940)',
      'Around the World in 80 Days (1956)',
      "It's a Wonderful Life (1946)",
      'Mr. Smith Goes to Washington (1939)',
      'Bringing Up Baby (1938)',
      'Bringing Up Baby (1938)',
      'Penny Serenade (1941)',
      '39 Steps, The (1935)',
      'Night of the Living Dead (1968)',
      'African Queen, The (1951)',
      'Beat the Devil (1953)',
      'Cat on a Hot Tin Roof (1958)',
      'Meet John Doe (1941)',
      'Farewell to Arms, A (1932)',
      'Fly Away Home (1996)',
      'Michael Collins (1996)',
      'Big Night (1996)',
      'Chamber, The (1996)',
      'Chamber, The (1996)',
      'Love Bug, The (1969)',
      'Love Bug, The (1969)',
      'Parent Trap, The (1961)',
      'Cinderella (1950)',
      'Sword in the Stone, The (1963)',
      'Sword in the Stone, The (1963)',
      'Mary Poppins (1964)',
      'Mary Poppins (1964)',
      'Dumbo (1941)',
      "Pete's Dragon (1977)",
      'Alice in Wonderland (1951)',
      'Fox and the Hound, The (1981)',
      'Sound of Music, The (1965)',
      'Secrets & Lies (1996)',
      'That Thing You Do! (1996)',
      "William Shakespeare's Romeo + Juliet (1996)",
      'Shall We Dance (1937)',
      'Crossfire (1947)',
      'Innocents, The (1961)',
      'Fish Called Wanda, A (1988)',
      "Monty Python's Life of Brian (1979)",
      "Monty Python's Life of Brian (1979)",
      "Monty Python's Life of Brian (1979)",
      'Victor/Victoria (1982)',
      'Candidate, The (1972)',
      'Bonnie and Clyde (1967)',
      'Bonnie and Clyde (1967)',
      'Dirty Dancing (1987)',
      'Dirty Dancing (1987)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Platoon (1986)',
      'Doors, The (1991)',
      'Doors, The (1991)',
      'Doors, The (1991)',
      "Sophie's Choice (1982)",
      'E.T. the Extra-Terrestrial (1982)',
      'Top Gun (1986)',
      'Rebel Without a Cause (1955)',
      'Rebel Without a Cause (1955)',
      'Streetcar Named Desire, A (1951)',
      'On Golden Pond (1981)',
      'Return of the Pink Panther, The (1975)',
      'Private Benjamin (1980)',
      'Monty Python and the Holy Grail (1975)',
      'Monty Python and the Holy Grail (1975)',
      'When We Were Kings (1996)',
      'When We Were Kings (1996)',
      'Wallace & Gromit: The Wrong Trousers (1993)',
      'Bob Roberts (1992)',
      'Enchanted April (1992)',
      'Paths of Glory (1957)',
      'Paths of Glory (1957)',
      'Grifters, The (1990)',
      'English Patient, The (1996)',
      'My Left Foot (1989)',
      'Passion Fish (1992)',
      'Strictly Ballroom (1992)',
      'Strictly Ballroom (1992)',
      'Thin Blue Line, The (1988)',
      'Thin Blue Line, The (1988)',
      "One Flew Over the Cuckoo's Nest (1975)",
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Princess Bride, The (1987)',
      'Princess Bride, The (1987)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
      'Aliens (1986)',
      'Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)',
      '12 Angry Men (1957)',
      'Lawrence of Arabia (1962)',
      'Clockwork Orange, A (1971)',
      'To Kill a Mockingbird (1962)',
      'To Kill a Mockingbird (1962)',
      'Apocalypse Now (1979)',
      "Once Upon a Time in the West (C'era una volta il West) (1968)",
      'Star Wars: Episode VI - Return of the Jedi (1983)',
      'Star Wars: Episode VI - Return of the Jedi (1983)',
      'Star Wars: Episode VI - Return of the Jedi (1983)',
      'Third Man, The (1949)',
      'Third Man, The (1949)',
      'Third Man, The (1949)',
      'Goodfellas (1990)',
      'Alien (1979)',
      'Ran (1985)',
      'Psycho (1960)',
      'Blues Brothers, The (1980)',
      'Godfather: Part II, The (1974)',
      'Full Metal Jacket (1987)',
      'Grand Day Out with Wallace and Gromit, A (1989)',
      'Henry V (1989)',
      'Amadeus (1984)',
      'Amadeus (1984)',
      'Raging Bull (1980)',
      'Annie Hall (1977)',
      'Right Stuff, The (1983)',
      'Right Stuff, The (1983)',
      'Boot, Das (Boat, The) (1981)',
      'Sting, The (1973)',
      'Harold and Maude (1971)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Terminator, The (1984)',
      'Glory (1989)',
      'Rosencrantz and Guildenstern Are Dead (1990)',
      'Manhattan (1979)',
      "Miller's Crossing (1990)",
      'Dead Poets Society (1989)',
      'Graduate, The (1967)',
      'Femme Nikita, La (Nikita) (1990)',
      'Bridge on the River Kwai, The (1957)',
      'Chinatown (1974)',
      'Day the Earth Stood Still, The (1951)',
      'Treasure of the Sierra Madre, The (1948)',
      'Better Off Dead... (1985)',
      'Shining, The (1980)',
      'Stand by Me (1986)',
      'M (1931)',
      'Great Escape, The (1963)',
      'Deer Hunter, The (1978)',
      'Unforgiven (1992)',
      'Manchurian Candidate, The (1962)',
      'Manchurian Candidate, The (1962)',
      'Arsenic and Old Lace (1944)',
      'Back to the Future (1985)',
      'Patton (1970)',
      'Cool Hand Luke (1967)',
      'Cyrano de Bergerac (1990)',
      'Young Frankenstein (1974)',
      'Raise the Red Lantern (Da hong deng long gao gao gua) (1991)',
      'Great Dictator, The (1940)',
      'Fantasia (1940)',
      'High Noon (1952)',
      'Big Sleep, The (1946)',
      'Heathers (1989)',
      'This Is Spinal Tap (1984)',
      'This Is Spinal Tap (1984)',
      'This Is Spinal Tap (1984)',
      'Indiana Jones and the Last Crusade (1989)',
      'Indiana Jones and the Last Crusade (1989)',
      'Being There (1979)',
      'Gandhi (1982)',
      'Room with a View, A (1986)',
      'Killing Fields, The (1984)',
      'Killing Fields, The (1984)',
      'Forbidden Planet (1956)',
      'Forbidden Planet (1956)',
      'Field of Dreams (1989)',
      'Man Who Would Be King, The (1975)',
      'Butch Cassidy and the Sundance Kid (1969)',
      'When Harry Met Sally... (1989)',
      'Birds, The (1963)',
      'Cape Fear (1991)',
      'Cape Fear (1962)',
      'Carrie (1976)',
      'Carrie (1976)',
      'Carrie (1976)',
      'Nosferatu (Nosferatu, eine Symphonie des Grauens) (1922)',
      'Omen, The (1976)',
      'Mirror Has Two Faces, The (1996)',
      'Breaking the Waves (1996)',
      'Star Trek: First Contact (1996)',
      'Sling Blade (1996)',
      'Paradise Lost: The Child Murders at Robin Hood Hills (1996)',
      "Preacher's Wife, The (1996)",
      'Crucible, The (1996)',
      '101 Dalmatians (1996)',
      '101 Dalmatians (1996)',
      'Star Trek VI: The Undiscovered Country (1991)',
      'Star Trek II: The Wrath of Khan (1982)',
      'Star Trek IV: The Voyage Home (1986)',
      'Batman Returns (1992)',
      'Grease (1978)',
      'Grease 2 (1982)',
      'Jaws (1975)',
      'Jaws 3-D (1983)',
      'Jerry Maguire (1996)',
      'Raising Arizona (1987)',
      'Sneakers (1992)',
      'In Love and War (1996)',
      'Scream (1996)',
      'Scream (1996)',
      'Last of the Mohicans, The (1992)',
      'Last of the Mohicans, The (1992)',
      'Hamlet (1996)',
      'Whole Wide World, The (1996)',
      'Evita (1996)',
      'Hearts and Minds (1996)',
      'Hearts and Minds (1996)',
      'Benny & Joon (1993)',
      'Kolya (Kolja) (1996)',
      'Waiting for Guffman (1996)',
      'Donnie Brasco (1997)',
      'Jungle2Jungle (a.k.a. Jungle 2 Jungle) (1997)',
      'Liar Liar (1997)',
      'Anna Karenina (1997)',
      'Grosse Pointe Blank (1997)',
      'Grosse Pointe Blank (1997)',
      "Romy and Michele's High School Reunion (1997)",
      'Shall We Dance? (Shall We Dansu?) (1996)',
      'Brassed Off (1996)',
      'Lost World: Jurassic Park, The (1997)',
      'Ponette (1996)',
      "My Best Friend's Wedding (1997)",
      'Face/Off (1997)',
      'Men in Black (a.k.a. MIB) (1997)',
      'G.I. Jane (1997)',
      'Air Force One (1997)',
      'Hunt for Red October, The (1990)',
      'L.A. Confidential (1997)',
      'Game, The (1997)',
      'Ice Storm, The (1997)',
      'Mrs. Brown (a.k.a. Her Majesty, Mrs. Brown) (1997)',
      'Mrs. Brown (a.k.a. Her Majesty, Mrs. Brown) (1997)',
      "The Devil's Advocate (1997)",
      'Washington Square (1997)',
      'Gattaca (1997)',
      'Stripes (1981)',
      'Rainmaker, The (1997)',
      'Witness (1985)',
      'Sliding Doors (1998)',
      'Truman Show, The (1998)',
      'Wings of the Dove, The (1997)',
      'Midnight in the Garden of Good and Evil (1997)',
      'Scream 2 (1997)',
      'Scream 2 (1997)',
      'Scream 2 (1997)',
      'Sweet Hereafter, The (1997)',
      'Sweet Hereafter, The (1997)',
      'Titanic (1997)',
      'Big Lebowski, The (1998)',
      'Great Expectations (1998)',
      'Dark City (1998)',
      'Wedding Singer, The (1998)',
      'Everest (1998)',
      'Primary Colors (1998)',
      'Spanish Prisoner, The (1997)',
      'Last Days of Disco, The (1998)',
      'Beyond Silence (Jenseits der Stille) (1996)',
      'Children of Heaven, The (Bacheha-Ye Aseman) (1997)',
      'Dream for an Insomniac (1996)',
      'X-Files: Fight the Future, The (1998)',
      'Out of Sight (1998)',
      'Picnic at Hanging Rock (1975)',
      'Armageddon (1998)',
      'Pi (1998)',
      'Mutiny on the Bounty (1935)',
      'Lost Weekend, The (1945)',
      "Gentleman's Agreement (1947)",
      "Gentleman's Agreement (1947)",
      "All the King's Men (1949)",
      "All the King's Men (1949)",
      "All the King's Men (1949)",
      'On the Waterfront (1954)',
      'On the Waterfront (1954)',
      'West Side Story (1961)',
      'In the Heat of the Night (1967)',
      'Oliver! (1968)',
      'Midnight Cowboy (1969)',
      'French Connection, The (1971)',
      'French Connection, The (1971)',
      'Rocky (1976)',
      'Rocky (1976)',
      'Kramer vs. Kramer (1979)',
      'Chariots of Fire (1981)',
      'Chariots of Fire (1981)',
      'Out of Africa (1985)',
      'Out of Africa (1985)',
      'Last Emperor, The (1987)',
      'Rain Man (1988)',
      'Driving Miss Daisy (1989)',
      'Friday the 13th Part 2 (1981)',
      'Friday the 13th Part 3: 3D (1982)',
      'Friday the 13th Part IV: The Final Chapter (1984)',
      'Friday the 13th Part V: A New Beginning (1985)',
      'Friday the 13th Part VI: Jason Lives (1986)',
      'Halloween II (1981)',
      'Halloween III: Season of the Witch (1982)',
      'Poltergeist (1982)',
      'Poltergeist II: The Other Side (1986)',
      'Poltergeist III (1988)',
      'Exorcist, The (1973)',
      'Mask of Zorro, The (1998)',
      'Mask of Zorro, The (1998)',
      'Back to the Future Part II (1989)',
      'Seven Samurai (Shichinin no samurai) (1954)',
      'Dangerous Liaisons (1988)',
      'Last Temptation of Christ, The (1988)',
      'Last Temptation of Christ, The (1988)',
      'Saving Private Ryan (1998)',
      'Honey, I Shrunk the Kids (1989)',
      'Negotiator, The (1998)',
      'Parent Trap, The (1998)',
      'Roger & Me (1989)',
      'Roger & Me (1989)',
      'Purple Rose of Cairo, The (1985)',
      'Out of the Past (1947)',
      'Tender Mercies (1983)',
      'And the Band Played On (1993)',
      'And the Band Played On (1993)',
      'Blue Velvet (1986)',
      'Blue Velvet (1986)',
      'Jungle Book, The (1967)',
      'Lady and the Tramp (1955)',
      '101 Dalmatians (One Hundred and One Dalmatians) (1961)',
      'Something Wicked This Way Comes (1983)',
      'Something Wicked This Way Comes (1983)',
      'Splash (1984)',
      'L.A. Story (1991)',
      'Outsiders, The (1983)',
      'Indiana Jones and the Temple of Doom (1984)',
      'Lord of the Rings, The (1978)',
      'Dead Zone, The (1983)',
      'Cujo (1983)',
      'Children of the Corn (1984)',
      'Secret of NIMH, The (1982)',
      'Pretty in Pink (1986)',
      'Pretty in Pink (1986)',
      "Rosemary's Baby (1968)",
      'Blade (1998)',
      'Next Stop Wonderland (1998)',
      'Next Stop Wonderland (1998)',
      'Next Stop Wonderland (1998)',
      'Strangers on a Train (1951)',
      'Stage Fright (1950)',
      'Untouchables, The (1987)',
      'Untouchables, The (1987)',
      'Lifeboat (1944)',
      'Shadow of a Doubt (1943)',
      'Suspicion (1941)',
      'Lady Vanishes, The (1938)',
      'Broadcast News (1987)',
      'Married to the Mob (1988)',
      'Say Anything... (1989)',
      'Say Anything... (1989)',
      'My Blue Heaven (1990)',
      'Few Good Men, A (1992)',
      'Few Good Men, A (1992)',
      'Edward Scissorhands (1990)',
      'Impostors, The (1998)',
      'Producers, The (1968)',
      'My Cousin Vinny (1992)',
      'Mighty, The (1998)',
      'Children of a Lesser God (1986)',
      'Elephant Man, The (1980)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'American History X (1998)',
      'Siege, The (1998)',
      'Waterboy, The (1998)',
      'Elizabeth (1998)',
      'Nights of Cabiria (Notti di Cabiria, Le) (1957)',
      "Bug's Life, A (1998)",
      'Central Station (Central do Brasil) (1998)',
      'Celebration, The (Festen) (1998)',
      'Little Voice (1998)',
      'Simple Plan, A (1998)',
      'Prince of Egypt, The (1998)',
      'Prince of Egypt, The (1998)',
      'Shakespeare in Love (1998)',
      'Rocky III (1982)',
      'Rocky IV (1985)',
      'Rocky V (1990)',
      'Karate Kid, The (1984)',
      'Karate Kid, Part II, The (1986)',
      'Karate Kid, Part III, The (1989)',
      "Christmas Vacation (National Lampoon's Christmas Vacation) (1989)",
      "You've Got Mail (1998)",
      "You've Got Mail (1998)",
      'Patch Adams (1998)',
      'Hilary and Jackie (1998)',
      'Ruthless People (1986)',
      'Name of the Rose, The (Name der Rose, Der) (1986)',
      'Crocodile Dundee (1986)',
      'Crocodile Dundee II (1988)',
      'Simply Irresistible (1999)',
      'Last Days, The (1998)',
      'Last Days, The (1998)',
      'Office Space (1999)',
      'Office Space (1999)',
      'Other Sister, The (1999)',
      'Christine (1983)',
      'Planet of the Apes (1968)',
      'Haunting, The (1963)',
      'Village of the Damned (1960)',
      'Children of the Damned (1963)',
      'King and I, The (1956)',
      'Matrix, The (1999)',
      '10 Things I Hate About You (1999)',
      'Following (1998)',
      'Never Been Kissed (1999)',
      "Cookie's Fortune (1999)",
      'Pushing Tin (1999)',
      'eXistenZ (1999)',
      'Mildred Pierce (1945)',
      'Star Wars: Episode I - The Phantom Menace (1999)',
      'Star Wars: Episode I - The Phantom Menace (1999)',
      'Superman (1978)',
      'Superman II (1980)',
      'Superman III (1983)',
      'Frankenstein (1931)',
      'Thing from Another World, The (1951)',
      'Invasion of the Body Snatchers (1956)',
      'Buena Vista Social Club (1999)',
      'Run Lola Run (Lola rennt) (1998)',
      'Trekkies (1997)',
      'Arachnophobia (1990)',
      'Blair Witch Project, The (1999)',
      'Blair Witch Project, The (1999)',
      'Eyes Wide Shut (1999)',
      'Ghostbusters (a.k.a. Ghost Busters) (1984)',
      'Ghostbusters II (1989)',
      'Haunting, The (1999)',
      'Mystery Men (1999)',
      'Killing, The (1956)',
      'Spartacus (1960)',
      'Spartacus (1960)',
      'Spartacus (1960)',
      'Mission, The (1986)',
      'Mission, The (1986)',
      'Radio Days (1987)',
      'Radio Days (1987)',
      'Iron Giant, The (1999)',
      'Sixth Sense, The (1999)',
      'Sixth Sense, The (1999)',
      "Cat's Eye (1985)",
      'Airplane! (1980)',
      'Airplane! (1980)',
      'Big (1988)',
      'Oscar and Lucinda (a.k.a. Oscar & Lucinda) (1997)',
      'Pelican Brief, The (1993)',
      'Christmas Story, A (1983)',
      "Astronaut's Wife, The (1999)",
      "Hard Day's Night, A (1964)",
      'Deliverance (1972)',
      'Excalibur (1981)',
      'Excalibur (1981)',
      'Sanjuro (Tsubaki Sanjûrô) (1962)',
      'Risky Business (1983)',
      "Ferris Bueller's Day Off (1986)",
      'Year of Living Dangerously, The (1982)',
      'Year of Living Dangerously, The (1982)',
      'Year of Living Dangerously, The (1982)',
      'Brief Encounter (1946)',
      'Lady Eve, The (1941)',
      'Palm Beach Story, The (1942)',
      'Niagara (1953)',
      'Gilda (1946)',
      'South Pacific (1958)',
      'Indochine (1992)',
      'Sydney (Hard Eight) (1996)',
      'Fight Club (1999)',
      'Straight Story, The (1999)',
      'Straight Story, The (1999)',
      'Bad Seed, The (1956)',
      'Time Bandits (1981)',
      'Who Framed Roger Rabbit? (1988)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Insider, The (1999)',
      'Insider, The (1999)',
      'American Movie (1999)',
      "They Shoot Horses, Don't They? (1969)",
      'Taming of the Shrew, The (1967)',
      'Yojimbo (1961)',
      'Trading Places (1983)',
      'Meatballs (1979)',
      'Dead Again (1991)',
      'Dead Again (1991)',
      'Dogma (1999)',
      'Commitments, The (1991)',
      'Stand and Deliver (1988)',
      'Moonstruck (1987)',
      '42 Up (1998)',
      '42 Up (1998)',
      'Sleepy Hollow (1999)',
      'Scrooged (1988)',
      'Scrooged (1988)',
      'Grapes of Wrath, The (1940)',
      'Grapes of Wrath, The (1940)',
      'Shop Around the Corner, The (1940)',
      'Natural, The (1984)',
      'Natural, The (1984)',
      'Fatal Attraction (1987)',
      'Fisher King, The (1991)',
      'Places in the Heart (1984)',
      'Toy Story 2 (1999)',
      'Flawless (1999)',
      'End of the Affair, The (1999)',
      'Great Santini, The (1979)',
      'Green Mile, The (1999)',
      'Last Picture Show, The (1971)',
      'Magnolia (1999)',
      'Man on the Moon (1999)',
      'Galaxy Quest (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Talented Mr. Ripley, The (1999)',
      'Hurricane, The (1999)',
      'Titus (1999)',
      'Terrorist, The (a.k.a. Malli) (Theeviravaathi) (1998)',
      'Fast Times at Ridgemont High (1982)',
      'Cry in the Dark, A (1988)',
      'Cry in the Dark, A (1988)',
      'Sister Act (1992)',
      'Alive (1993)',
      'Death Becomes Her (1992)',
      'Far and Away (1992)',
      'Howards End (1992)',
      "White Men Can't Jump (1992)",
      'Cutting Edge, The (1992)',
      'Scream 3 (2000)',
      'Scream 3 (2000)',
      'Brandon Teena Story, The (1998)',
      'Circus, The (1928)',
      'Circus, The (1928)',
      'City Lights (1931)',
      'Kid, The (1921)',
      'Wonder Boys (2000)',
      'Splendor in the Grass (1961)',
      'Splendor in the Grass (1961)',
      'For All Mankind (1989)',
      'Born Yesterday (1950)',
      'Hoosiers (a.k.a. Best Shot) (1986)',
      'Taking of Pelham One Two Three, The (1974)',
      'Volunteers (1985)',
      'JFK (1991)',
      'JFK (1991)',
      'Erin Brockovich (2000)',
      'Erin Brockovich (2000)',
      '...And Justice for All (1979)',
      'Animal House (1978)',
      'Animal House (1978)',
      'Do the Right Thing (1989)',
      'Creature Comforts (1989)',
      'Double Indemnity (1944)',
      'Good Earth, The (1937)',
      "Guess Who's Coming to Dinner (1967)",
      "Guess Who's Coming to Dinner (1967)",
      "Guess Who's Coming to Dinner (1967)",
      "Guess Who's Coming to Dinner (1967)",
      'Color of Paradise, The (Rang-e khoda) (1999)',
      'Lord of the Flies (1963)',
      'Modern Times (1936)',
      'Hustler, The (1961)',
      'Inherit the Wind (1960)',
      'Inherit the Wind (1960)',
      'Inherit the Wind (1960)',
      'Place in the Sun, A (1951)',
      'High Fidelity (2000)',
      'Hook (1991)',
      'Midnight Express (1978)',
      'My Life (1993)',
      'Network (1976)',
      'Odd Couple, The (1968)',
      'Return to Me (2000)',
      'Me Myself I (2000)',
      'Parenthood (1989)',
      'Prince of Tides, The (1991)',
      'Keeping the Faith (2000)',
      'Keeping the Faith (2000)',
      'What Ever Happened to Baby Jane? (1962)',
      'Auntie Mame (1958)',
      'Guys and Dolls (1955)',
      'Caddyshack (1980)',
      'Virgin Suicides, The (1999)',
      'Virgin Suicides, The (1999)',
      'Limelight (1952)',
      'Big Kahuna, The (2000)',
      'Big Kahuna, The (2000)',
      "Pee-wee's Big Adventure (1985)",
      'Gold Rush, The (1925)',
      'White Christmas (1954)',
      'Eraserhead (1977)',
      'Blood Simple (1984)',
      'Soapdish (1991)',
      'Hamlet (1990)',
      'Coming Home (1978)',
      'Conversation, The (1974)',
      'Serpico (1973)',
      'X-Men (2000)',
      'Anatomy of a Murder (1959)',
      'Breaker Morant (1980)',
      'Official Story, The (La historia oficial) (1985)',
      'Steel Magnolias (1989)',
      'The Spiral Staircase (1945)',
      'Eyes of Tammy Faye, The (2000)',
      'Eyes of Tammy Faye, The (2000)',
      'Supergirl (1984)',
      'Almost Famous (2000)',
      'Almost Famous (2000)',
      'Almost Famous (2000)',
      'Dancer in the Dark (2000)',
      'Dancer in the Dark (2000)',
      'Best in Show (2000)',
      'Invisible Man, The (1933)',
      'Requiem for a Dream (2000)',
      'Two Family House (2000)',
      'Billy Elliot (2000)',
      'You Can Count on Me (2000)',
      'How the Grinch Stole Christmas (a.k.a. The Grinch) (2000)',
      'Unbreakable (2000)',
      'Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)',
      'Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)',
      'Wall Street (1987)',
      'Punchline (1988)',
      'Finding Forrester (2000)',
      'House of Mirth, The (2000)',
      'Miss Congeniality (2000)',
      'O Brother, Where Art Thou? (2000)',
      'State and Main (2000)',
      'Traffic (2000)',
      'Friendly Persuasion (1956)',
      'With a Friend Like Harry... (Harry, un ami qui vous veut du bien) (2000)',
      'Evil Dead, The (1981)',
      'Glass Menagerie, The (1987)',
      'Hope and Glory (1987)',
      'My Demon Lover (1987)',
      'Widow of St. Pierre, The (Veuve de Saint-Pierre, La) (2000)',
      "Caveman's Valentine, The (2001)",
      "Caveman's Valentine, The (2001)",
      'Series 7: The Contenders (2001)',
      "Long Night's Journey Into Day (2000)",
      'Avalon (1990)',
      "Bishop's Wife, The (1947)",
      'Greatest Story Ever Told, The (1965)',
      'Greatest Story Ever Told, The (1965)',
      'Elmer Gantry (1960)',
      'Elmer Gantry (1960)',
      'Alfie (1966)',
      "I Know Where I'm Going! (1945)",
      "Losin' It (1983)",
      'Manhunter (1986)',
      'Manhunter (1986)',
      'Reversal of Fortune (1990)',
      'Revenge of the Nerds (1984)',
      "River's Edge (1986)",
      'Necessary Roughness (1991)',
      'Memento (2000)',
      'Spy Kids (2001)',
      "Bridget Jones's Diary (2001)",
      'Luzhin Defence, The (2000)',
      'Days of Wine and Roses (1962)',
      'Norma Rae (1979)',
      'Love Story (1970)',
      'Shrek (2001)',
      "Himalaya (Himalaya - l'enfance d'un chef) (1999)",
      'Ice Castles (1978)',
      'Mississippi Burning (1988)',
      'Mississippi Burning (1988)',
      'Throw Momma from the Train (1987)',
      'Divided We Fall (Musíme si pomáhat) (2000)',
      'Gentlemen Prefer Blondes (1953)',
      'Seven Year Itch, The (1955)',
      'Tootsie (1982)',
      'A.I. Artificial Intelligence (2001)',
      'Princess and the Warrior, The (Krieger und die Kaiserin, Der) (2000)',
      'Cannonball Run, The (1981)',
      'Faust (1926)',
      'Legally Blonde (2001)',
      'Score, The (2001)',
      'Accused, The (1988)',
      'Big Business (1988)',
      'Big Top Pee-Wee (1988)',
      'Coming to America (1988)',
      'Coming to America (1988)',
      'Crossing Delancey (1988)',
      'My Stepmother Is an Alien (1988)',
      'Running on Empty (1988)',
      'Short Circuit (1986)',
      'Vanishing, The (Spoorloos) (1988)',
      'Vanishing, The (Spoorloos) (1988)',
      'Twins (1988)',
      "Bill & Ted's Excellent Adventure (1989)",
      'Gross Anatomy (a.k.a. A Cut Above) (1989)',
      "Look Who's Talking (1989)",
      'Major League (1989)',
      'Jurassic Park III (2001)',
      "America's Sweethearts (2001)",
      "America's Sweethearts (2001)",
      'Ghost World (2001)',
      'Turner & Hooch (1989)',
      'UHF (1989)',
      'Black Robe (1991)',
      'Little Man Tate (1991)',
      'Little Foxes, The (1941)',
      'Vanishing, The (1993)',
      'Silkwood (1983)',
      'Silkwood (1983)',
      'SpaceCamp (1986)',
      'SpaceCamp (1986)',
      'Serendipity (2001)',
      "Coal Miner's Daughter (1980)",
      'Fiddler on the Roof (1971)',
      'Fiddler on the Roof (1971)',
      'K-PAX (2001)',
      'K-PAX (2001)',
      'Donnie Darko (2001)',
      "Man Who Wasn't There, The (2001)",
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      "Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)",
      "Devil's Backbone, The (Espinazo del diablo, El) (2001)",
      'Now, Voyager (1942)',
      "Ocean's Eleven (2001)",
      'And Then There Were None (1945)',
      'Blue Angel, The (Blaue Engel, Der) (1930)',
      'Blue Angel, The (Blaue Engel, Der) (1930)',
      "Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)",
      'Royal Tenenbaums, The (2001)',
      "Bill & Ted's Bogus Journey (1991)",
      'Lord of the Rings: The Fellowship of the Ring, The (2001)',
      'Defiant Ones, The (1958)',
      'Witness for the Prosecution (1957)',
      'Witness for the Prosecution (1957)',
      'Yentl (1983)',
      'Yentl (1983)',
      "Monster's Ball (2001)",
      'Truly, Madly, Deeply (1991)',
      'The Count of Monte Cristo (2002)',
      'The Count of Monte Cristo (2002)',
      'Bad and the Beautiful, The (1952)',
      'Sleuth (1972)',
      'Monsoon Wedding (2001)',
      'Bad News Bears, The (1976)',
      ...],
     ['Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Twelve Monkeys (a.k.a. 12 Monkeys) (1995)',
      'Babe (1995)',
      'Happy Gilmore (1996)',
      'Billy Madison (1995)',
      'Billy Madison (1995)',
      'Shawshank Redemption, The (1994)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Terminator 2: Judgment Day (1991)',
      'Silence of the Lambs, The (1991)',
      'Silence of the Lambs, The (1991)',
      'Silence of the Lambs, The (1991)',
      'Silence of the Lambs, The (1991)',
      'Silence of the Lambs, The (1991)',
      'Rock, The (1996)',
      'Rock, The (1996)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Ransom (1996)',
      'Ransom (1996)',
      'Ransom (1996)',
      'Vertigo (1958)',
      'Vertigo (1958)',
      'Vertigo (1958)',
      'Vertigo (1958)',
      'Rear Window (1954)',
      'Rear Window (1954)',
      'Rear Window (1954)',
      'Rear Window (1954)',
      'Rear Window (1954)',
      'North by Northwest (1959)',
      'North by Northwest (1959)',
      'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)',
      'Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Reservoir Dogs (1992)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Star Wars: Episode V - The Empire Strikes Back (1980)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Psycho (1960)',
      'Full Metal Jacket (1987)',
      'Akira (1988)',
      'Akira (1988)',
      'Akira (1988)',
      'Star Trek: Insurrection (1998)',
      'Star Trek: Insurrection (1998)',
      'Star Trek: Insurrection (1998)',
      'Big Daddy (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'American Pie (1999)',
      'Iron Giant, The (1999)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Princess Mononoke (Mononoke-hime) (1997)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Predator (1987)',
      'Unbreakable (2000)',
      'Unbreakable (2000)',
      'Unbreakable (2000)',
      'Unbreakable (2000)',
      'Unbreakable (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'More (1998)',
      'More (1998)',
      'More (1998)',
      'More (1998)',
      'More (1998)',
      'More (1998)',
      'More (1998)',
      'Session 9 (2001)',
      'Session 9 (2001)',
      'Session 9 (2001)',
      'Session 9 (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)',
      'Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)',
      'Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Punisher, The (2004)',
      'Village, The (2004)',
      'Village, The (2004)',
      'Village, The (2004)',
      'Incredibles, The (2004)',
      'Incredibles, The (2004)',
      'King Kong (2005)',
      'Voices of a Distant Star (Hoshi no koe) (2003)',
      'X-Men: The Last Stand (2006)',
      'X-Men: The Last Stand (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Lady in the Water (2006)',
      'Illusionist, The (2006)',
      'Illusionist, The (2006)',
      'Illusionist, The (2006)',
      'Illusionist, The (2006)',
      'Paprika (Papurika) (2006)',
      'Fido (2006)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Superbad (2007)',
      'Tekkonkinkreet (Tekkon kinkurîto) (2006)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'I Am Legend (2007)',
      'Juno (2007)',
      'Juno (2007)',
      'Juno (2007)',
      'Juno (2007)',
      'Juno (2007)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'In Bruges (2008)',
      'Son of Rambow (2007)',
      'WALL·E (2008)',
      'WALL·E (2008)',
      'WALL·E (2008)',
      'WALL·E (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'Burn After Reading (2008)',
      'FLCL (2000)',
      'FLCL (2000)',
      'FLCL (2000)',
      'FLCL (2000)',
      "Dr. Horrible's Sing-Along Blog (2008)",
      "Dr. Horrible's Sing-Along Blog (2008)",
      "Dr. Horrible's Sing-Along Blog (2008)",
      "Dr. Horrible's Sing-Along Blog (2008)",
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'I Love You, Man (2009)',
      'Moon (2009)',
      'Moon (2009)',
      'Moon (2009)',
      'X-Men Origins: Wolverine (2009)',
      'X-Men Origins: Wolverine (2009)',
      'X-Men Origins: Wolverine (2009)',
      'X-Men Origins: Wolverine (2009)',
      'X-Men Origins: Wolverine (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Star Trek (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Terminator Salvation (2009)',
      'Neon Genesis Evangelion: Death & Rebirth (Shin seiki Evangelion Gekijô-ban: Shito shinsei) (1997)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      'Transformers: Revenge of the Fallen (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      '(500) Days of Summer (2009)',
      'District 9 (2009)',
      'Gentlemen Broncos (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Scott Pilgrim vs. the World (2010)',
      'Scott Pilgrim vs. the World (2010)',
      'Scott Pilgrim vs. the World (2010)',
      'Scott Pilgrim vs. the World (2010)',
      'Rare Exports: A Christmas Tale (Rare Exports) (2010)',
      'Tron: Legacy (2010)',
      'Tron: Legacy (2010)',
      'Tron: Legacy (2010)',
      'Tron: Legacy (2010)',
      'Tron: Legacy (2010)'],
     ['Whiplash (2014)', 'Whiplash (2014)', 'Whiplash (2014)'],
     ['Whiplash (2014)', 'Whiplash (2014)', 'Whiplash (2014)'],
     ['Sintel (2010)', 'Sintel (2010)', 'Sintel (2010)'],
     ['Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      "All the President's Men (1976)"],
     ['Trading Places (1983)', 'Meet Dave (2008)'],
     ['Forrest Gump (1994)', 'Forrest Gump (1994)', 'Forrest Gump (1994)'],
     ["Schindler's List (1993)",
      "Schindler's List (1993)",
      "Schindler's List (1993)",
      "Schindler's List (1993)",
      'Game, The (1997)',
      'Titanic (1997)',
      'American History X (1998)',
      'American History X (1998)',
      'American History X (1998)',
      'Matrix, The (1999)',
      'Matrix, The (1999)',
      'Requiem for a Dream (2000)',
      'Requiem for a Dream (2000)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Catch Me If You Can (2002)',
      'Notebook, The (2004)',
      'Notebook, The (2004)',
      'Notebook, The (2004)',
      'American Gangster (2007)',
      'American Gangster (2007)',
      'American Gangster (2007)',
      'American Gangster (2007)',
      'Bank Job, The (2008)',
      'Bank Job, The (2008)',
      'Bank Job, The (2008)',
      'Hangover, The (2009)',
      'Hangover, The (2009)',
      'Hangover, The (2009)',
      'Hangover, The (2009)',
      'Hangover, The (2009)',
      'Hurt Locker, The (2008)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Avatar (2009)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Town, The (2010)',
      'Town, The (2010)',
      'Town, The (2010)',
      'Town, The (2010)',
      'Inside Job (2010)',
      'Inside Job (2010)',
      'Inside Job (2010)',
      'Inside Job (2010)',
      'Inside Job (2010)',
      'Inside Job (2010)',
      'Avengers, The (2012)',
      'Avengers, The (2012)',
      'Avengers, The (2012)',
      'Avengers, The (2012)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Margin Call (2011)',
      'Intouchables (2011)',
      'Intouchables (2011)',
      'Thousand Words, A (2012)',
      'Ted (2012)',
      'Argo (2012)',
      'Argo (2012)',
      'Argo (2012)',
      'Django Unchained (2012)',
      'Grown Ups 2 (2013)',
      'Gravity (2013)',
      'Gravity (2013)',
      'Gravity (2013)',
      'Gravity (2013)',
      'Gravity (2013)',
      'Gravity (2013)',
      'Captain Phillips (2013)',
      'Captain Phillips (2013)',
      'Captain Phillips (2013)',
      'Wolf of Wall Street, The (2013)',
      'Wolf of Wall Street, The (2013)'],
     ['Chalet Girl (2011)'],
     ['Toy Story (1995)',
      'Bottle Rocket (1996)',
      'Bottle Rocket (1996)',
      'Three Colors: White (Trzy kolory: Bialy) (1994)',
      'Forrest Gump (1994)',
      'Forrest Gump (1994)',
      'Forrest Gump (1994)',
      'Forrest Gump (1994)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Blade Runner (1982)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      '12 Angry Men (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'Seventh Seal, The (Sjunde inseglet, Det) (1957)',
      'M (1931)',
      'M (1931)',
      'M (1931)',
      'M (1931)',
      'M (1931)',
      'M (1931)',
      'Gattaca (1997)',
      'Gattaca (1997)',
      'Good Will Hunting (1997)',
      'Good Will Hunting (1997)',
      "Buffalo '66 (a.k.a. Buffalo 66) (1998)",
      "Buffalo '66 (a.k.a. Buffalo 66) (1998)",
      "Buffalo '66 (a.k.a. Buffalo 66) (1998)",
      "Buffalo '66 (a.k.a. Buffalo 66) (1998)",
      "Buffalo '66 (a.k.a. Buffalo 66) (1998)",
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Pi (1998)',
      'Watership Down (1978)',
      'Watership Down (1978)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      'Life Is Beautiful (La Vita è bella) (1997)',
      '400 Blows, The (Les quatre cents coups) (1959)',
      '400 Blows, The (Les quatre cents coups) (1959)',
      '400 Blows, The (Les quatre cents coups) (1959)',
      '400 Blows, The (Les quatre cents coups) (1959)',
      '400 Blows, The (Les quatre cents coups) (1959)',
      'Grand Illusion (La grande illusion) (1937)',
      'Grand Illusion (La grande illusion) (1937)',
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      "Man Bites Dog (C'est arrivé près de chez vous) (1992)",
      'Do the Right Thing (1989)',
      'Blazing Saddles (1974)',
      'Blazing Saddles (1974)',
      'Blazing Saddles (1974)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Eraserhead (1977)',
      'Unbreakable (2000)',
      'Unbreakable (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'In the Mood For Love (Fa yeung nin wa) (2000)',
      'Memento (2000)',
      'Memento (2000)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Tetsuo, the Ironman (Tetsuo) (1988)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Donnie Darko (2001)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Punch-Drunk Love (2002)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Grave of the Fireflies (Hotaru no haka) (1988)',
      'Irreversible (Irréversible) (2002)',
      'Irreversible (Irréversible) (2002)',
      'Lilya 4-Ever (Lilja 4-ever) (2002)',
      'Lilya 4-Ever (Lilja 4-ever) (2002)',
      'Lilya 4-Ever (Lilja 4-ever) (2002)',
      'Finding Nemo (2003)',
      'Ikiru (1952)',
      'Ikiru (1952)',
      'Ikiru (1952)',
      'Ikiru (1952)',
      'Come and See (Idi i smotri) (1985)',
      'Come and See (Idi i smotri) (1985)',
      'Come and See (Idi i smotri) (1985)',
      'Come and See (Idi i smotri) (1985)',
      'Come and See (Idi i smotri) (1985)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Salo, or The 120 Days of Sodom (Salò o le 120 giornate di Sodoma) (1976)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Eternal Sunshine of the Spotless Mind (2004)',
      'Make Way for Tomorrow (1937)',
      'Jetée, La (1962)',
      'Andalusian Dog, An (Chien andalou, Un) (1929)',
      'Andalusian Dog, An (Chien andalou, Un) (1929)',
      'Andalusian Dog, An (Chien andalou, Un) (1929)',
      'Begotten (1990)',
      'Begotten (1990)',
      'Begotten (1990)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Old Boy (2003)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Life Aquatic with Steve Zissou, The (2004)',
      'Match Factory Girl, The (Tulitikkutehtaan tyttö) (1990)',
      'Marie Antoinette (2006)',
      'Marie Antoinette (2006)',
      "Pan's Labyrinth (Laberinto del fauno, El) (2006)",
      "Pan's Labyrinth (Laberinto del fauno, El) (2006)",
      "Pan's Labyrinth (Laberinto del fauno, El) (2006)",
      'Prestige, The (2006)',
      'Prestige, The (2006)',
      'Ratatouille (2007)',
      'Ratatouille (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'There Will Be Blood (2007)',
      'Cat Soup (Nekojiru-so) (2001)',
      'Cat Soup (Nekojiru-so) (2001)',
      'Cat Soup (Nekojiru-so) (2001)',
      'Cat Soup (Nekojiru-so) (2001)',
      'Funny Games U.S. (2007)',
      'Funny Games U.S. (2007)',
      'Funny Games U.S. (2007)',
      'Funny Games U.S. (2007)',
      'Dark Knight, The (2008)',
      'Dark Knight, The (2008)',
      'Changeling (2008)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Up (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Mary and Max (2009)',
      'Town Called Panic, A (Panique au village) (2009)',
      'Town Called Panic, A (Panique au village) (2009)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Inception (2010)',
      'Social Network, The (2010)',
      'Social Network, The (2010)',
      'Social Network, The (2010)',
      'Avengers, The (2012)',
      'Avengers, The (2012)',
      'Avengers, The (2012)',
      'Moonrise Kingdom (2012)',
      'Beasts of the Southern Wild (2012)',
      'Beasts of the Southern Wild (2012)',
      'Beasts of the Southern Wild (2012)',
      'Skyfall (2012)',
      'Skyfall (2012)',
      'Skyfall (2012)',
      "It's Such a Beautiful Day (2012)",
      "It's Such a Beautiful Day (2012)",
      "It's Such a Beautiful Day (2012)",
      "It's Such a Beautiful Day (2012)",
      'Upstream Color (2013)',
      'Upstream Color (2013)',
      'Upstream Color (2013)',
      'Upstream Color (2013)',
      'Upstream Color (2013)',
      'Upstream Color (2013)',
      'History of Future Folk, The (2012)',
      'Short Term 12 (2013)',
      'Short Term 12 (2013)',
      'Short Term 12 (2013)',
      'Short Term 12 (2013)',
      'Short Term 12 (2013)',
      'Captain Phillips (2013)',
      'Captain Phillips (2013)',
      'Inside Llewyn Davis (2013)',
      'Inside Llewyn Davis (2013)',
      'Inside Llewyn Davis (2013)',
      'Inside Llewyn Davis (2013)',
      'Snowpiercer (2013)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'The Lego Movie (2014)',
      'Interstellar (2014)',
      'Interstellar (2014)',
      'Interstellar (2014)',
      'Interstellar (2014)',
      'Frank (2014)',
      'Frank (2014)',
      'Frank (2014)',
      'Frank (2014)',
      'Frank (2014)',
      'Frank (2014)',
      'Whiplash (2014)',
      'Whiplash (2014)',
      'Whiplash (2014)',
      'Guardians of the Galaxy (2014)',
      'Guardians of the Galaxy (2014)',
      'Guardians of the Galaxy (2014)',
      'Guardians of the Galaxy (2014)',
      'Two Days, One Night (Deux jours, une nuit) (2014)',
      "Angel's Egg (Tenshi no tamago) (1985)",
      "Angel's Egg (Tenshi no tamago) (1985)",
      "Angel's Egg (Tenshi no tamago) (1985)",
      "Angel's Egg (Tenshi no tamago) (1985)",
      'The Imitation Game (2014)',
      'The Imitation Game (2014)',
      "The Rabbi's Cat (Le chat du rabbin) (2011)",
      "The Rabbi's Cat (Le chat du rabbin) (2011)",
      'Paddington (2014)',
      'Kingsman: The Secret Service (2015)',
      'Mad Max: Fury Road (2015)',
      'Mad Max: Fury Road (2015)',
      'Mad Max: Fury Road (2015)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Avengers: Infinity War - Part I (2018)',
      'Thor: Ragnarok (2017)',
      'Thor: Ragnarok (2017)',
      'Thor: Ragnarok (2017)',
      'Thor: Ragnarok (2017)',
      'Thor: Ragnarok (2017)',
      'Guardians of the Galaxy 2 (2017)',
      'Doctor Strange (2016)',
      'A Pigeon Sat on a Branch Reflecting on Existence (2014)',
      'A Pigeon Sat on a Branch Reflecting on Existence (2014)',
      'The Martian (2015)',
      'Kung Fury (2015)',
      'Kung Fury (2015)',
      'Kung Fury (2015)',
      'The Man from U.N.C.L.E. (2015)',
      'The Man from U.N.C.L.E. (2015)',
      'The Revenant (2015)',
      'The Revenant (2015)',
      'The Revenant (2015)',
      'Sicario (2015)',
      'Sicario (2015)',
      'Room (2015)',
      'Room (2015)',
      'Beasts of No Nation (2015)',
      'Silence (2016)',
      'Silence (2016)',
      'Silence (2016)',
      'Silence (2016)',
      'Big Short, The (2015)',
      'Big Short, The (2015)',
      'Big Short, The (2015)',
      'World of Tomorrow (2015)',
      'World of Tomorrow (2015)',
      '10 Cloverfield Lane (2016)',
      '10 Cloverfield Lane (2016)',
      'Rabbits (2002)',
      'Eye in the Sky (2016)',
      'Eye in the Sky (2016)',
      'Paterson',
      'Paterson',
      'Paterson',
      "Don't Breathe (2016)",
      'Arrival (2016)',
      'Arrival (2016)',
      'Arrival (2016)',
      'Arrival (2016)',
      'Arrival (2016)',
      'Arrival (2016)',
      'Arrival (2016)',
      'La La Land (2016)',
      'La La Land (2016)',
      'The Lego Batman Movie (2017)',
      'The Lego Batman Movie (2017)',
      'Logan (2017)',
      'Logan (2017)',
      'Logan (2017)',
      'Logan (2017)',
      'Logan (2017)',
      'It Comes at Night (2017)',
      'It Comes at Night (2017)',
      'Dunkirk (2017)',
      'Dunkirk (2017)',
      'Dunkirk (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Blade Runner 2049 (2017)',
      'Mother! (2017)',
      'Mother! (2017)',
      'Mother! (2017)',
      'The Shape of Water (2017)',
      'The Shape of Water (2017)',
      'The Greatest Showman (2017)'],
     ['Houseguest (1994)',
      'Houseguest (1994)',
      'Airheads (1994)',
      'Sliver (1993)',
      'Mulholland Falls (1996)',
      'Chain Reaction (1996)',
      'Benny & Joon (1993)',
      'Sphere (1998)',
      'Man in the Iron Mask, The (1998)',
      'Lord of the Rings, The (1978)',
      'Patch Adams (1998)',
      'War of the Worlds, The (1953)',
      'Fistful of Dollars, A (Per un pugno di dollari) (1964)',
      "Dude, Where's My Car? (2000)",
      'Evolution (2001)',
      'Evolution (2001)',
      'Final Fantasy: The Spirits Within (2001)',
      'Score, The (2001)',
      'The Count of Monte Cristo (2002)',
      'Blade II (2002)',
      'Blade II (2002)',
      'City of God (Cidade de Deus) (2002)',
      'Daredevil (2003)',
      'Daredevil (2003)',
      'Bruce Almighty (2003)',
      '40-Year-Old Virgin, The (2005)',
      '40-Year-Old Virgin, The (2005)',
      '40-Year-Old Virgin, The (2005)',
      '40-Year-Old Virgin, The (2005)',
      'Invisible, The (2007)',
      'Invisible, The (2007)'],
     ['Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Léon: The Professional (a.k.a. The Professional) (Léon) (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      'Pulp Fiction (1994)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      '2001: A Space Odyssey (1968)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Big Lebowski, The (1998)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)',
      'Fight Club (1999)'],
     ["Mary Shelley's Frankenstein (Frankenstein) (1994)"],
     ['Shine (1996)',
      'Tom Jones (1963)',
      'Gladiator (2000)',
      'Staying Alive (1983)',
      'Night of the Shooting Stars (Notte di San Lorenzo, La) (1982)',
      "I'm Not Scared (Io non ho paura) (2003)",
      'Shame (Skammen) (1968)'],
     ['Hard-Boiled (Lat sau san taam) (1992)',
      'Hard-Boiled (Lat sau san taam) (1992)',
      'John Wick: Chapter Two (2017)']]




```python
len(master.userId.unique())
```




    58




```python
len(merge_list)
```




    58




```python
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(merge_list).transform(merge_list)
#Via the fit method, the TransactionEncoder learns the unique labels in the dataset, 
#and via the transform method, it transforms the input dataset (a Python list of lists) into 
#a one-hot encoded NumPy boolean array:

df = pd.DataFrame(te_ary, columns=te.columns_)
```


```python
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
      <th>(500) Days of Summer (2009)</th>
      <th>...And Justice for All (1979)</th>
      <th>10 Cloverfield Lane (2016)</th>
      <th>10 Things I Hate About You (1999)</th>
      <th>101 Dalmatians (1996)</th>
      <th>101 Dalmatians (One Hundred and One Dalmatians) (1961)</th>
      <th>11'09"01 - September 11 (2002)</th>
      <th>12 Angry Men (1957)</th>
      <th>127 Hours (2010)</th>
      <th>13 Going on 30 (2004)</th>
      <th>2001: A Space Odyssey (1968)</th>
      <th>21 Grams (2003)</th>
      <th>25th Hour (2002)</th>
      <th>28 Days Later (2002)</th>
      <th>39 Steps, The (1935)</th>
      <th>3:10 to Yuma (2007)</th>
      <th>40-Year-Old Virgin, The (2005)</th>
      <th>400 Blows, The (Les quatre cents coups) (1959)</th>
      <th>42 Up (1998)</th>
      <th>84 Charing Cross Road (1987)</th>
      <th>8MM (1999)</th>
      <th>A Million Ways to Die in the West (2014)</th>
      <th>A Pigeon Sat on a Branch Reflecting on Existence (2014)</th>
      <th>A Story of Children and Film (2013)</th>
      <th>A.I. Artificial Intelligence (2001)</th>
      <th>About a Boy (2002)</th>
      <th>Accused, The (1988)</th>
      <th>Adam's Rib (1949)</th>
      <th>Addams Family Values (1993)</th>
      <th>Addams Family, The (1991)</th>
      <th>Adventures of Priscilla, Queen of the Desert, The (1994)</th>
      <th>Adventures of Robin Hood, The (1938)</th>
      <th>African Queen, The (1951)</th>
      <th>After the Thin Man (1936)</th>
      <th>Age of Innocence, The (1993)</th>
      <th>Air Force One (1997)</th>
      <th>Airheads (1994)</th>
      <th>Airplane! (1980)</th>
      <th>Akira (1988)</th>
      <th>Aladdin (1992)</th>
      <th>Alfie (1966)</th>
      <th>Alice Adams (1935)</th>
      <th>Alice Doesn't Live Here Anymore (1974)</th>
      <th>Alice in Wonderland (1951)</th>
      <th>Alien (1979)</th>
      <th>Aliens (1986)</th>
      <th>Alive (1993)</th>
      <th>All About Eve (1950)</th>
      <th>All the King's Men (1949)</th>
      <th>All the President's Men (1976)</th>
      <th>All the Real Girls (2003)</th>
      <th>Almost Famous (2000)</th>
      <th>Amadeus (1984)</th>
      <th>Amelie (Fabuleux destin d'Amélie Poulain, Le) (2001)</th>
      <th>Amen. (2002)</th>
      <th>America's Sweethearts (2001)</th>
      <th>American Gangster (2007)</th>
      <th>American History X (1998)</th>
      <th>American Movie (1999)</th>
      <th>American Pie (1999)</th>
      <th>American President, The (1995)</th>
      <th>American Splendor (2003)</th>
      <th>American in Paris, An (1951)</th>
      <th>Americanization of Emily, The (1964)</th>
      <th>Anastasia (1956)</th>
      <th>Anatomy of a Murder (1959)</th>
      <th>Anchorman 2: The Legend Continues (2013)</th>
      <th>Anchorman: The Legend of Ron Burgundy (2004)</th>
      <th>And Then There Were None (1945)</th>
      <th>And the Band Played On (1993)</th>
      <th>Andalusian Dog, An (Chien andalou, Un) (1929)</th>
      <th>Angel's Egg (Tenshi no tamago) (1985)</th>
      <th>Angie (1994)</th>
      <th>Animal House (1978)</th>
      <th>Animatrix, The (2003)</th>
      <th>Anna Karenina (1997)</th>
      <th>Anne Frank Remembered (1995)</th>
      <th>Anne of the Thousand Days (1969)</th>
      <th>Annie Hall (1977)</th>
      <th>Another Thin Man (1939)</th>
      <th>Apartment, The (1960)</th>
      <th>Apocalypse Now (1979)</th>
      <th>Apollo 13 (1995)</th>
      <th>Arachnophobia (1990)</th>
      <th>Argo (2012)</th>
      <th>Aristocats, The (1970)</th>
      <th>Armageddon (1998)</th>
      <th>Around the World in 80 Days (1956)</th>
      <th>Arrival (2016)</th>
      <th>Arrival, The (1996)</th>
      <th>Arsenic and Old Lace (1944)</th>
      <th>Astronaut's Wife, The (1999)</th>
      <th>Au Hasard Balthazar (1966)</th>
      <th>Auntie Mame (1958)</th>
      <th>Avalon (1990)</th>
      <th>Avatar (2009)</th>
      <th>Avengers, The (2012)</th>
      <th>Avengers: Infinity War - Part I (2018)</th>
      <th>Aviator, The (2004)</th>
      <th>Awful Truth, The (1937)</th>
      <th>Babadook, The (2014)</th>
      <th>Babe (1995)</th>
      <th>Babel (2006)</th>
      <th>Babette's Feast (Babettes gæstebud) (1987)</th>
      <th>Babylon 5: In the Beginning (1998)</th>
      <th>Bachelor and the Bobby-Soxer, The (1947)</th>
      <th>Back to the Future (1985)</th>
      <th>Back to the Future Part II (1989)</th>
      <th>Bad Day at Black Rock (1955)</th>
      <th>Bad News Bears, The (1976)</th>
      <th>Bad Seed, The (1956)</th>
      <th>Bad and the Beautiful, The (1952)</th>
      <th>Ballad of Jack and Rose, The (2005)</th>
      <th>Bank Job, The (2008)</th>
      <th>Barton Fink (1991)</th>
      <th>Basketball Diaries, The (1995)</th>
      <th>Batman (1989)</th>
      <th>Batman Forever (1995)</th>
      <th>Batman Returns (1992)</th>
      <th>Batman v Superman: Dawn of Justice (2016)</th>
      <th>Battle Royale (Batoru rowaiaru) (2000)</th>
      <th>Battle of Algiers, The (La battaglia di Algeri) (1966)</th>
      <th>Beasts of No Nation (2015)</th>
      <th>Beasts of the Southern Wild (2012)</th>
      <th>Beat the Devil (1953)</th>
      <th>Beautiful Mind, A (2001)</th>
      <th>Beauty and the Beast (1991)</th>
      <th>Before Sunrise (1995)</th>
      <th>Before Sunset (2004)</th>
      <th>Begotten (1990)</th>
      <th>Being Julia (2004)</th>
      <th>Being There (1979)</th>
      <th>Believer, The (2001)</th>
      <th>Bend It Like Beckham (2002)</th>
      <th>Benny &amp; Joon (1993)</th>
      <th>Best in Show (2000)</th>
      <th>Better Luck Tomorrow (2002)</th>
      <th>Better Off Dead... (1985)</th>
      <th>Beyond Silence (Jenseits der Stille) (1996)</th>
      <th>Big (1988)</th>
      <th>Big Business (1988)</th>
      <th>Big Daddy (1999)</th>
      <th>Big Eyes (2014)</th>
      <th>Big Fish (2003)</th>
      <th>Big Hero 6 (2014)</th>
      <th>Big Kahuna, The (2000)</th>
      <th>Big Lebowski, The (1998)</th>
      <th>Big Night (1996)</th>
      <th>Big Short, The (2015)</th>
      <th>Big Sleep, The (1946)</th>
      <th>Big Top Pee-Wee (1988)</th>
      <th>Bill &amp; Ted's Bogus Journey (1991)</th>
      <th>Bill &amp; Ted's Excellent Adventure (1989)</th>
      <th>Bill Cosby, Himself (1983)</th>
      <th>Billabong Odyssey (2003)</th>
      <th>Billy Elliot (2000)</th>
      <th>Billy Madison (1995)</th>
      <th>Birdman of Alcatraz (1962)</th>
      <th>Birds, The (1963)</th>
      <th>Bishop's Wife, The (1947)</th>
      <th>Black Beauty (1994)</th>
      <th>Black Mirror: White Christmas (2014)</th>
      <th>Black Narcissus (1947)</th>
      <th>Black Orpheus (Orfeu Negro) (1959)</th>
      <th>Black Robe (1991)</th>
      <th>Black Stallion, The (1979)</th>
      <th>Black Swan (2010)</th>
      <th>Blade (1998)</th>
      <th>Blade II (2002)</th>
      <th>Blade Runner (1982)</th>
      <th>Blade Runner 2049 (2017)</th>
      <th>Blair Witch Project, The (1999)</th>
      <th>Blazing Saddles (1974)</th>
      <th>Blood Diamond (2006)</th>
      <th>Blood Simple (1984)</th>
      <th>Blue Angel, The (Blaue Engel, Der) (1930)</th>
      <th>Blue Car (2002)</th>
      <th>Blue Sky (1994)</th>
      <th>Blue Velvet (1986)</th>
      <th>Blues Brothers, The (1980)</th>
      <th>Bob Roberts (1992)</th>
      <th>Bonnie and Clyde (1967)</th>
      <th>Boomerang (1992)</th>
      <th>Boot, Das (Boat, The) (1981)</th>
      <th>Born Free (1966)</th>
      <th>Born Yesterday (1950)</th>
      <th>Born into Brothels (2004)</th>
      <th>Bottle Rocket (1996)</th>
      <th>Bourne Ultimatum, The (2007)</th>
      <th>Bowling for Columbine (2002)</th>
      <th>Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008)</th>
      <th>Brandon Teena Story, The (1998)</th>
      <th>Brassed Off (1996)</th>
      <th>Braveheart (1995)</th>
      <th>Breaker Morant (1980)</th>
      <th>Breakfast at Tiffany's (1961)</th>
      <th>Breakfast on Pluto (2005)</th>
      <th>Breaking the Waves (1996)</th>
      <th>Bridge on the River Kwai, The (1957)</th>
      <th>Bridget Jones's Diary (2001)</th>
      <th>Brief Encounter (1946)</th>
      <th>Bringing Up Baby (1938)</th>
      <th>Broadcast News (1987)</th>
      <th>Broadway Danny Rose (1984)</th>
      <th>Broken Flowers (2005)</th>
      <th>Brothers Bloom, The (2008)</th>
      <th>Browning Version, The (1951)</th>
      <th>Bruce Almighty (2003)</th>
      <th>Buena Vista Social Club (1999)</th>
      <th>Buffalo '66 (a.k.a. Buffalo 66) (1998)</th>
      <th>Bug (2007)</th>
      <th>Bug's Life, A (1998)</th>
      <th>Burn After Reading (2008)</th>
      <th>Butch Cassidy and the Sundance Kid (1969)</th>
      <th>Butterflies Are Free (1972)</th>
      <th>Caddyshack (1980)</th>
      <th>Caine Mutiny, The (1954)</th>
      <th>Call Northside 777 (1948)</th>
      <th>Camelot (1967)</th>
      <th>Candidate, The (1972)</th>
      <th>Cannonball Run, The (1981)</th>
      <th>Cape Fear (1962)</th>
      <th>Cape Fear (1991)</th>
      <th>Capote (2005)</th>
      <th>Captain Blood (1935)</th>
      <th>Captain Fantastic (2016)</th>
      <th>Captain Phillips (2013)</th>
      <th>Capturing the Friedmans (2003)</th>
      <th>Carlito's Way (1993)</th>
      <th>Carrie (1976)</th>
      <th>Casablanca (1942)</th>
      <th>Casino (1995)</th>
      <th>Cat People (1942)</th>
      <th>Cat Returns, The (Neko no ongaeshi) (2002)</th>
      <th>Cat Soup (Nekojiru-so) (2001)</th>
      <th>Cat on a Hot Tin Roof (1958)</th>
      <th>Cat's Eye (1985)</th>
      <th>Catch Me If You Can (2002)</th>
      <th>Caveman's Valentine, The (2001)</th>
      <th>Celebration, The (Festen) (1998)</th>
      <th>Central Station (Central do Brasil) (1998)</th>
      <th>Chain Reaction (1996)</th>
      <th>Chalet Girl (2011)</th>
      <th>Chamber, The (1996)</th>
      <th>Changeling (2008)</th>
      <th>Chaplin (1992)</th>
      <th>Charade (1963)</th>
      <th>Chariots of Fire (1981)</th>
      <th>Charlie and the Chocolate Factory (2005)</th>
      <th>Chicago (2002)</th>
      <th>...</th>
      <th>Staying Alive (1983)</th>
      <th>Steel Magnolias (1989)</th>
      <th>Step Brothers (2008)</th>
      <th>Stevie (2002)</th>
      <th>Sting, The (1973)</th>
      <th>Stone Reader (2002)</th>
      <th>Story of the Weeping Camel, The (Geschichte vom weinenden Kamel, Die) (2003)</th>
      <th>Strada, La (1954)</th>
      <th>Straight Story, The (1999)</th>
      <th>Stranger than Fiction (2006)</th>
      <th>Strangers on a Train (1951)</th>
      <th>Stray Dog (Nora inu) (1949)</th>
      <th>Streetcar Named Desire, A (1951)</th>
      <th>Strictly Ballroom (1992)</th>
      <th>Stripes (1981)</th>
      <th>Suicide Squad (2016)</th>
      <th>Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)</th>
      <th>Super Size Me (2004)</th>
      <th>Superbad (2007)</th>
      <th>Supergirl (1984)</th>
      <th>Superman (1978)</th>
      <th>Superman II (1980)</th>
      <th>Superman III (1983)</th>
      <th>Sure Thing, The (1985)</th>
      <th>Suspicion (1941)</th>
      <th>Sweet Charity (1969)</th>
      <th>Sweet Hereafter, The (1997)</th>
      <th>Swing Time (1936)</th>
      <th>Sword in the Stone, The (1963)</th>
      <th>Sydney (Hard Eight) (1996)</th>
      <th>Syriana (2005)</th>
      <th>Taken 2 (2012)</th>
      <th>Taken 3 (2015)</th>
      <th>Taking of Pelham One Two Three, The (1974)</th>
      <th>Tale of Two Cities, A (1935)</th>
      <th>Talented Mr. Ripley, The (1999)</th>
      <th>Talk of the Town, The (1942)</th>
      <th>Talk to Her (Hable con Ella) (2002)</th>
      <th>Taming of the Shrew, The (1967)</th>
      <th>Tarnation (2003)</th>
      <th>Taxi Driver (1976)</th>
      <th>Ted (2012)</th>
      <th>Tekkonkinkreet (Tekkon kinkurîto) (2006)</th>
      <th>Tender Mercies (1983)</th>
      <th>Terminator 2: Judgment Day (1991)</th>
      <th>Terminator Salvation (2009)</th>
      <th>Terminator, The (1984)</th>
      <th>Terrorist, The (a.k.a. Malli) (Theeviravaathi) (1998)</th>
      <th>Tetsuo, the Ironman (Tetsuo) (1988)</th>
      <th>That Thing You Do! (1996)</th>
      <th>That's Entertainment (1974)</th>
      <th>The Butterfly Effect (2004)</th>
      <th>The Count of Monte Cristo (2002)</th>
      <th>The DUFF (2015)</th>
      <th>The Devil's Advocate (1997)</th>
      <th>The Greatest Showman (2017)</th>
      <th>The Hateful Eight (2015)</th>
      <th>The Hobbit: The Battle of the Five Armies (2014)</th>
      <th>The Hunger Games (2012)</th>
      <th>The Hunger Games: Mockingjay - Part 2 (2015)</th>
      <th>The Imitation Game (2014)</th>
      <th>The Importance of Being Earnest (1952)</th>
      <th>The Interview (2014)</th>
      <th>The Lego Batman Movie (2017)</th>
      <th>The Lego Movie (2014)</th>
      <th>The Machinist (2004)</th>
      <th>The Man from U.N.C.L.E. (2015)</th>
      <th>The Martian (2015)</th>
      <th>The Rabbi's Cat (Le chat du rabbin) (2011)</th>
      <th>The Revenant (2015)</th>
      <th>The Shape of Water (2017)</th>
      <th>The Spiral Staircase (1945)</th>
      <th>There Will Be Blood (2007)</th>
      <th>There's Something About Mary (1998)</th>
      <th>They Drive by Night (1940)</th>
      <th>They Shoot Horses, Don't They? (1969)</th>
      <th>Thin Blue Line, The (1988)</th>
      <th>Thin Man Goes Home, The (1945)</th>
      <th>Thin Man, The (1934)</th>
      <th>Thing from Another World, The (1951)</th>
      <th>Third Man, The (1949)</th>
      <th>Thirteen (2003)</th>
      <th>This Gun for Hire (1942)</th>
      <th>This Is Spinal Tap (1984)</th>
      <th>Thor: Ragnarok (2017)</th>
      <th>Thousand Words, A (2012)</th>
      <th>Three Colors: Blue (Trois couleurs: Bleu) (1993)</th>
      <th>Three Colors: White (Trzy kolory: Bialy) (1994)</th>
      <th>Three Faces of Eve, The (1957)</th>
      <th>Three Musketeers, The (1973)</th>
      <th>Three Musketeers, The (1993)</th>
      <th>Throne of Blood (Kumonosu jô) (1957)</th>
      <th>Throw Momma from the Train (1987)</th>
      <th>Time Bandits (1981)</th>
      <th>Time to Kill, A (1996)</th>
      <th>Tin Cup (1996)</th>
      <th>Titanic (1997)</th>
      <th>Titus (1999)</th>
      <th>To Die For (1995)</th>
      <th>To Kill a Mockingbird (1962)</th>
      <th>To Live (Huozhe) (1994)</th>
      <th>Tokyo Godfathers (2003)</th>
      <th>Tom Jones (1963)</th>
      <th>Tomb Raider (2018)</th>
      <th>Tootsie (1982)</th>
      <th>Top Gun (1986)</th>
      <th>Top Hat (1935)</th>
      <th>Touching the Void (2003)</th>
      <th>Town Called Panic, A (Panique au village) (2009)</th>
      <th>Town, The (2010)</th>
      <th>Toy Story (1995)</th>
      <th>Toy Story 2 (1999)</th>
      <th>Trading Places (1983)</th>
      <th>Traffic (2000)</th>
      <th>Trainspotting (1996)</th>
      <th>Transformers: Revenge of the Fallen (2009)</th>
      <th>Treasure of the Sierra Madre, The (1948)</th>
      <th>Trekkies (1997)</th>
      <th>Triplets of Belleville, The (Les triplettes de Belleville) (2003)</th>
      <th>Tron: Legacy (2010)</th>
      <th>True Grit (2010)</th>
      <th>True Lies (1994)</th>
      <th>Truly, Madly, Deeply (1991)</th>
      <th>Truman Show, The (1998)</th>
      <th>Truth About Cats &amp; Dogs, The (1996)</th>
      <th>Tucker &amp; Dale vs Evil (2010)</th>
      <th>Tupac: Resurrection (2003)</th>
      <th>Turner &amp; Hooch (1989)</th>
      <th>Twelfth Night (1996)</th>
      <th>Twelve Monkeys (a.k.a. 12 Monkeys) (1995)</th>
      <th>Twentieth Century (1934)</th>
      <th>Twilight (2008)</th>
      <th>Twilight Samurai, The (Tasogare Seibei) (2002)</th>
      <th>Twins (1988)</th>
      <th>Twister (1996)</th>
      <th>Two Days, One Night (Deux jours, une nuit) (2014)</th>
      <th>Two Family House (2000)</th>
      <th>UHF (1989)</th>
      <th>Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)</th>
      <th>Unbreakable (2000)</th>
      <th>Unforgiven (1992)</th>
      <th>Untouchables, The (1987)</th>
      <th>Unvanquished, The (Aparajito) (1957)</th>
      <th>Up (2009)</th>
      <th>Up Close and Personal (1996)</th>
      <th>Upside Down: The Creation Records Story (2010)</th>
      <th>Upstream Color (2013)</th>
      <th>Usual Suspects, The (1995)</th>
      <th>Vanishing, The (1993)</th>
      <th>Vanishing, The (Spoorloos) (1988)</th>
      <th>Vera Drake (2004)</th>
      <th>Vertigo (1958)</th>
      <th>Very Bad Things (1998)</th>
      <th>Very Brady Sequel, A (1996)</th>
      <th>Vicky Cristina Barcelona (2008)</th>
      <th>Victor/Victoria (1982)</th>
      <th>Village of the Damned (1960)</th>
      <th>Village, The (2004)</th>
      <th>Virgin Suicides, The (1999)</th>
      <th>Virtuosity (1995)</th>
      <th>Voices of a Distant Star (Hoshi no koe) (2003)</th>
      <th>Volunteers (1985)</th>
      <th>WALL·E (2008)</th>
      <th>Waco: The Rules of Engagement (1997)</th>
      <th>Wages of Fear, The (Salaire de la peur, Le) (1953)</th>
      <th>Wait Until Dark (1967)</th>
      <th>Waiting for Guffman (1996)</th>
      <th>Waiting... (2005)</th>
      <th>Walk the Line (2005)</th>
      <th>Walk, Don't Run (1966)</th>
      <th>Wall Street (1987)</th>
      <th>Wallace &amp; Gromit in The Curse of the Were-Rabbit (2005)</th>
      <th>Wallace &amp; Gromit: A Close Shave (1995)</th>
      <th>Wallace &amp; Gromit: The Best of Aardman Animation (1996)</th>
      <th>Wallace &amp; Gromit: The Wrong Trousers (1993)</th>
      <th>War Room, The (1993)</th>
      <th>War of the Worlds, The (1953)</th>
      <th>WarGames (1983)</th>
      <th>Warrior (2011)</th>
      <th>Washington Square (1997)</th>
      <th>Watch on the Rhine (1943)</th>
      <th>Waterboy, The (1998)</th>
      <th>Watership Down (1978)</th>
      <th>Weather Underground, The (2002)</th>
      <th>Wedding Banquet, The (Xi yan) (1993)</th>
      <th>Wedding Crashers (2005)</th>
      <th>Wedding Singer, The (1998)</th>
      <th>Welcome to the Dollhouse (1995)</th>
      <th>West Side Story (1961)</th>
      <th>Whale Rider (2002)</th>
      <th>What Ever Happened to Baby Jane? (1962)</th>
      <th>What If (2013)</th>
      <th>What's Eating Gilbert Grape (1993)</th>
      <th>What's Love Got to Do with It? (1993)</th>
      <th>When Harry Met Sally... (1989)</th>
      <th>When We Were Kings (1996)</th>
      <th>When a Man Loves a Woman (1994)</th>
      <th>While You Were Sleeping (1995)</th>
      <th>Whiplash (2014)</th>
      <th>White Christmas (1954)</th>
      <th>White Men Can't Jump (1992)</th>
      <th>Who Framed Roger Rabbit? (1988)</th>
      <th>Who Killed Chea Vichea? (2010)</th>
      <th>Whole Wide World, The (1996)</th>
      <th>Why We Fight (2005)</th>
      <th>Widow of St. Pierre, The (Veuve de Saint-Pierre, La) (2000)</th>
      <th>Wild Parrots of Telegraph Hill, The (2003)</th>
      <th>Wild Tales (2014)</th>
      <th>William Shakespeare's Romeo + Juliet (1996)</th>
      <th>Winged Migration (Peuple migrateur, Le) (2001)</th>
      <th>Wings of the Dove, The (1997)</th>
      <th>Wit (2001)</th>
      <th>With a Friend Like Harry... (Harry, un ami qui vous veut du bien) (2000)</th>
      <th>Witness (1985)</th>
      <th>Witness for the Prosecution (1957)</th>
      <th>Wizard of Oz, The (1939)</th>
      <th>Wolf of Wall Street, The (2013)</th>
      <th>Woman Under the Influence, A (1974)</th>
      <th>Woman of the Year (1942)</th>
      <th>Women, The (1939)</th>
      <th>Wonder Boys (2000)</th>
      <th>Wonderful, Horrible Life of Leni Riefenstahl, The (Macht der Bilder: Leni Riefenstahl, Die) (1993)</th>
      <th>Woodsman, The (2004)</th>
      <th>World of Apu, The (Apur Sansar) (1959)</th>
      <th>World of Henry Orient, The (1964)</th>
      <th>World of Tomorrow (2015)</th>
      <th>X-Files: Fight the Future, The (1998)</th>
      <th>X-Men (2000)</th>
      <th>X-Men Origins: Wolverine (2009)</th>
      <th>X-Men: The Last Stand (2006)</th>
      <th>X2: X-Men United (2003)</th>
      <th>Yankee Doodle Dandy (1942)</th>
      <th>Year of Living Dangerously, The (1982)</th>
      <th>Yearling, The (1946)</th>
      <th>Yentl (1983)</th>
      <th>Yojimbo (1961)</th>
      <th>You Can Count on Me (2000)</th>
      <th>You Only Live Once (1937)</th>
      <th>You'll Never Get Rich (1941)</th>
      <th>You've Got Mail (1998)</th>
      <th>Young Frankenstein (1974)</th>
      <th>Z (1969)</th>
      <th>Zack and Miri Make a Porno (2008)</th>
      <th>Zelary (2003)</th>
      <th>Zelig (1983)</th>
      <th>Zero Dark Thirty (2012)</th>
      <th>Zombieland (2009)</th>
      <th>Zoolander (2001)</th>
      <th>Zulu (1964)</th>
      <th>eXistenZ (1999)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1572 columns</p>
</div>




```python
df.shape
```




    (58, 1572)




```python
from mlxtend.frequent_patterns import apriori
%time
apriori_frequent_itemsets = apriori(df, min_support=0.01,use_colnames=True, max_len = 2)
#1% support value, and maxixmum length of two items in each row is selected
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 5.72 µs



```python
apriori_frequent_itemsets['itemsets'].apply(lambda x: len(x)).value_counts()
#len(x) counts objects in list
#.value_counts() counts unique len(x) counts

```




    2    774986
    1      1572
    Name: itemsets, dtype: int64




```python
apriori_frequent_itemsets
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
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.017241</td>
      <td>((500) Days of Summer (2009))</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.017241</td>
      <td>(...And Justice for All (1979))</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.017241</td>
      <td>(10 Cloverfield Lane (2016))</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.017241</td>
      <td>(10 Things I Hate About You (1999))</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.017241</td>
      <td>(101 Dalmatians (1996))</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>776553</td>
      <td>0.017241</td>
      <td>(Zulu (1964), Zelary (2003))</td>
    </tr>
    <tr>
      <td>776554</td>
      <td>0.017241</td>
      <td>(eXistenZ (1999), Zelary (2003))</td>
    </tr>
    <tr>
      <td>776555</td>
      <td>0.017241</td>
      <td>(Zelig (1983), Zulu (1964))</td>
    </tr>
    <tr>
      <td>776556</td>
      <td>0.017241</td>
      <td>(Zelig (1983), eXistenZ (1999))</td>
    </tr>
    <tr>
      <td>776557</td>
      <td>0.017241</td>
      <td>(eXistenZ (1999), Zulu (1964))</td>
    </tr>
  </tbody>
</table>
<p>776558 rows × 2 columns</p>
</div>




```python
apriori_frequent_itemsets['length'] = apriori_frequent_itemsets['itemsets'].apply(lambda x: len(x))
apriori_frequent_itemsets
apriori_frequent_itemsets.to_csv('frequent_itemset.csv')
```


```python
apriori_frequent_itemsets[(apriori_frequent_itemsets['length'] > 1)
                          & (apriori_frequent_itemsets['support'] > 0.051)]
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
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>51559</td>
      <td>0.051724</td>
      <td>(American History X (1998), Game, The (1997))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>51844</td>
      <td>0.051724</td>
      <td>(American History X (1998), Matrix, The (1999))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>52147</td>
      <td>0.051724</td>
      <td>(American History X (1998), Schindler's List (...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148083</td>
      <td>0.068966</td>
      <td>(Blade Runner (1982), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148090</td>
      <td>0.051724</td>
      <td>(Dr. Strangelove or: How I Learned to Stop Wor...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148117</td>
      <td>0.068966</td>
      <td>(Eternal Sunshine of the Spotless Mind (2004),...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148340</td>
      <td>0.051724</td>
      <td>(Inception (2010), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148508</td>
      <td>0.068966</td>
      <td>(Memento (2000), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148679</td>
      <td>0.051724</td>
      <td>(Pi (1998), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148723</td>
      <td>0.051724</td>
      <td>(Psycho (1960), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148767</td>
      <td>0.051724</td>
      <td>(Reservoir Dogs (1992), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>148977</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Blade Runn...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>149048</td>
      <td>0.051724</td>
      <td>(Twelve Monkeys (a.k.a. 12 Monkeys) (1995), Bl...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>149057</td>
      <td>0.051724</td>
      <td>(Unbreakable (2000), Blade Runner (1982))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>318413</td>
      <td>0.051724</td>
      <td>(Dr. Strangelove or: How I Learned to Stop Wor...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>318440</td>
      <td>0.086207</td>
      <td>(Eternal Sunshine of the Spotless Mind (2004),...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>318663</td>
      <td>0.051724</td>
      <td>(Inception (2010), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>318832</td>
      <td>0.068966</td>
      <td>(Memento (2000), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319003</td>
      <td>0.051724</td>
      <td>(Pi (1998), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319047</td>
      <td>0.051724</td>
      <td>(Psycho (1960), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319091</td>
      <td>0.051724</td>
      <td>(Reservoir Dogs (1992), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319257</td>
      <td>0.051724</td>
      <td>(Star Wars: Episode IV - A New Hope (1977), Do...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319301</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Donnie Dar...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319372</td>
      <td>0.051724</td>
      <td>(Twelve Monkeys (a.k.a. 12 Monkeys) (1995), Do...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319381</td>
      <td>0.051724</td>
      <td>(Unbreakable (2000), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>324291</td>
      <td>0.051724</td>
      <td>(Eternal Sunshine of the Spotless Mind (2004),...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>324672</td>
      <td>0.051724</td>
      <td>(Memento (2000), Dr. Strangelove or: How I Lea...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>325209</td>
      <td>0.051724</td>
      <td>(Unbreakable (2000), Dr. Strangelove or: How I...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>348984</td>
      <td>0.051724</td>
      <td>(Eternal Sunshine of the Spotless Mind (2004),...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349153</td>
      <td>0.068966</td>
      <td>(Memento (2000), Eternal Sunshine of the Spotl...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349324</td>
      <td>0.051724</td>
      <td>(Pi (1998), Eternal Sunshine of the Spotless M...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349368</td>
      <td>0.051724</td>
      <td>(Psycho (1960), Eternal Sunshine of the Spotle...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349412</td>
      <td>0.051724</td>
      <td>(Reservoir Dogs (1992), Eternal Sunshine of th...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349578</td>
      <td>0.051724</td>
      <td>(Eternal Sunshine of the Spotless Mind (2004),...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349622</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Eternal Su...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349693</td>
      <td>0.051724</td>
      <td>(Twelve Monkeys (a.k.a. 12 Monkeys) (1995), Et...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>349702</td>
      <td>0.051724</td>
      <td>(Unbreakable (2000), Eternal Sunshine of the S...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>376813</td>
      <td>0.051724</td>
      <td>(Fight Club (1999), Pulp Fiction (1994))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>410921</td>
      <td>0.051724</td>
      <td>(Game, The (1997), Matrix, The (1999))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>411224</td>
      <td>0.051724</td>
      <td>(Schindler's List (1993), Game, The (1997))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>516345</td>
      <td>0.051724</td>
      <td>(Memento (2000), Inception (2010))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>605492</td>
      <td>0.051724</td>
      <td>(Schindler's List (1993), Matrix, The (1999))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610095</td>
      <td>0.051724</td>
      <td>(Memento (2000), Pi (1998))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610139</td>
      <td>0.051724</td>
      <td>(Memento (2000), Psycho (1960))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610183</td>
      <td>0.051724</td>
      <td>(Memento (2000), Reservoir Dogs (1992))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610393</td>
      <td>0.051724</td>
      <td>(Memento (2000), Terminator 2: Judgment Day (1...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610464</td>
      <td>0.051724</td>
      <td>(Memento (2000), Twelve Monkeys (a.k.a. 12 Mon...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>610473</td>
      <td>0.051724</td>
      <td>(Memento (2000), Unbreakable (2000))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>703733</td>
      <td>0.051724</td>
      <td>(Psycho (1960), Reservoir Dogs (1992))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>703935</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Psycho (19...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>703993</td>
      <td>0.051724</td>
      <td>(Psycho (1960), Twelve Monkeys (a.k.a. 12 Monk...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>704284</td>
      <td>0.051724</td>
      <td>(Pulp Fiction (1994), Star Wars: Episode IV - ...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>718458</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Reservoir ...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>718516</td>
      <td>0.051724</td>
      <td>(Reservoir Dogs (1992), Twelve Monkeys (a.k.a....</td>
      <td>2</td>
    </tr>
    <tr>
      <td>765370</td>
      <td>0.051724</td>
      <td>(Terminator 2: Judgment Day (1991), Twelve Mon...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
apriori_frequent_itemsets[(apriori_frequent_itemsets['length'] != 1)]
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
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1572</td>
      <td>0.017241</td>
      <td>(Akira (1988), (500) Days of Summer (2009))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1573</td>
      <td>0.017241</td>
      <td>(American Pie (1999), (500) Days of Summer (20...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1574</td>
      <td>0.017241</td>
      <td>(Avatar (2009), (500) Days of Summer (2009))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1575</td>
      <td>0.017241</td>
      <td>(Babe (1995), (500) Days of Summer (2009))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1576</td>
      <td>0.017241</td>
      <td>(Big Daddy (1999), (500) Days of Summer (2009))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>776553</td>
      <td>0.017241</td>
      <td>(Zulu (1964), Zelary (2003))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>776554</td>
      <td>0.017241</td>
      <td>(eXistenZ (1999), Zelary (2003))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>776555</td>
      <td>0.017241</td>
      <td>(Zelig (1983), Zulu (1964))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>776556</td>
      <td>0.017241</td>
      <td>(Zelig (1983), eXistenZ (1999))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>776557</td>
      <td>0.017241</td>
      <td>(eXistenZ (1999), Zulu (1964))</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>774986 rows × 3 columns</p>
</div>




```python
apriori_frequent_itemsets[apriori_frequent_itemsets['itemsets'].apply(lambda x: 'Donnie Darko (2001)' in str(x))]
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
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>370</td>
      <td>0.086207</td>
      <td>(Donnie Darko (2001))</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1581</td>
      <td>0.017241</td>
      <td>((500) Days of Summer (2009), Donnie Darko (20...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>1923</td>
      <td>0.017241</td>
      <td>(...And Justice for All (1979), Donnie Darko (...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2898</td>
      <td>0.017241</td>
      <td>(10 Cloverfield Lane (2016), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3264</td>
      <td>0.017241</td>
      <td>(Donnie Darko (2001), 10 Things I Hate About Y...</td>
      <td>2</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>319473</td>
      <td>0.017241</td>
      <td>(Donnie Darko (2001), Zelary (2003))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319474</td>
      <td>0.017241</td>
      <td>(Donnie Darko (2001), Zelig (1983))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319475</td>
      <td>0.017241</td>
      <td>(Zoolander (2001), Donnie Darko (2001))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319476</td>
      <td>0.017241</td>
      <td>(Donnie Darko (2001), Zulu (1964))</td>
      <td>2</td>
    </tr>
    <tr>
      <td>319477</td>
      <td>0.017241</td>
      <td>(Donnie Darko (2001), eXistenZ (1999))</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1398 rows × 3 columns</p>
</div>




```python
import seaborn as sns
sns.heatmap(data=apriori_frequent_itemsets.corr(method='spearman'),
           annot=True,
           vmin=-1,
           vmax=1,
           center=0,
           cmap='YlGnBu');
```


    
![png](output_74_0.png)
    



```python
%%time
from mlxtend.frequent_patterns import association_rules
rules = association_rules(apriori_frequent_itemsets,metric="lift",min_threshold=0.01)
```

    CPU times: user 10.9 s, sys: 497 ms, total: 11.4 s
    Wall time: 11.1 s



```python
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>((500) Days of Summer (2009))</td>
      <td>(Akira (1988))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1</td>
      <td>(Akira (1988))</td>
      <td>((500) Days of Summer (2009))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>2</td>
      <td>((500) Days of Summer (2009))</td>
      <td>(American Pie (1999))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>3</td>
      <td>(American Pie (1999))</td>
      <td>((500) Days of Summer (2009))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>4</td>
      <td>((500) Days of Summer (2009))</td>
      <td>(Avatar (2009))</td>
      <td>0.017241</td>
      <td>0.034483</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>0.016647</td>
      <td>inf</td>
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
    </tr>
    <tr>
      <td>1549967</td>
      <td>(Zelig (1983))</td>
      <td>(Zulu (1964))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549968</td>
      <td>(Zelig (1983))</td>
      <td>(eXistenZ (1999))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549969</td>
      <td>(eXistenZ (1999))</td>
      <td>(Zelig (1983))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549970</td>
      <td>(Zulu (1964))</td>
      <td>(eXistenZ (1999))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549971</td>
      <td>(eXistenZ (1999))</td>
      <td>(Zulu (1964))</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
<p>1549972 rows × 9 columns</p>
</div>




```python
rules['conviction'].unique()
```




    array([       inf, 1.96551724, 1.31034483, 1.22844828, 1.47413793,
           1.09195402, 1.93103448, 1.44827586, 1.89655172, 1.6091954 ,
           1.28735632, 1.86206897, 2.89655172, 1.65517241, 1.07279693,
           1.82758621, 1.20689655, 1.26436782, 1.39655172, 2.79310345,
           2.84482759, 1.42241379, 1.58045977, 2.74137931, 1.18534483,
           1.37068966, 3.79310345, 2.48275862, 1.24137931, 1.1637931 ,
           1.2183908 , 1.10344828, 1.03448276, 4.65517241, 2.32758621,
           3.65517241, 3.72413793, 1.24137931, 1.05363985, 2.28448276,
           1.55172414, 4.56896552, 2.37068966, 2.06896552, 1.30541872,
           1.01532567, 1.37931034, 1.14224138, 3.31034483, 1.33004926])




```python
rules[rules["antecedents"].apply(lambda x: 'Donnie Darko (2001)' in str(x))].sort_values(ascending=False,by='lift')
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>19</td>
      <td>(Donnie Darko (2001))</td>
      <td>((500) Days of Summer (2009))</td>
      <td>0.086207</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>11.600000</td>
      <td>0.015755</td>
      <td>1.228448</td>
    </tr>
    <tr>
      <td>634846</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Penny Serenade (1941))</td>
      <td>0.086207</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>11.600000</td>
      <td>0.015755</td>
      <td>1.228448</td>
    </tr>
    <tr>
      <td>634862</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Pi (1998))</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0.051724</td>
      <td>0.6</td>
      <td>11.600000</td>
      <td>0.047265</td>
      <td>2.370690</td>
    </tr>
    <tr>
      <td>634860</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Phone Booth (2002))</td>
      <td>0.086207</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>11.600000</td>
      <td>0.015755</td>
      <td>1.228448</td>
    </tr>
    <tr>
      <td>634858</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Philadelphia Story, The (1940))</td>
      <td>0.086207</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>11.600000</td>
      <td>0.015755</td>
      <td>1.228448</td>
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
    </tr>
    <tr>
      <td>634436</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Léon: The Professional (a.k.a. The Profession...</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>3.866667</td>
      <td>0.012782</td>
      <td>1.185345</td>
    </tr>
    <tr>
      <td>518397</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Corpse Bride (2005))</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>3.866667</td>
      <td>0.012782</td>
      <td>1.185345</td>
    </tr>
    <tr>
      <td>603874</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Departed, The (2006))</td>
      <td>0.086207</td>
      <td>0.051724</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>3.866667</td>
      <td>0.012782</td>
      <td>1.185345</td>
    </tr>
    <tr>
      <td>635370</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Star Wars: Episode IV - A New Hope (1977))</td>
      <td>0.086207</td>
      <td>0.172414</td>
      <td>0.051724</td>
      <td>0.6</td>
      <td>3.480000</td>
      <td>0.036861</td>
      <td>2.068966</td>
    </tr>
    <tr>
      <td>118732</td>
      <td>(Donnie Darko (2001))</td>
      <td>(Anchorman: The Legend of Ron Burgundy (2004))</td>
      <td>0.086207</td>
      <td>0.068966</td>
      <td>0.017241</td>
      <td>0.2</td>
      <td>2.900000</td>
      <td>0.011296</td>
      <td>1.163793</td>
    </tr>
  </tbody>
</table>
<p>1397 rows × 9 columns</p>
</div>




```python
rules[rules["antecedents"].apply(lambda x: 'Donnie Darko (2001)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['lift']].max().sort_values(ascending=False,by='lift').head(10)
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
      <th>lift</th>
    </tr>
    <tr>
      <th>antecedents</th>
      <th>consequents</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" valign="top">(Donnie Darko (2001))</td>
      <td>((500) Days of Summer (2009))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(A Pigeon Sat on a Branch Reflecting on Existence (2014))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(127 Hours (2010))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(25th Hour (2002))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(About a Boy (2002))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(39 Steps, The (1935))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(400 Blows, The (Les quatre cents coups) (1959))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(42 Up (1998))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(84 Charing Cross Road (1987))</td>
      <td>11.6</td>
    </tr>
    <tr>
      <td>(8MM (1999))</td>
      <td>11.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules[rules["antecedents"].apply(lambda x: 'Donnie Darko (2001)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['confidence']].max().sort_values(ascending=False,
      by='confidence').head(10).plot(kind='bar').invert_xaxis()

plt.title('Top movies that are likley to be watched with Donnie Darko');
```


    
![png](output_80_0.png)
    



```python
rules['antecedents'] = rules.antecedents.apply(lambda x: next(iter(x)))
rules['consequents'] = rules.consequents.apply(lambda x: next(iter(x)))
#The next() function returns the next item from the iterator.
```


```python
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>(500) Days of Summer (2009)</td>
      <td>Akira (1988)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Akira (1988)</td>
      <td>(500) Days of Summer (2009)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>2</td>
      <td>(500) Days of Summer (2009)</td>
      <td>American Pie (1999)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>3</td>
      <td>American Pie (1999)</td>
      <td>(500) Days of Summer (2009)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>4</td>
      <td>(500) Days of Summer (2009)</td>
      <td>Avatar (2009)</td>
      <td>0.017241</td>
      <td>0.034483</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>0.016647</td>
      <td>inf</td>
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
    </tr>
    <tr>
      <td>1549967</td>
      <td>Zelig (1983)</td>
      <td>Zulu (1964)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549968</td>
      <td>Zelig (1983)</td>
      <td>eXistenZ (1999)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549969</td>
      <td>eXistenZ (1999)</td>
      <td>Zelig (1983)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549970</td>
      <td>Zulu (1964)</td>
      <td>eXistenZ (1999)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
    <tr>
      <td>1549971</td>
      <td>eXistenZ (1999)</td>
      <td>Zulu (1964)</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>0.017241</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>0.016944</td>
      <td>inf</td>
    </tr>
  </tbody>
</table>
<p>1549972 rows × 9 columns</p>
</div>




```python
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
edges = nx.from_pandas_edgelist(rules.sort_values(ascending=False,by='lift').head(100)
                           ,source='antecedents',target='consequents',edge_attr=None)
plt.subplots(figsize=(40,30))
plt.suptitle('Top 100 movies in term of (lift)', fontsize = 50,fontweight = 'bold')
pos = nx.planar_layout(edges)
nx.draw_networkx_nodes(edges, pos, node_size = 2000,alpha= 0.7,node_color = 'tomato')
nx.draw_networkx_edges(edges, pos, width = 6, alpha = 0.2, edge_color = 'indigo')
nx.draw_networkx_labels(edges, pos, font_size = 25, font_family = 'FreeMono',weight='bold')
plt.grid()
plt.axis('off')
plt.tight_layout()
plt.show()
```

    findfont: Font family ['FreeMono'] not found. Falling back to DejaVu Sans.



    
![png](output_83_1.png)
    



```python
rules[rules["antecedents"].apply(lambda x: 'Mean Creek (2004)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['lift']].max().sort_values(ascending=False,by='lift').head(10)
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
      <th>lift</th>
    </tr>
    <tr>
      <th>antecedents</th>
      <th>consequents</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" valign="top">Mean Creek (2004)</td>
      <td>...And Justice for All (1979)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Place in the Sun, A (1951)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Pat and Mike (1952)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of the Christ, The (2004)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of Joan of Arc, The (Passion de Jeanne d'Arc, La) (1928)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion Fish (1992)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passage to India, A (1984)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parenthood (1989)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1998)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1961)</td>
      <td>58.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules[rules["antecedents"].apply(lambda x: 'Meatballs (1979)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['lift']].max().sort_values(ascending=False,by='lift').head(10)
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
      <th>lift</th>
    </tr>
    <tr>
      <th>antecedents</th>
      <th>consequents</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" valign="top">Meatballs (1979)</td>
      <td>...And Justice for All (1979)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Place in the Sun, A (1951)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Pat and Mike (1952)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of the Christ, The (2004)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of Joan of Arc, The (Passion de Jeanne d'Arc, La) (1928)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion Fish (1992)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passage to India, A (1984)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parenthood (1989)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1998)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1961)</td>
      <td>58.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules[rules["antecedents"].apply(lambda x: 'Manhattan (1979)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['lift']].max().sort_values(ascending=False,by='lift').head(10)
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
      <th>lift</th>
    </tr>
    <tr>
      <th>antecedents</th>
      <th>consequents</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" valign="top">Manhattan (1979)</td>
      <td>...And Justice for All (1979)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Place in the Sun, A (1951)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Pat and Mike (1952)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of the Christ, The (2004)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion of Joan of Arc, The (Passion de Jeanne d'Arc, La) (1928)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passion Fish (1992)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Passage to India, A (1984)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parenthood (1989)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1998)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Parent Trap, The (1961)</td>
      <td>58.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
rules[rules["antecedents"].apply(lambda x: '(500) Days of Summer (2009)' in str(x))].groupby(
    ['antecedents', 'consequents'])[['lift']].max().sort_values(ascending=False,by='lift').head(10)
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
      <th>lift</th>
    </tr>
    <tr>
      <th>antecedents</th>
      <th>consequents</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10" valign="top">(500) Days of Summer (2009)</td>
      <td>Akira (1988)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Session 9 (2001)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Lady in the Water (2006)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Men in Black II (a.k.a. MIIB) (a.k.a. MIB 2) (2002)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Moon (2009)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>More (1998)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Neon Genesis Evangelion: Death &amp; Rebirth (Shin seiki Evangelion Gekijô-ban: Shito shinsei) (1997)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>American Pie (1999)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Paprika (Papurika) (2006)</td>
      <td>58.0</td>
    </tr>
    <tr>
      <td>Punisher, The (2004)</td>
      <td>58.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
