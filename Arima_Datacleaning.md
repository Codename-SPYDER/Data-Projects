***
## IMPORTING THE REQUIRED LIBRARIES & DATA SET
***


```python
import pandas as pd
import numpy as np
import scipy
import re
import string

import seaborn as sns
import matplotlib.pyplot as plt
#import scikitplot as skplt
from wordcloud import WordCloud


from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
#import lightgbm as lgb

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore') 

from IPython.display import Image

%matplotlib inline

# Educ tutorial 
#from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn import decomposition, ensemble
```


```python
clothes_reviews = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
```


```python
clothes_reviews
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
      <th>Unnamed: 0</th>
      <th>Clothing ID</th>
      <th>Age</th>
      <th>Title</th>
      <th>Review Text</th>
      <th>Rating</th>
      <th>Recommended IND</th>
      <th>Positive Feedback Count</th>
      <th>Division Name</th>
      <th>Department Name</th>
      <th>Class Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>767</td>
      <td>33</td>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comf...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>Initmates</td>
      <td>Intimate</td>
      <td>Intimates</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1080</td>
      <td>34</td>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happene...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1077</td>
      <td>60</td>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and reall...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1049</td>
      <td>50</td>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, fl...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Bottoms</td>
      <td>Pants</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>847</td>
      <td>47</td>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to th...</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>General</td>
      <td>Tops</td>
      <td>Blouses</td>
    </tr>
    <tr>
      <th>...</th>
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
      <th>23481</th>
      <td>23481</td>
      <td>1104</td>
      <td>34</td>
      <td>Great dress for many occasions</td>
      <td>I was very happy to snag this dress at such a ...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>23482</th>
      <td>23482</td>
      <td>862</td>
      <td>48</td>
      <td>Wish it was made of cotton</td>
      <td>It reminds me of maternity clothes. soft, stre...</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Tops</td>
      <td>Knits</td>
    </tr>
    <tr>
      <th>23483</th>
      <td>23483</td>
      <td>1104</td>
      <td>31</td>
      <td>Cute, but see through</td>
      <td>This fit well, but the top was very see throug...</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>General Petite</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>23484</th>
      <td>23484</td>
      <td>1084</td>
      <td>28</td>
      <td>Very cute dress, perfect for summer parties an...</td>
      <td>I bought this dress for a wedding i have this ...</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
    <tr>
      <th>23485</th>
      <td>23485</td>
      <td>1104</td>
      <td>52</td>
      <td>Please make more like this one!</td>
      <td>This dress in a lovely platinum is feminine an...</td>
      <td>5</td>
      <td>1</td>
      <td>22</td>
      <td>General Petite</td>
      <td>Dresses</td>
      <td>Dresses</td>
    </tr>
  </tbody>
</table>
<p>23486 rows × 11 columns</p>
</div>



***
## EXPLORATORY DATA ANALYSIS & CLEANING
***


```python
#drop rows with missing in reviews text
clothes_reviews=clothes_reviews.dropna(how='any',
                    subset=['Review Text'])
```


```python
#define the plot size
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size
```

    6.0
    4.0



```python
# To show the distrubution of reviews by Age
clothes_reviews['Age'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b2179d0708>




    
![png](output_7_1.png)
    



```python
# Pie chart to depict the weight of each rating 
clothes_reviews.Rating.value_counts().plot(kind='pie', autopct='%1.0f%%')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b212a966c8>




    
![png](output_8_1.png)
    



```python
# Pie chart to depict the weight of each division
clothes_reviews['Division Name'].value_counts().plot(kind='pie', autopct='%1.0f%%')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b218b1d808>




    
![png](output_9_1.png)
    



```python
# Pie chart to depict the weight of each department
clothes_reviews['Department Name'].value_counts().plot(kind='pie', autopct='%1.0f%%')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b218b54388>




    
![png](output_10_1.png)
    



```python
#correlation matrix 
corr = clothes_reviews.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50,220,n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticketlabels(),
    rotation=45,
    horizontalalignment='right'
);
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-51-d7e2f9340aa1> in <module>
          8 )
          9 ax.set_xticklabels(
    ---> 10     ax.get_xticketlabels(),
         11     rotation=45,
         12     horizontalalignment='right'


    AttributeError: 'AxesSubplot' object has no attribute 'get_xticketlabels'



    
![png](output_11_1.png)
    


***
## TEXT FEATURES
***


```python
# adding extra column to get the length of the review text
clothes_reviews['Review Length'] = clothes_reviews['Review Text'].map(lambda text: len(text))
clothes_reviews.head()
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
      <th>Unnamed: 0</th>
      <th>Clothing ID</th>
      <th>Age</th>
      <th>Title</th>
      <th>Review Text</th>
      <th>Rating</th>
      <th>Recommended IND</th>
      <th>Positive Feedback Count</th>
      <th>Division Name</th>
      <th>Department Name</th>
      <th>Class Name</th>
      <th>Review Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>767</td>
      <td>33</td>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comf...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>Initmates</td>
      <td>Intimate</td>
      <td>Intimates</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1080</td>
      <td>34</td>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happene...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
      <td>303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1077</td>
      <td>60</td>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and reall...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1049</td>
      <td>50</td>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, fl...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Bottoms</td>
      <td>Pants</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>847</td>
      <td>47</td>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to th...</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>General</td>
      <td>Tops</td>
      <td>Blouses</td>
      <td>192</td>
    </tr>
  </tbody>
</table>
</div>




```python
# assign reviews with score > 3 as positive sentiment and scores < 3 as negative sentiment and neutral sentiment when = 3
clothes_reviews['Sentiment'] = clothes_reviews['Rating'].apply(lambda Rating : +1 if Rating > 3 else (0 if Rating == 3 else -1))
```


```python
clothes_reviews.head()
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
      <th>Unnamed: 0</th>
      <th>Clothing ID</th>
      <th>Age</th>
      <th>Title</th>
      <th>Review Text</th>
      <th>Rating</th>
      <th>Recommended IND</th>
      <th>Positive Feedback Count</th>
      <th>Division Name</th>
      <th>Department Name</th>
      <th>Class Name</th>
      <th>Review Length</th>
      <th>Sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>767</td>
      <td>33</td>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comf...</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>Initmates</td>
      <td>Intimate</td>
      <td>Intimates</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1080</td>
      <td>34</td>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happene...</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
      <td>303</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1077</td>
      <td>60</td>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and reall...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>General</td>
      <td>Dresses</td>
      <td>Dresses</td>
      <td>500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1049</td>
      <td>50</td>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, fl...</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>General Petite</td>
      <td>Bottoms</td>
      <td>Pants</td>
      <td>124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>847</td>
      <td>47</td>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to th...</td>
      <td>5</td>
      <td>1</td>
      <td>6</td>
      <td>General</td>
      <td>Tops</td>
      <td>Blouses</td>
      <td>192</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_feat = clothes_reviews[['Title','Review Text','Sentiment','Review Length']]
text_feat.head()
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
      <th>Title</th>
      <th>Review Text</th>
      <th>Sentiment</th>
      <th>Review Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Absolutely wonderful - silky and sexy and comf...</td>
      <td>1</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Love this dress!  it's sooo pretty.  i happene...</td>
      <td>1</td>
      <td>303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Some major design flaws</td>
      <td>I had such high hopes for this dress and reall...</td>
      <td>0</td>
      <td>500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>My favorite buy!</td>
      <td>I love, love, love this jumpsuit. it's fun, fl...</td>
      <td>1</td>
      <td>124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Flattering shirt</td>
      <td>This shirt is very flattering to all due to th...</td>
      <td>1</td>
      <td>192</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_feat['Review'] = text_feat['Title'] + ' ' + text_feat['Review Text']
text_feat = text_feat.drop(labels=['Title','Review Text'], axis =1)
text_feat.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>303</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>Some major design flaws I had such high hopes ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>My favorite buy! I love, love, love this jumps...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>Flattering shirt This shirt is very flattering...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dropping null values 
text_feat.Review.isna().sum()
```




    2966




```python
text_feat = text_feat[~text_feat.Review.isna()]
print("My data's shape is:", text_feat.shape)
text_feat.head()
```

    My data's shape is: (19675, 3)





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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>Some major design flaws I had such high hopes ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>My favorite buy! I love, love, love this jumps...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>Flattering shirt This shirt is very flattering...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>Not for the very petite I love tracy reese dre...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>Cagrcoal shimmer fun I aded this in my basket ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_feat['Sentiment'].value_counts(normalize=True)
```




     1    0.770521
     0    0.125235
    -1    0.104244
    Name: Sentiment, dtype: float64




```python
#plotting text length
text_feat['Review Length'].plot(bins=20, kind='hist')
text_feat['Review Length'].describe()
```




    count    19675.000000
    mean       318.353698
    std        142.282429
    min          9.000000
    25%        199.000000
    50%        315.000000
    75%        475.000000
    max        508.000000
    Name: Review Length, dtype: float64




    
![png](output_21_1.png)
    



```python
text_feat.hist(column='Review Length', by='Sentiment', bins=50)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001B212927DC8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001B212955E08>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x000001B21298D388>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x000001B2129BE9C8>]],
          dtype=object)




    
![png](output_22_1.png)
    



```python
sns.set(rc={'figure.figsize':(15,5)})
sns.distplot(text_feat['Review Length'] ,hist=True, bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b219565f88>




    
![png](output_23_1.png)
    



```python
df_pos = text_feat[text_feat['Sentiment']==1]
df_neut = text_feat[text_feat['Sentiment']==0]
df_neg = text_feat[text_feat['Sentiment']==-1]
```


```python
sns.distplot(df_pos[['Review Length']] ,hist=False)
sns.distplot(df_neut[['Review Length']], hist=False)
sns.distplot(df_neg[['Review Length']] ,hist=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b2197d58c8>




    
![png](output_25_1.png)
    


def count_exclamation_mark(string_text):
    count = 0
    for char in string_text:
        if char == '!':
            count += 1
    return count

text_feat['Count Exc'] = text_feat['Review'].apply(count_exclamation_mark)
text_feat.head(5)


```python
text_feat['Polarity'] = text_feat['Review'].apply(lambda Review: TextBlob(Review).sentiment.polarity)
text_feat.head(5)
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>Some major design flaws I had such high hopes ...</td>
      <td>0.073209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>My favorite buy! I love, love, love this jumps...</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>Flattering shirt This shirt is very flattering...</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>Not for the very petite I love tracy reese dre...</td>
      <td>0.181111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>Cagrcoal shimmer fun I aded this in my basket ...</td>
      <td>0.157500</td>
    </tr>
  </tbody>
</table>
</div>




```python
text_feat['Polarity'].plot(kind='hist', bins=100)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b21263b348>




    
![png](output_29_1.png)
    


***

## DATA PREPROCESSING 

***


```python
# all lower case
text_feat['Review']=text_feat['Review'].str.lower()
```


```python
string.punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
# removing punctuations 
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str
```


```python
text_feat['Review'] = text_feat['Review'].apply(punctuation_removal)
text_feat['Review'].head()
```




    2    some major design flaws i had such high hopes ...
    3    my favorite buy i love love love this jumpsuit...
    4    flattering shirt this shirt is very flattering...
    5    not for the very petite i love tracy reese dre...
    6    cagrcoal shimmer fun i aded this in my basket ...
    Name: Review, dtype: object




```python
# part of speech tag 
def adj_collector(review_string):
    new_string=[]
    review_string = word_tokenize(review_string)
    tup_word = nltk.pos_tag(review_string)
    for tup in tup_word:
        if 'VB' in tup[1] or tup[1]=='JJ':  #Verbs and Adjectives
            new_string.append(tup[0])  
    return ' '.join(new_string)
```


```python
text_feat['Review'] = text_feat['Review'].apply(adj_collector)
text_feat['Review'].head(5)
```




    2    major had such high wanted work i ordered smal...
    3         favorite love love fabulous wear i get great
    4    flattering is flattering due adjustable is per...
    5    petite love reese is petite am tall wear was i...
    6    aded last see look i went am pale is gorgeous ...
    Name: Review, dtype: object



***
***


```python
# removing stopwords
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
```




    "i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"




```python
stop = stopwords.words('english')
stop.append("i'm")
```


```python
stop_words = []

for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item)
print(stop_words[::10])
```

    ['i', 'youve', 'himself', 'they', 'that', 'been', 'a', 'while', 'through', 'in', 'here', 'few', 'own', 'just', 're', 'doesn', 'ma', 'shouldnt']



```python
# adding to stopwords list 
clothes_list =['dress', 'top','sweater','shirt',
               'skirt','material', 'white', 'black',
              'jeans', 'fabric', 'color','order', 'wear']
```


```python
def stopwords_removal(messy_str):
    messy_str = word_tokenize(messy_str)
    return [word.lower() for word in messy_str 
            if word.lower() not in stop_words and word.lower() not in clothes_list ]
```

# removing eights, size, and other numbers
def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)

text_feat['Review'] = text_feat['Review'].apply(drop_numbers)
text_feat['Review'].head()


```python
# Removing frequent words
from collections import Counter
cnt = Counter()
for text in text_feat['Review'].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)
```




    [('is', 27836),
     ('i', 16347),
     ('was', 11058),
     ('have', 7346),
     ('love', 7251),
     ('great', 7063),
     ('are', 6596),
     ('be', 6377),
     ('wear', 5389),
     ('am', 5189)]




```python
FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

text_feat['Review'] = text_feat['Review'].apply(lambda text: remove_freqwords(text))
text_feat.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>major had such high wanted work ordered small ...</td>
      <td>0.073209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>favorite fabulous get</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>flattering flattering due adjustable perfect p...</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>petite reese petite tall long full overwhelmed...</td>
      <td>0.181111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>aded last see look went pale gorgeous turns ma...</td>
      <td>0.157500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop the two columns which are no more needed 
#text_feat.drop(['text_stemmed','text_lemmatized','text_wo_stopfreq'], axis=1, inplace=True)

n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])

text_feat['Review'] = text_feat['Review'].apply(lambda text: remove_rarewords(text))
text_feat.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>major had such high wanted work ordered small ...</td>
      <td>0.073209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>favorite fabulous get</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>flattering flattering due adjustable perfect p...</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>petite reese petite tall long full overwhelmed...</td>
      <td>0.181111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>aded last see look went pale gorgeous turns ma...</td>
      <td>0.157500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# stemming
from nltk.stem.porter import PorterStemmer

# Drop the two columns 
#text_feat.drop(["text_wo_stopfreq", "text_wo_stopfreqrare"], axis=1, inplace=True) 

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

text_feat['Review'] = text_feat['Review'].apply(lambda text: stem_words(text))
text_feat.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>major had such high want work order small usua...</td>
      <td>0.073209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>favorit fabul get</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>flatter flatter due adjust perfect pair cardigan</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>petit rees petit tall long full overwhelm smal...</td>
      <td>0.181111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>ade last see look went pale gorgeou turn mathc...</td>
      <td>0.157500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lammatizing 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

text_feat['Review'] = text_feat['Review'].apply(lambda text: lemmatize_words(text))
text_feat.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>major had such high want work order small usua...</td>
      <td>0.073209</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>favorit fabul get</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>flatter flatter due adjust perfect pair cardigan</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>petit rees petit tall long full overwhelm smal...</td>
      <td>0.181111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>ade last see look went pale gorgeou turn mathc...</td>
      <td>0.157500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# WordCloud
```


```python
pos_df = text_feat[text_feat.Sentiment== 1]
neut_df = text_feat[text_feat.Sentiment== 1]
neg_df = text_feat[text_feat.Sentiment== 0]
neut_df.head(5)
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Review</th>
      <th>Polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>favorit fabul get</td>
      <td>0.560714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>flatter flatter due adjust perfect pair cardigan</td>
      <td>0.512891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>ade last see look went pale gorgeou turn mathc...</td>
      <td>0.157500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>482</td>
      <td>goe order had tri use top pair went nice went ...</td>
      <td>0.230342</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>166</td>
      <td>flatter get run littl order flatter feminin usual</td>
      <td>0.002500</td>
    </tr>
  </tbody>
</table>
</div>




```python
pos_words =[]
neut_words=[]
neg_words = []

for review in pos_df.Review:
    pos_words.append(review) 
pos_words = ' '.join(pos_words)
pos_words[:30]

for review in neut_df.Review:
    neut_words.append(review) 
neut_words = ' '.join(neut_words)
neut_words[:30]

for review in neg_df.Review:
    neg_words.append(review)
neg_words = ' '.join(neg_words)
neg_words[:100]
```




    'major had such high want work order small usual found small small zip reorder petit ok overal top co'




```python
# wordcloud for positive sentiment

wordcloud = WordCloud().generate(pos_words)

wordcloud = WordCloud(background_color="white",max_words=len(pos_words),\
                      max_font_size=40, relative_scaling=.5, colormap='summer').generate(pos_words)
plt.figure(figsize=(13,13))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


    
![png](output_53_0.png)
    



```python
# wordcloud for neutral sentiment

wordcloud = WordCloud().generate(neut_words)

wordcloud = WordCloud(background_color="white",max_words=len(pos_words),\
                      max_font_size=40, relative_scaling=.5, colormap='winter').generate(neut_words)
plt.figure(figsize=(13,13))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


    
![png](output_54_0.png)
    



```python
# wordcloud for negative sentiment
wordcloud = WordCloud().generate(neg_words)

wordcloud = WordCloud(background_color="white",max_words=len(neg_words),\
                      max_font_size=40, relative_scaling=.5, colormap='gist_heat').generate(neg_words)
plt.figure(figsize=(13,13))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```


    
![png](output_55_0.png)
    



```python
import anvil.server
```


```python
anvil.server.connect("YESFY5MNJDDERL4GUEZUXFH4-MU4DV2L533AQVBHS")
```

    Connecting to wss://anvil.works/uplink
    Anvil websocket open
    Connected to "Default environment (dev)" as SERVER



```python

```


```python

```


```python
#vectorizing bag of words
def text_vectorizing_process(sentence_string):
    return [word for word in sentence_string.split()]
```


```python
bow_transformer = CountVectorizer(text_vectorizing_process)
```


```python
bow_transformer.fit(text_feat['Review'])
```




    CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                    dtype=<class 'numpy.int64'>, encoding='utf-8',
                    input=<function text_vectorizing_process at 0x000001B221774798>,
                    lowercase=True, max_df=1.0, max_features=None, min_df=1,
                    ngram_range=(1, 1), preprocessor=None, stop_words=None,
                    strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                    tokenizer=None, vocabulary=None)




```python
Reviews = bow_transformer.transform(text_feat['Review'])
Reviews
```




    <19675x7409 sparse matrix of type '<class 'numpy.int64'>'
    	with 260302 stored elements in Compressed Sparse Row format>




```python
print('Shape of Sparse Matrix', Reviews.shape)
print('Amount of Non-Zero occurences:', Reviews.nnz)
```

    Shape of Sparse Matrix (19675, 7409)
    Amount of Non-Zero occurences: 260302



```python
tfidf_transformer = TfidfTransformer().fit(Reviews)
```


```python
[i for i in bow_transformer.vocabulary_.items() if i[1]==3510]
```




    [('loft', 3510)]




```python
[i for i in bow_transformer.vocabulary_.items()][6:60:10]
```




    [('order', 4305),
     ('comfort', 1191),
     ('get', 2554),
     ('full', 2475),
     ('look', 3544),
     ('decid', 1532)]




```python
reviews_tfidf = tfidf_transformer.transform(Reviews)
```


```python
reviews_tfidf = messages_tfidf.toarray()
reviews_tfidf = pd.DataFrame(reviews_tfidf)
print(reviews_tfidf.shape)
reviews_tfidf.head()
```

    (19675, 7409)





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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>7399</th>
      <th>7400</th>
      <th>7401</th>
      <th>7402</th>
      <th>7403</th>
      <th>7404</th>
      <th>7405</th>
      <th>7406</th>
      <th>7407</th>
      <th>7408</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.20611</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7409 columns</p>
</div>




```python
text_feat_all = pd.merge(text_feat.drop(columns='Review'), reviews_tfidf, 
                  left_index=True, right_index=True )
text_feat_all.head()
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
      <th>Sentiment</th>
      <th>Review Length</th>
      <th>Polarity</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>...</th>
      <th>7399</th>
      <th>7400</th>
      <th>7401</th>
      <th>7402</th>
      <th>7403</th>
      <th>7404</th>
      <th>7405</th>
      <th>7406</th>
      <th>7407</th>
      <th>7408</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>500</td>
      <td>0.073209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>124</td>
      <td>0.560714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>192</td>
      <td>0.512891</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1</td>
      <td>488</td>
      <td>0.181111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>496</td>
      <td>0.157500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7412 columns</p>
</div>




```python
#Splitting the data
X = text_feat_all.drop('Sentiment', axis=1)
y = text_feat_all.Sentiment

X.head()
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
      <th>Review Length</th>
      <th>Polarity</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>...</th>
      <th>7399</th>
      <th>7400</th>
      <th>7401</th>
      <th>7402</th>
      <th>7403</th>
      <th>7404</th>
      <th>7405</th>
      <th>7406</th>
      <th>7407</th>
      <th>7408</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>500</td>
      <td>0.073209</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>124</td>
      <td>0.560714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>192</td>
      <td>0.512891</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>488</td>
      <td>0.181111</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>496</td>
      <td>0.157500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7411 columns</p>
</div>




```python
X.describe()
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
      <th>Review Length</th>
      <th>Polarity</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>...</th>
      <th>7399</th>
      <th>7400</th>
      <th>7401</th>
      <th>7402</th>
      <th>7403</th>
      <th>7404</th>
      <th>7405</th>
      <th>7406</th>
      <th>7407</th>
      <th>7408</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.0</td>
      <td>16481.0</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>...</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
      <td>16481.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>318.302834</td>
      <td>0.265987</td>
      <td>0.000023</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000022</td>
      <td>0.000022</td>
      <td>0.000103</td>
      <td>0.000052</td>
      <td>0.000024</td>
      <td>...</td>
      <td>0.000022</td>
      <td>0.000057</td>
      <td>0.000074</td>
      <td>0.000070</td>
      <td>0.000050</td>
      <td>0.000021</td>
      <td>0.002029</td>
      <td>0.000302</td>
      <td>0.000030</td>
      <td>0.000123</td>
    </tr>
    <tr>
      <th>std</th>
      <td>142.308680</td>
      <td>0.172803</td>
      <td>0.002964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.002802</td>
      <td>0.002879</td>
      <td>0.006782</td>
      <td>0.004788</td>
      <td>0.003071</td>
      <td>...</td>
      <td>0.002843</td>
      <td>0.005280</td>
      <td>0.004819</td>
      <td>0.005175</td>
      <td>0.004512</td>
      <td>0.002671</td>
      <td>0.026215</td>
      <td>0.010453</td>
      <td>0.003801</td>
      <td>0.007148</td>
    </tr>
    <tr>
      <th>min</th>
      <td>9.000000</td>
      <td>-0.987500</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>199.000000</td>
      <td>0.158333</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>315.000000</td>
      <td>0.260863</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>475.000000</td>
      <td>0.370312</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>508.000000</td>
      <td>1.000000</td>
      <td>0.380549</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.359753</td>
      <td>0.369649</td>
      <td>0.541592</td>
      <td>0.500109</td>
      <td>0.394249</td>
      <td>...</td>
      <td>0.364975</td>
      <td>0.560754</td>
      <td>0.367715</td>
      <td>0.403128</td>
      <td>0.442586</td>
      <td>0.342937</td>
      <td>0.631888</td>
      <td>0.476824</td>
      <td>0.488008</td>
      <td>0.499975</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 7411 columns</p>
</div>




```python
X_train, X_test, y_train, y_test = split(X,y, test_size=0.2, stratify=y, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((13184, 7411), (3297, 7411), (13184,), (3297,))




```python
y_train.value_counts(normalize=True)
```




     1    0.769645
     0    0.127124
    -1    0.103231
    Name: Sentiment, dtype: float64




```python
y_test.value_counts(normalize=True)
```




     1    0.769791
     0    0.127085
    -1    0.103124
    Name: Sentiment, dtype: float64




```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
# pd.DataFrame(X_train_scaled,columns= X_train.columns).describe()
```


```python
pca_transformer = PCA(n_components=2).fit(X_train_scaled)
X_train_scaled_pca = pca_transformer.transform(X_train_scaled)
X_test_scaled_pca = pca_transformer.transform(X_test_scaled)
X_train_scaled_pca[:1]
```




    array([[-0.27545987, -0.01901534]])




```python
plt.figure(figsize=(15,7))
sns.scatterplot(x=X_train_scaled_pca[:, 0], 
                y=X_train_scaled_pca[:, 1], 
                hue=y_train, 
                sizes=100,
                palette="summer") 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b219f41f88>




    
![png](output_79_1.png)
    



```python
X_train_scaled = scipy.sparse.csr_matrix(X_train_scaled)
X_test_scaled = scipy.sparse.csr_matrix(X_test_scaled)

X_train = scipy.sparse.csr_matrix(X_train.values)
X_test = scipy.sparse.csr_matrix(X_test.values)
X_test
```




    <3297x7411 sparse matrix of type '<class 'numpy.float64'>'
    	with 50275 stored elements in Compressed Sparse Row format>




```python
# Model
def report(y_true, y_pred, labels):
    cm = pd.DataFrame(confusion_matrix(y_true=y_true, y_pred=y_pred), 
                                        index=labels, columns=labels)
    rep = classification_report(y_true=y_true, y_pred=y_pred)
    return (f'Confusion Matrix:\n{cm}\n\nClassification Report:\n{rep}')
```


```python
# SVM model
svc_model = SVC(C=1.0, 
             kernel='linear',
             class_weight='balanced', 
             probability=True,
             random_state=111)
svc_model.fit(X_train_scaled, y_train)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
        max_iter=-1, probability=True, random_state=111, shrinking=True, tol=0.001,
        verbose=False)




```python
test_predictions = svc_model.predict(X_test_scaled)
print(report(y_test, test_predictions, svc_model.classes_ ))
```

    Confusion Matrix:
         -1    0     1
    -1  155  101    84
     0  141  120   158
     1  413  494  1631
    
    Classification Report:
                  precision    recall  f1-score   support
    
              -1       0.22      0.46      0.30       340
               0       0.17      0.29      0.21       419
               1       0.87      0.64      0.74      2538
    
        accuracy                           0.58      3297
       macro avg       0.42      0.46      0.42      3297
    weighted avg       0.71      0.58      0.63      3297
    



```python
import scikitplot as skplt
```


```python
skplt.metrics.plot_roc(y_test, svc_model.predict_proba(X_test_scaled)) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b219e6a788>




    
![png](output_85_1.png)
    



```python
# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=5, 
                                  class_weight='balanced', random_state=3)
rf_model.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                           criterion='gini', max_depth=5, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=1000,
                           n_jobs=None, oob_score=False, random_state=3, verbose=0,
                           warm_start=False)




```python
test_predictions = rf_model.predict(X_test)
print(report(y_test, test_predictions, rf_model.classes_ ))
```

    Confusion Matrix:
         -1    0     1
    -1  188   72    80
     0  165   82   172
     1  387  220  1931
    
    Classification Report:
                  precision    recall  f1-score   support
    
              -1       0.25      0.55      0.35       340
               0       0.22      0.20      0.21       419
               1       0.88      0.76      0.82      2538
    
        accuracy                           0.67      3297
       macro avg       0.45      0.50      0.46      3297
    weighted avg       0.73      0.67      0.69      3297
    



```python
skplt.metrics.plot_roc(y_test, rf_model.predict_proba(X_test), 
                       title='ROC Curves - Random Forest') 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b221ad1748>




    
![png](output_88_1.png)
    

