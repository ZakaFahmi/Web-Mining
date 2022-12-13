#!/usr/bin/env python
# coding: utf-8

# #**ensembleLearning2**
# 

# Ensemble Learning
# Ensemble Learning adalah proses di mana beberapa model, seperti pengklasifikasi atau ahli, secara strategis dihasilkan dan digabungkan untuk memecahkan kecerdasan komputasi tertentu. Ensemble Learning utamanya digunakan untuk meningkatkan (klasifikasi, prediksi, perkiraan fungsi, dll.) kinerja model, atau mengurangi kemungkinan pemilihan model yang buruk.
# 
# **pertama Install library yang di butuhkan**

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install snscrape\n!pip install pandas\n!pip install Sastrawi\n!pip install scikit-learn')


# ## **Menulis Script Konfigurasi Snscrape**

# In[2]:



import snscrape.modules.twitter as sntwitter
import pandas as pd
# from google.colab import data_table
# data_table.enable_dataframe_formatter()

search_query = "dedi mulyadi"
jumlah_tweets = 100
tweets = []


for tweet in sntwitter.TwitterSearchScraper(search_query).get_items():
    
    if len(tweets) == jumlah_tweets:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content, 'None'])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Label'])
print(df)


# ### Melakukan Pengecekan Berkas Hasil Scrape Lama
# 
# Karena setelah melakukan scraping data tweet kita perlu memberikan label secara manual, maka untuk menghindari berkas lama tertimpa oleh berkas baru, disini kita akan melakukan pengecekan berkas hasil scrape lama, apakah ada pada direktori atau tidak. Jika berkas hasil scrape lama tidak ditemukan pada direktori, maka hasil scrape sebelumnya (yang ada pada Pandas Dataframe) akan diespor menjadi berkas csv.
# 

# In[3]:


get_ipython().run_cell_magic('capture', '', "import os\n\n\n\noutput_stream = os.popen('tweets_labeled.csv')\nres = output_stream.read()\nif res == '':\n  df.to_csv('tweets_labeled.csv')")


# ## **Import Hasil**

# In[4]:


import pandas as pd


# pd.options.mode.chained_assignment = None
# pd.options.display.max_colwidth = None
# pd.options.display.max_columns = None
# pd.options.display.max_rows = None
df = pd.read_csv('tweets_labeled.csv', usecols=['User', 'Tweet', 'Label'])
df


# ## **Preprocessing Data Tweets**

# In[5]:


import string

#Mengapus link dan mention
df['Tweet'] = df['Tweet'].replace(r'\s+',' ', regex=True)
indx = 0
for i in df['Tweet']:
  temp = df['Tweet'][indx].split()
  for j in temp:
    if 'http' in j:
      df['Tweet'] = df['Tweet'].replace(r'%s'%j," ", regex=True)
    if '@' in j:
      df['Tweet'] = df['Tweet'].replace(r'%s'%j," ", regex=True)
  indx+=1



#mengubah menjadi huruf kecil
df['Tweet'] = df['Tweet'].str.lower()

#menghapus tanda baca
for char in string.punctuation:
    df['Tweet'] = df['Tweet'].replace(r'[\%s]'%char," ", regex=True)

#menghapus angka
df['Tweet'] = df['Tweet'].replace(r'\d+',' ', regex=True)

#menghapus karakter kosong
df['Tweet'] = df['Tweet'].replace(r'\s+',' ', regex=True)
df['Tweet'][2]


# ### Stopwords Removal

# In[6]:


import urllib.request, json 
with urllib.request.urlopen("https://raw.githubusercontent.com/smilesense/stopwords-id/master/stopwords-id.json") as list_stopwords:
    data_stopword = json.load(list_stopwords)

for i in data_stopword:
    df['Tweet'] = df['Tweet'].replace(r'\b%s\b'%i, '', regex=True)
df['Tweet'] = df['Tweet'].replace(r'\s+',' ', regex=True)

df['Tweet'][2]


# ### Stemming

# In[7]:


# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# stemming process

try:
  ind = 0
  for sentence in df['Tweet']:
      df['Tweet'][ind] = stemmer.stem(str(sentence))
      ind+=1
  raise KeyboardInterrupt
except KeyboardInterrupt:
  print(df['Tweet'][2])
else :
  print(df['Tweet'][2])


# ### Tokenisasi

# In[8]:


indeks = 0
df2 = df.copy()
for tweet in df2['Tweet']:
    df2['Tweet'][indeks] = str(tweet).split()
    indeks+=1

df2['Tweet'][2]


# ### Hasil Preprocessing
# 

# In[9]:


df2


# ## **Term Frequency (TF)**

# ### Term Frequency Keseluruhan

# In[10]:


listkata = []
for tweet in df['Tweet']:
    listkata = listkata + str(tweet).split()

listkata_nodup = list(dict.fromkeys(listkata))
hasil_hitung = {}
for test1 in listkata_nodup:
    jumlah = 0
    for test2 in range(len(listkata)):
        if test1 == listkata[test2]:
            jumlah+=1
    hasil_hitung.update({'%s'%test1 : jumlah})
    
hasil_hitung = dict(sorted(hasil_hitung.items(), key=lambda item: item[1], reverse=True))
print(hasil_hitung)


# ### Term Frequency Tiap Tweet

# In[11]:


def terms(dataframe):
  # pd.options.mode.chained_assignment = None
  # pd.options.display.max_colwidth = None
  # pd.options.display.max_columns = None
  # pd.options.display.max_rows = None

  df3 = dataframe.copy()
  a = 1
  for inter in range(len(df3['Tweet'])):
      for fitur in hasil_hitung:
          df3['%s'%(fitur)] = 0
          a+=1

  for inter in range(len(df3['Tweet'])):
    for fitur in hasil_hitung:
        cek = df3['Tweet'][inter]
        jumlah = 0
        for iter2 in range(len(cek)):
          if fitur == cek[iter2]:
            jumlah+=1
        df3['%s'%fitur][inter] = jumlah
        a+=1
  df3.to_csv(r'my_data.csv', index=False)
  return df3
terms(df2)


# ## Melakukan Training 

# In[61]:


df4 = pd.read_csv('my_data.csv')
df4['Label'].unique()


# In[62]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df4.drop(labels=['Label', 'User', 'Tweet'], axis=1),
    df4['Label'],
    test_size=0.3,
    random_state=0)
y_train


# ## **Bagging Classifier**

# In[63]:


X = df4.drop(labels=['Label', 'User', 'Tweet'], axis=1)
y = df4['Label']


# In[64]:


from sklearn import model_selection

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification

# kfold = model_selection.KFold(n_splits = 3,
#                        random_state = 5, shuffle=True)

X, y = make_classification(n_samples=100, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=True)
clf = BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=5).fit(X, y)
clf.predict([[0, 0, 0, 0]])

# results = model_selection.cross_val_score(clf, X, y, cv = kfold)
print("accuracy :")
# print(results.mean())
clf.score(X, y)


# ## **Random Forest**

# In[65]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=True)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
# RandomForestClassifier(...)
# print(clf.predict([[0, 0, 0, 0]]))
print("accuracy :")
clf.score(X, y)


# ####Naive Bayes

# In[66]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
y_pred = clf.fit(X_train, y_train).predict(X_test)
clf.fit(X_train, y_train).score(X_test, y_test)


# ## **Stacking Classifier**

# In[67]:


from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train).score(X_test, y_test)


# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import StackingClassifier
# 
# from sklearn.tree import DecisionTreeClassifier
# 
# estimators = [('rf', RandomForestClassifier(max_depth=2, random_state=0)),('rf1', RandomForestClassifier(max_depth=2, random_state=0)),('rf2', RandomForestClassifier(max_depth=2, random_state=0)),('rf3', RandomForestClassifier(max_depth=2, random_state=0))]
# clf = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
# from sklearn.model_selection import train_test_split
# print("accuracy :")
# clf.fit(X_train, y_train).score(X_test, y_test)
