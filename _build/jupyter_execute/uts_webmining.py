#!/usr/bin/env python
# coding: utf-8

# # UTS WEBMING

# 1. Lakukan analisa clustering dengan menggunakan k-mean clustering pada data twitter denga kunci pencarian " tragedi kanjuruhan"
# 
# 2. Lakukan peringkasan dokumen dari berita online ( link berita bebas) menggunakan metode pagerank
# 
# 

# In[1]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# In[2]:


get_ipython().system('pip install twint')


# In[3]:


get_ipython().system('pip install nest-asyncio')


# In[4]:


get_ipython().system('pip install aiohttp==3.7.0')


# In[5]:


import twint


# In[6]:


get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply()


# In[7]:


c = twint.Config()
c.Search = 'tragedi kanjuruhan'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataKanjuruhan.csv"
twint.run.Search(c)


# In[8]:


import pandas as pd


# In[9]:


data = pd.read_csv('dataKanjuruhan.csv')
data


# In[10]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_csv('dataKanjuruhan.csv')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# In[11]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# In[12]:


matrik_vsm[0]


# In[13]:


a=vectorizer.get_feature_names()


# In[15]:


from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# In[16]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# In[17]:


label = pd.read_csv('dataKanjuruhan.csv')
dj = pd.concat([dataTF.reset_index(), label["tweet"]], axis=1)
dj


# In[18]:


dj['tweet'].unique()


# In[19]:


get_ipython().system('pip install -U scikit-learn')


# In[ ]:


from sklearn.model_selection import train_test_split
#membagi kumpulan data menjadi data pelatihan dan data pengujian.
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['tweet'], axis=1),
    dj['tweet'],
    test_size=0.3,
    random_state=0)


# In[21]:


#import plt
import matplotlib.pyplot as plt
#import metrics
from sklearn import metrics


# In[22]:


# from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
gauss = GaussianNB()
gauss.fit(X_train, y_train)


# In[23]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# In[ ]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# In[ ]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/tugas/contents/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# In[ ]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# In[27]:


def preprocessing(text):
    #case folding
    text = text.lower()

    #remove non ASCII (emoticon, chinese word, .etc)
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")

    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')

    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())

    #replace weird characters
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    text = text.replace('-', ' ')

    #tokenization and remove stopwords
    text = remove_stopwords(text)

    #remove punctuation    
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]  

    #stemming
    text = stemming(text)

    #remove empty string
    text = list(filter(None, text))
    return text


# In[28]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/tugas/contents')


# In[29]:


#data['tweet'].apply(preprocessing).to_excel('preprocessing.xlsx')


# In[30]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_excel('/content/drive/MyDrive/webmining/tugas/contents/preprocessing.xlsx')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre

