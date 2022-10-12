#!/usr/bin/env python
# coding: utf-8

# # data-crawling-twitter
# 
# Crawling adalah semacam pengambilan data dari media sosial kemudian di kumpulkan menjadi satu untuk di evakuasi dan di bentuk agar menjadi sebuah penelitian.
# 
# Prosesnya cukup mudah tergantung kamu ingin mengambil data dari sosial media mana. Misalkan kamu ingin crawling data dari twitter ada dua cara yaitu dengan menggunakan API dan tanpa API.
# 
# Data yang dapat kamu kumpulkan dapat berupa text, audio, video, dan gambar. Kamu dapat memulai dengan melakukan penambangan data pada API yang bersifat open source seperti yang disediakan oleh Twitter.
# 
# Untuk yang tools yang di gunakan untuk crawling data twitter kali ini kita akan menggunakan beberapa. Apa aja itu ? Simak berikut ini.
# 
#   1. Jupyter Notebook
#   2. Python
#   
# Sudah itu saja tools yang di rekomendasikan artikel ini untuk crawling data twitter.

#  # Twint
# 
#  selanjutnya kita menggunakan twint nah pasti banyak yang belom tahu apa itu twint.
# 
#  Twint adalah alat pengikis Twitter canggih yang ditulis dengan Python yang memungkinkan untuk mengambil data Tweet dari profil Twitter tanpa menggunakan API Twitter.
# 
# # Manfaat Twint
# 
# Twint kemampuan menarik data Tweet tanpa batas. Semangat pembuatan Twint lahir untuk mengakali terbatas dan mahalnya layanan Twitter API. Keampuhan Twint terletak pada kemampuannya menarik tweet dari berbagai model seperti akun, hastag, periode.
# 
# 
# 

# In[1]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# In[ ]:


get_ipython().system('pip install twint')


# kalau pada saat import twint mengalami error mungkin bisa tambahkan code berikut ini, kemudian running lah.

# In[ ]:


pip install nest-asyncio


# In[ ]:


get_ipython().system('pip install aiohttp==3.7.0')


# Setelah menginstal twint tentunya belum bisa di jalankan karena kamu perlu menginportnya terlebih dahulu supaya twint bisa digunakan.
# 
# Caranya cukup mudah masukan saja kode di bawah ini.

# In[ ]:


import twint


# ### Install Nest Asyncio dan lakukan Import
# 
# Nest Asyncio Secara opsional, loop spesifik yang perlu ditambal dapat diberikan sebagai argumen untuk diterapkan , jika tidak, loop peristiwa saat ini digunakan. Loop peristiwa dapat ditambal apakah sudah berjalan atau belum. Hanya loop acara dari asyncio yang dapat ditambal; Loop dari proyek lain, seperti uvloop atau quamash, umumnya tidak dapat ditambal.

# In[ ]:


get_ipython().system('pip install nest_asyncio')
import nest_asyncio
nest_asyncio.apply()


# # Proses Ambil Data Twitter
# Sudah import semuanya ? kini kamu tinggal menentukan data apa yang ingin di ambil. Kita ambil saja salah satu hastag yang sedang viral di twitter yaitu 
# 
# 

# Kemudian running kode tersebut maka akan muncul semua data yang ada di twitter khusunya yang berhastag 
# 
# Kamu bisa juga mengeksport semua data tersebut ke csv. caranya kamu masukan kode di bawah ini untuk membuat tabelnya terlebih dahulu.
# 
# 

# Kemudian masukan kode di bawah ini untuk mengeksport ke dalam file berekstensi csv.

# In[ ]:


c = twint.Config()
c.Search = '#ganjarpranowo'
c.Pandas = True
c.Limit = 60
c.Store_csv = True
c.Custom["tweet"] = ["tweet"]
c.Output = "dataGanjar.csv"
twint.run.Search(c)


# # Pandas
# Pandas adalah paket Python open source yang paling sering dipakai untuk menganalisis data serta membangun sebuah machine learning. Pandas dibuat berdasarkan satu package lain bernama Numpy
# 
# menurut wekipedia pandas adalah perpustakaan perangkat lunak yang ditulis untuk bahasa pemrograman Python untuk manipulasi dan analisis data. Secara khusus, ia menawarkan struktur data dan operasi untuk memanipulasi tabel numerik dan deret waktu. Ini adalah perangkat lunak gratis yang dirilis di bawah lisensi BSD tiga klausa.

# In[ ]:


import pandas as pd


# data pd read pandas digunakan untuk mengecek data apakah data sudah ada

# In[ ]:


data = pd.read_csv('dataGanjar.csv')
data


# # NTLK
# Natural Language Toolkit, atau lebih umum NLTK, adalah rangkaian perpustakaan dan program untuk pemrosesan bahasa alami simbolis dan statistik untuk bahasa Inggris yang ditulis dalam bahasa pemrograman Python.
# 
# # SASTRAWI
# Sastrawi adalah library Python sederhana yang memungkinkan Anda untuk mereduksi kata-kata infleksi dalam Bahasa Indonesia (Bahasa Indonesia) ke bentuk dasarnya ( stem ).

# In[ ]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install Sastrawi')


# ### Re module Python menyediakan seperangkat fungsi yang memungkinkan kita untuk mencari sebuah string untuk match (match).

# In[ ]:


import pandas as pd
import re
import numpy as np

import nltk
nltk.download('punkt')
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Selanjutnya membuat Function Remove Stopwords yang fungsinya adalah menghapus kata-kata yang tidak diperlukan dalam proses nantinya,sehingga dapat mempercepat proses VSM

# In[ ]:


def remove_stopwords(text):
    with open('/content/drive/MyDrive/webmining/tugas/contents/stopwords.txt') as f:
        stopwords = f.readlines()
        stopwords = [x.strip() for x in stopwords]
    
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stopwords]
                     
    return text


# Steming merupakan proses mengubah kata dalam bahasa Indonesia ke akar katanya misalkan ‘Mereka meniru-nirukannya’ menjadi ‘mereka tiru’

# In[ ]:


def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    result = [stemmer.stem(word) for word in text]
    
    return result


# Selanjutnya tahap preprocessing,untuk tahap ini ada beberapa proses seperti:
# 
# 1.Mengubah Text menjadi huruf kecil
# 
# 2.Menghapus Kata non Ascii
# 
# 4.Menghapus Hastag,Link dan Mention
# 
# 5.Mengubah/menghilangkan tanda (misalkan garis miring menjadi spasi)
# 
# 6.Melakukan tokenization kata dan Penghapusan Kata yang tidak digunakan
# 
# 7.Memfilter kata dari tanda baca
# 
# 8.Mengubah kata dalam bahasa Indonesia ke akar katanya
# 
# 9.Menghapus String kosong

# In[ ]:


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


# In[ ]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/tugas/contents')


# In[ ]:


#data['tweet'].apply(preprocessing).to_excel('preprocessing.xlsx')


# Tokenizing adalah proses pemisahan teks menjadi potongan-potongan yang disebut sebagai token untuk kemudian di analisa. Kata, angka, simbol, tanda baca dan entitas penting lainnya dapat dianggap sebagai token.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
dataTextPre = pd.read_excel('/content/drive/MyDrive/webmining/tugas/contents/preprocessing.xlsx')
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre


# Melihat Jumlah Baris dan Kata

# In[ ]:


matrik_vsm=bag.toarray()
matrik_vsm.shape


# 

# In[ ]:


matrik_vsm[0]


# In[ ]:


a=vectorizer.get_feature_names()


# Tampilan data VSM dengan labelnya

# In[ ]:


dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF


# lalu data diatas ditambahkan dengan label (positif,netral dan negatif)

# In[ ]:


label = pd.read_csv('/content/drive/MyDrive/webmining/tugas/twint/dataGanjar.csv')
label
dj = pd.concat([dataTF.reset_index(), label["tweet"]], axis=1)
dj


# # Fungsi UNIQUE
# Mengembalikan baris unik dalam rentang sumber yang diberikan, dengan membuang duplikat. Baris dikembalikan sesuai urutan saat muncul pertama kali dalam rentang sumber.

# In[ ]:


dj['tweet'].unique()


# In[ ]:


dj.info()


# # Scikit Learn
# Scikit Learn difokuskan pada Machine Learning, misalnya pemodelan data. Ini tidak melihat bagaimana proses pemuatan, penanganan, manipulasi, dan visualisasi data. Dengan demikian, merupakan praktik yang wajar dan umum untuk menggunakan pustaka di atas, terutama NumPy, untuk langkah-langkah ekstra tersebut; mereka dibuat untuk satu sama lain dan saling melengkapi.
# 
# Adapun scikit-learn lebih berfokus pada algoritma machine learning. Serangkaian penawaran algoritma Scikit-Learn yang kuat mencakup:
# 
# 1. Regresi: Memasang model linier dan non-linier
# 
# 2. Classification : Klasifikasi tanpa pengawasan
# 
# 3. Decision Tree : Induksi dan pemangkasan pohon untuk tugas klasifikasi dan regresi
# 
# 4. Neural Networks : Pelatihan ujung ke ujung untuk klasifikasi dan regresi. Lapisan dapat dengan mudah ditentukan dalam tupel
# 
# 6. SVM: untuk mempelajari batasan keputusan
# 
# 7. Naive Bayes: Pemodelan probabilistik langsung

# In[ ]:


get_ipython().system('pip install -U scikit-learn')


# # Information Gain
# Information Gain, atau singkatnya IG, mengukur pengurangan entropi atau kejutan dengan membagi kumpulan data menurut nilai tertentu dari variabel acak.
# 
# Keuntungan informasi yang lebih besar menunjukkan kelompok entropi yang lebih rendah atau kelompok sampel, dan karenanya kurang mengejutkan.
# 
# Anda mungkin ingat bahwa informasi mengukur seberapa mengejutkan suatu peristiwa dalam bit. Kejadian dengan probabilitas yang lebih rendah memiliki lebih banyak informasi, kejadian dengan probabilitas yang lebih tinggi memiliki lebih sedikit informasi. Entropi mengkuantifikasi berapa banyak informasi yang ada dalam variabel acak, atau lebih khusus lagi distribusi probabilitasnya. Distribusi miring memiliki entropi rendah, sedangkan distribusi di mana peristiwa memiliki probabilitas yang sama memiliki entropi yang lebih besar.
# 
# Dalam teori informasi, kami ingin menggambarkan " kejutan " dari suatu peristiwa. Peristiwa probabilitas rendah lebih mengejutkan karena itu memiliki jumlah informasi yang lebih besar. Sedangkan distribusi probabilitas dimana kejadiannya sama-sama mungkin lebih mengejutkan dan memiliki entropi yang lebih besar.
# 
# sekarang mari kita lanjutkan kode pythonya

# In[ ]:


from sklearn.model_selection import train_test_split
#membagi kumpulan data menjadi data pelatihan dan data pengujian.
X_train,X_test,y_train,y_test=train_test_split(dj.drop(labels=['tweet'], axis=1),
    dj['tweet'],
    test_size=0.3,
    random_state=0)


# In[ ]:


X_train.info(verbose=True)


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X_train, y_train)
mutual_info

