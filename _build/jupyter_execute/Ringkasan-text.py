#!/usr/bin/env python
# coding: utf-8

# # Ringkasan-text

# BeautifulSoup adalah library Python yang digunakan untuk mengambil data HTML dan XML. BeautifulSoup berfungsi sebagai parser untuk memisahkan komponen-komponen HTML menjadi rangkain elemen yang mudah dibaca.

# In[1]:


pip install beautifulsoup4


# In[2]:


from urllib.request import urlopen
from bs4 import BeautifulSoup


# import port libraray pandas


import pandas as pd


# buat variabel yang berisi tujuan link yang akan di crawling


alamat = "https://news.kompas.com/"
html = urlopen(alamat)
data = BeautifulSoup(html, 'html.parser')


# pemrosesan data yang akan di crawling


def kompas ():
    items = data.findAll("h4", {"class":"most__title"})
    hasil = [item.get_text() for item in items]
    df = pd.DataFrame(hasil, columns=['Judul Berita Populer'])
    print(df)


# menampilkan data yang telah di hasilkan


kompas()

