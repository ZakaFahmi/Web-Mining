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

# In[1]:


#Perintah untuk melakukan koneksi Google colab dengan goole Drive sebagai penyimpanan

from google.colab import drive
drive.mount('/content/drive')


# In[24]:


#Pindah Path ke /content/drive/MyDrive/webmining/webmining

get_ipython().run_line_magic('cd', '/content/drive/MyDrive/webmining/tugas')


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

# In[25]:


get_ipython().system('git clone --depth=1 https://github.com/twintproject/twint.git')
get_ipython().run_line_magic('cd', 'twint')
get_ipython().system('pip3 install . -r requirements.txt')


# In[26]:


get_ipython().system('pip install twint')


# kalau pada saat import twint mengalami error mungkin bisa tambahkan code berikut ini, kemudian running lah.

# In[27]:


get_ipython().system('pip install aiohttp==3.7.0')


# Setelah menginstal twint tentunya belum bisa di jalankan karena kamu perlu menginportnya terlebih dahulu supaya twint bisa digunakan.
# 
# Caranya cukup mudah masukan saja kode di bawah ini.

# In[28]:


import twint


# In[29]:


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

# In[30]:


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

# In[31]:


import pandas as pd


# In[ ]:


data = pd.read_excel('dataGanjar.xlsx')
data


# In[ ]:




