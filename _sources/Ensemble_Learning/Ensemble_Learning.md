---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="nd3tnFBZ9JGI"}
# Ensemble Learning (Bagging, Stacking, dan Random Forest Classification) dan Grid Search

Ensemble Learning adalah algoritma dalam pembelajaran mesin (machine
learning) dimana algoritma ini sebagai pencarian solusi prediksi terbaik
dibandingkan dengan algoritma yang lain karena metode ensemble ini
menggunakan beberapa algoritma pembelajaran untuk pencapaian solusi
prediksi yang lebih baik daripada algoritma yang bisa diperoleh dari
salah satu pembelajaran algoritma kosituen saja. Tidak seperti ansamble
statistika didalam mekanika statistika biasanya selalu tak terbatas.
Ansemble Pembelajaran hanya terdiri dari seperangkat model alternatif
yang bersifat terbatas, namun biasanya memungkinkan untuk menjadi lebih
banyak lagi struktur fleksibel yang ada diantara alternatif model itu
sendiri.`<br>`{=html} Evaluasi prediksi dari ensemble biasanya
memerlukan banyak komputasi daripada evaluasi prediksi model tunggal
(single model), jadi ensemble ini memungkinkan untuk mengimbangi poor
learning algorithms oleh performasi lebih dari komputasi itu. Terdapat
beberapa metode ensemble learning yaitu bagging, stacking dan random
forest classification. Dan pada content kali ini akan membahas metode
metode tersebut dengan melakukan tahapan-tahapan berikut.
:::

::: {.cell .markdown id="GEOPGCZ5OowZ"}
## **Praprepocessing Text**

Proses ini merupakan proses awal sebelum melakukan proses prepocessing
text, yaitu proses untuk mendapatkan dataset yang akan digunakan untuk
proses prepocessing, yang mana dataset yang akan digunakan diambil dari
website dengan melakukan crawling pada website.
:::

::: {.cell .markdown id="aCxtNEL-33mc"}
### Crawling Tweeter

Crawling merupakan suatu proses pengambilan data dengan menggunakan
mesin yang dilakukan secara online. Proses ini dilakukan untuk mengimpor
data yang ditemukan kedalam file lokal komputer. Kemudian data yang
telah di impor tersebut akan dilakukan tahap prepocessing text. Pada
proses crawling kali ini dilakukan crawling data pada twitter dengan
menggunakan tools Twint.
:::

::: {.cell .markdown id="OujNXgS3334x"}
#### Installasi Twint

Twint merupakan sebuah tools yang digunakan untuk dapat melakukan
scraping data dari media sosial yaitu twitter dengan menggunakan bahasa
pemrograman python. Twint dapat dijalankan tanpa harus menggunakan API
twitter itu sendiri, namun kapasitas scrapingnya dibatasi sebanyak 3200
tweet.

Twint tidak hanya digunakan untuk mengambil data tweet, twint juga bisa
digunakan untuk mengambil data user, follower, retweet, dan sejenisnya.
Twint memanfaatkan operator pencarian twitter yang digunakan untuk
memilih dan memilah informasi yang sensitif, termasuk email dan nomor
telepon di dalamnya.

Proses installasi Twint dapat dilakukan dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="n3PpZoY34B4n" outputId="a1489103-1df0-4623-a958-641ea9cacd2c"}
``` {.python}
!git clone --depth=1 https://github.com/twintproject/twint.git
%cd twint
!pip3 install . -r requirements.txt
```

::: {.output .stream .stdout}
    Cloning into 'twint'...
    remote: Enumerating objects: 47, done.ote: Counting objects: 100% (47/47), done.ote: Compressing objects: 100% (44/44), done.ote: Total 47 (delta 3), reused 14 (delta 0), pack-reused 0ple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Processing /content/twint
      DEPRECATION: A future pip version will change local packages to be built in-place without first copying to a temporary directory. We recommend you use --use-feature=in-tree-build to test your packages with this new behavior before it becomes the default.
       pip 21.3 will remove support for this functionality. You can find discussion regarding this at https://github.com/pypa/pip/issues/7555.
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 1)) (3.8.3)
    Collecting aiodns
      Downloading aiodns-3.0.0-py3-none-any.whl (5.0 kB)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 3)) (4.6.3)
    Collecting cchardet
      Downloading cchardet-2.1.7-cp37-cp37m-manylinux2010_x86_64.whl (263 kB)
    ent already satisfied: pysocks in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 7)) (1.7.1)
    Requirement already satisfied: pandas>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 8)) (1.3.5)
    Collecting aiohttp_socks<=0.4.1
      Downloading aiohttp_socks-0.4.1-py3-none-any.whl (17 kB)
    Collecting schedule
      Downloading schedule-1.1.0-py2.py3-none-any.whl (10 kB)
    Requirement already satisfied: geopy in /usr/local/lib/python3.7/dist-packages (from -r requirements.txt (line 11)) (1.17.0)
    Collecting fake-useragent
      Downloading fake_useragent-0.1.14-py3-none-any.whl (13 kB)
    Collecting googletransx
      Downloading googletransx-2.4.2.tar.gz (13 kB)
    Requirement already satisfied: numpy>=1.17.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23.0->-r requirements.txt (line 8)) (1.21.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23.0->-r requirements.txt (line 8)) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23.0->-r requirements.txt (line 8)) (2022.6)
    Requirement already satisfied: attrs>=19.2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp_socks<=0.4.1->-r requirements.txt (line 9)) (22.1.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (6.0.2)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (2.1.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (1.3.1)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (4.0.2)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (1.8.1)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (4.1.1)
    Requirement already satisfied: asynctest==0.13.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (0.13.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from aiohttp->-r requirements.txt (line 1)) (1.3.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas>=0.23.0->-r requirements.txt (line 8)) (1.15.0)
    Requirement already satisfied: idna>=2.0 in /usr/local/lib/python3.7/dist-packages (from yarl<2.0,>=1.0->aiohttp->-r requirements.txt (line 1)) (2.10)
    Collecting pycares>=4.0.0
      Downloading pycares-4.2.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (288 kB)
    ent already satisfied: cffi>=1.5.0 in /usr/local/lib/python3.7/dist-packages (from pycares>=4.0.0->aiodns->-r requirements.txt (line 2)) (1.15.1)
    Requirement already satisfied: pycparser in /usr/local/lib/python3.7/dist-packages (from cffi>=1.5.0->pycares>=4.0.0->aiodns->-r requirements.txt (line 2)) (2.21)
    Collecting elastic-transport<9,>=8
      Downloading elastic_transport-8.4.0-py3-none-any.whl (59 kB)
    ent already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from elastic-transport<9,>=8->elasticsearch->-r requirements.txt (line 6)) (2022.9.24)
    Requirement already satisfied: geographiclib<2,>=1.49 in /usr/local/lib/python3.7/dist-packages (from geopy->-r requirements.txt (line 11)) (1.52)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from googletransx->-r requirements.txt (line 13)) (2.23.0)
    Collecting requests
      Downloading requests-2.28.1-py3-none-any.whl (62 kB)
    e=twint-2.1.21-py3-none-any.whl size=38871 sha256=7e92b75b75bda0d87bb045ea8e8c8475d1b649ecd34587562fee6730c8584fa3
      Stored in directory: /tmp/pip-ephem-wheel-cache-2kekm_44/wheels/f7/3e/11/2803f3c6890e87a9bec35bb8e37ef1ad0777a00f43e2441fb1
      Building wheel for googletransx (setup.py) ... e=googletransx-2.4.2-py3-none-any.whl size=15968 sha256=71cfbef1bf3218395f57f216de9a5aca4c28c22b4d107740e17b489a4d94252a
      Stored in directory: /root/.cache/pip/wheels/66/d5/b1/31104b338f7fd45aa8f7d22587765db06773b13df48a89735f
    Successfully built twint googletransx
    Installing collected packages: urllib3, requests, pycares, elastic-transport, schedule, googletransx, fake-useragent, elasticsearch, dataclasses, cchardet, aiohttp-socks, aiodns, twint
      Attempting uninstall: urllib3
        Found existing installation: urllib3 1.24.3
        Uninstalling urllib3-1.24.3:
          Successfully uninstalled urllib3-1.24.3
      Attempting uninstall: requests
        Found existing installation: requests 2.23.0
        Uninstalling requests-2.23.0:
          Successfully uninstalled requests-2.23.0
    Successfully installed aiodns-3.0.0 aiohttp-socks-0.4.1 cchardet-2.1.7 dataclasses-0.6 elastic-transport-8.4.0 elasticsearch-8.5.0 fake-useragent-0.1.14 googletransx-2.4.2 pycares-4.2.2 requests-2.28.1 schedule-1.1.0 twint-2.1.21 urllib3-1.26.12
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4xP28oqM4FXA" outputId="e1e9d186-0927-4569-8951-70d9f956384e"}
``` {.python}
!pip install nest-asyncio
```

::: {.output .stream .stdout}
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting nest-asyncio
      Downloading nest_asyncio-1.5.6-py3-none-any.whl (5.2 kB)
    Installing collected packages: nest-asyncio
    Successfully installed nest-asyncio-1.5.6
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8qPOG4QH4H8C" outputId="e0ae4e10-0125-40ba-defd-b919ebd34f5b"}
``` {.python}
!pip install aiohttp==3.7.0
```

::: {.output .stream .stdout}
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting aiohttp==3.7.0
      Downloading aiohttp-3.7.0-cp37-cp37m-manylinux2014_x86_64.whl (1.3 MB)
    ent already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp==3.7.0) (6.0.2)
    Collecting async-timeout<4.0,>=3.0
      Downloading async_timeout-3.0.1-py3-none-any.whl (8.2 kB)
    Requirement already satisfied: chardet<4.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp==3.7.0) (3.0.4)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp==3.7.0) (22.1.0)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp==3.7.0) (1.8.1)
    Requirement already satisfied: idna>=2.0 in /usr/local/lib/python3.7/dist-packages (from yarl<2.0,>=1.0->aiohttp==3.7.0) (2.10)
    Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from yarl<2.0,>=1.0->aiohttp==3.7.0) (4.1.1)
    Installing collected packages: async-timeout, aiohttp
      Attempting uninstall: async-timeout
        Found existing installation: async-timeout 4.0.2
        Uninstalling async-timeout-4.0.2:
          Successfully uninstalled async-timeout-4.0.2
      Attempting uninstall: aiohttp
        Found existing installation: aiohttp 3.8.3
        Uninstalling aiohttp-3.8.3:
          Successfully uninstalled aiohttp-3.8.3
    Successfully installed aiohttp-3.7.0 async-timeout-3.0.1
:::
:::

::: {.cell .markdown id="zkO1H3aj4NLW"}
#### Scraping Data Tweeter

Setelah proses installasi Twint berhasil selanjutnya lakukan scraping
data tweeter. Scraping sendiri merupakan proses pengambilan data dari
website. Untuk melakukan proses scraping data dari tweeter, tinggal
import twint untuk melakukan scraping data tweeter dengan tweet yang
mengandung kata \"\#rockygerung\" dengan limit 100 menggunakan source
code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iLQ_yMd_4QAW" outputId="3b11381b-9a1b-4b6c-8532-b7e2c0e78523"}
``` {.python}
import nest_asyncio
nest_asyncio.apply() #digunakan sekali untuk mengaktifkan tindakan serentak dalam notebook jupyter.
import twint #untuk import twint
c = twint.Config()
c.Search = '#rockygerung'
c.Lang = "in"
c.Pandas = True
c.Limit = 100
twint.run.Search(c)
```

::: {.output .stream .stdout}
    1590933496121688064 2022-11-11 05:04:38 +0000 <rockygerungcom> Quote:  Negeri yang memburuk karena dikelola manajemen benda mati: kerja, kerja, kerja.  | #negeri | #rockygerungcom | #rockygerung
    1590931427251539968 2022-11-11 04:56:24 +0000 <fajaronline> Ketum Projo Temui Prabowo Usai Jokowi Berikan Sinyal Dukungan, Rocky Gerung: Ngapain, Mestinya Bubar Aja  https://t.co/NUtPTVPwVJ #BudiArieSetiadi #PrabowoSubianto #Projo #RockyGerung
    1590872156975804416 2022-11-11 01:00:53 +0000 <fajaronline> Puan dan Mega ke Itaewon, Tragedi Kanjuruhan Tak Pernah Dikunjungi, Rocky Gerung: Ini Soal Standar Pemimpin  https://t.co/38Dhy63g42 #Itaewon #Kanjuruhan #PuanMaharani #RockyGerung
    1590652396832976897 2022-11-10 10:27:38 +0000 <Rgtvchannel_id> #RockyGerung #PoliticsAndBeyond #IndonesiaBerpikir #FasliJalal
    1590636648638750720 2022-11-10 09:25:04 +0000 <jpnncom> Rocky Gerung mengusulkan Anies Baswedan menjadikan Gibran sebagai cawapres. Begini respons Gibran, putra Presiden Jokowi, itu. #RockyGerung  https://t.co/X2CFC9264Q
    1590607905320935424 2022-11-10 07:30:51 +0000 <Rgtvchannel_id> 10 November. Ada yang harus dikenang, ada yang harus diingat.  Selamat Hari Pahlawan.  #RGTVChannelid #politicsandbeyond  #RockyGerung  https://t.co/PTAWnCqgiC
    1590529183578750976 2022-11-10 02:18:02 +0000 <terkinidotid> Rocky Gerung Usulkan Luhut Pandjaitan Jadi Cawapres Anies Baswedan: Pengamat politik Rocky Gerung mengusulkan agar Menko Marves Luhut Pandjaitan menjadi Calon Wakil Presiden (Cawapres) Anies Baswedan pada…  https://t.co/flwt1HHZHf #News #rockygerung #rockygerungluhutpandjaitan
    1590339773016997888 2022-11-09 13:45:23 +0000 <Korantalknews> Mantan Ajudan Mengubah Kesaksian di Sidang, Akui Takut Pada Sosok Ferdy Sambo   https://t.co/aQy01Ywtt9  #korantalk #korantalknew #ferdysambo #like #brigadirj #bandung #bali #jakarta #indonesia #jokowi #jokowidodo #rockygerung #bbm #bbmnaik #surabaya #cicaheumbandung #instagood  https://t.co/ZuYL8Ymfcc
    1590295887766777862 2022-11-09 10:51:00 +0000 <Rgtvchannel_id> #RockyGerung #PoliticsAndBeyond #IndonesiaBerpikir
    1590189864045379585 2022-11-09 03:49:42 +0000 <rockygerungcom> Quote:  Memaki boleh.  Tapi jangan cuma itu keahlianmu, Bong:)  | #memaki | #rockygerungcom | #rockygerung
    1590137806428393472 2022-11-09 00:22:51 +0000 <fajaronline> Rocky Gerung Puji Buzzer Demokrat Pintar-pintar, Panca: Beda Sama Buzzer Coro, Nggak Aneh Ade Armando Ditelanjangi  https://t.co/BRnNrbR82X #Buzzer #CiptaPancaLaksana #PartaiDemokrat #RockyGerung
    1589986615275188227 2022-11-08 14:22:04 +0000 <Rgtvchannel_id> Banyak kisah Rhoma Irama di masa lalu terbongkar ketika ngobrol seru bareng Rocky Gerung.  Saksikan Rabu besok (9/11/2022) hanya di Youtube RGTV Channel ID  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma  https://t.co/NpYjeq5De3
    1589972488498581505 2022-11-08 13:25:56 +0000 <Rgtvchannel_id> #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma
    1589612333428768768 2022-11-07 13:34:48 +0000 <Rgtvchannel_id> Rhoma : "Di dalam negara demokrasi itu harus ada oposisi. Kalau ada politisi yang mengatakan tak boleh ada oposisi, itu... "  Rocky: "Dungu!" Rhoma: "Iya... dungu."  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma  https://t.co/C5ypnzNfTC
    1589590967396749313 2022-11-07 12:09:54 +0000 <Rgtvchannel_id> Banyak jalan menuju Rhoma Bila engkau gagal di dalam satu cara Cari lagi cara lainnya  dari lirik lagu "Banyak Jalan Menuju Roma" cipt. Rhoma Irama  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma  https://t.co/cT05glNCvO
    1589526799637426178 2022-11-07 07:54:55 +0000 <AgusKwadrat> Politik Identitas adalah suatu gerakan politik yang dengan sengaja disebut sebagai pembodohan atau penipuan dan sering kali dihadapkan pada aksi massa dari suatu pihak yang memiliki kepentingan.  #AniesBaswedan  #SuryaPaloh  #Munarman  #NovelBamukmin  #RockyGerung  #NichoSilalahi
    1589525707813957632 2022-11-07 07:50:35 +0000 <AnakKedungBulus> Populisme Agama adalah suatu pendekatan politik yang dengan sengaja disebut sebagai kepentingan "umat" dan sering kali dihadapkan pada kepentingan suatu kelompok yang disebut "saudara seiman".  #AniesBaswedan  #SuryaPaloh  #Munarman  #NovelBamukmin  #RockyGerung  #NichoSilalahi
    1589485967949561856 2022-11-07 05:12:40 +0000 <Rgtvchannel_id> #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma
    1589414589674192896 2022-11-07 00:29:02 +0000 <terkinidotid> Rocky Gerung Sebut Ade Armando Memecah Belah Suara Ganjar: Pengamat politik Rocky Gerung memberikan pendapatnya terkait video Ade Armando yang membahas tentang Anies Baswedan dan penganut agama Kristen.  https://t.co/iPSg4WdRnV #News #rockygerung #rockygerungadearmando
    1589258072891219969 2022-11-06 14:07:06 +0000 <Rgtvchannel_id> Di episode #BisikanRhoma #50 yang sudah tayang di channel Rhoma Irama Official, sang Raja kasih tiga pertanyaan berat buat saya sambil berulang kali bilang, "I love you Rocky"  I love you too bang Haji!  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #RhomaIrama #Soneta  https://t.co/bw9qr6g5Ch
    1589250517846601729 2022-11-06 13:37:04 +0000 <terkinidotid> Rocky Gerung Sebut Jokowi Lebih Nyaman Bertemu Pendukung Dibanding Aksi Demonstrasi!: Belum lama ini, melalui sebuah unggahan dikanal youtobe, Rocky Gerung soroti Presiden Jokowi.  https://t.co/4YRQ9y3kwg #News #rockygerung #jokowi #presidenjokowi #jokowidodo
    1589248078254833664 2022-11-06 13:27:23 +0000 <Rgtvchannel_id> Rho-Ro Kolaborasi Dua Fenomena  Rocky Gerung bertemu Rhoma Irama di markas Soneta. Apa jadinya kalau dua sosok fenomenal berjumpa? No Rocky No Party!  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir #RhomaIrama #Soneta #BisikanRhoma  https://t.co/6c7NXIZt08
    1589198248153354242 2022-11-06 10:09:22 +0000 <Rgtvchannel_id> #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir
    1589098485361836032 2022-11-06 03:32:57 +0000 <rockygerungcom> Quote:  “Kami mulai kedunguan di sini. Kami akan tetap di sini”.  ~prasasti di sebuah kolam  | #kolam | #rockygerung | #rockygerungcom
    1588986143387062272 2022-11-05 20:06:33 +0000 <muzaqi_moh> @henrysubiakto @PartaiSocmed Guru besar tp otak kecil #RockyGerung
    1588835400050946051 2022-11-05 10:07:33 +0000 <Rgtvchannel_id> Cari keringat di kaki Gunung Fuji.  Di waktu senggang saat memenuhi undangan diskusi di Jepang beberapa waktu lalu, saya sempatkan berolahraga di hutan Pinus di sana.  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir  https://t.co/hPc21nXcnb
    1588811967674068993 2022-11-05 08:34:26 +0000 <Rgtvchannel_id> Cinta dan pengabdian adalah Edelweiss itu sendiri  #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir  https://t.co/lBS67SOfcz
    1588778651348533248 2022-11-05 06:22:03 +0000 <terkinidotid> Rocky Gerung Sindir Jokowi Kabur dari Istana Setiap Ada Demo: Sekali Lagi Kedunguan Namanya!: Rocky Gerung menyindir Presiden Joko Widodo (Jokowi) yang disebut kerap tak ada di Istana setiap kali…  https://t.co/7P45LatbUf #News #rockygerung #presidenjokowi #demojokowi #gnprdemo
    1588765232788566016 2022-11-05 05:28:43 +0000 <rockygerungcom> Quote:  Membaca itu pake otak. Supaya bila ngamuk, tak terlihat dungu.  | #otak | #rockygerungcom | #rockygerung
    1588729591786778624 2022-11-05 03:07:06 +0000 <AnakKedungBulus> Gus Staquf : "Bermain Identitas Agama Berarti Menggiring Perpecahan"   #AniesBawedan  #SlametMaarif #NovelBamukmin  #Munarman  #ReflyHarun  #RockyGerung
    1588728817543434241 2022-11-05 03:04:01 +0000 <DameRomadona> Rocky Gerung Punya Jagoan Pendamping Anies di Pilpres 2024, Bukan Aher Apa Lagi AHY  https://t.co/HUGqtxfDiN  #rockygerung #AniesPresiden2024  #plipres2024 #PemiluSerentak2024 #Pemilu2024 #ahy #news #new #NewsUpdates #berita
    1588509502206521344 2022-11-04 12:32:33 +0000 <partaigeloraid> Rocky Gerung: Kebijakan Second Home Visa Itu Konyol ↘️ Saksikan di #GeloraTV  https://t.co/CYDm6ExVfl  https://t.co/CYDm6ExVfl  https://t.co/CYDm6ExVfl #geloratalks #rockygerung #partaigelora #AyoMoveOn #arahbaruindonesia  https://t.co/wUp5l3M7mL
    1588494814437597186 2022-11-04 11:34:11 +0000 <Rgtvchannel_id> #RockyGerung #RGTVChannelID #PoliticsAndBeyond #IndonesiaBerpikir #JJRizal #LaksaCibinong
    1588448599075344386 2022-11-04 08:30:32 +0000 <ZFazaa86> Apa? Merendah untuk dipuji? Itu sama halnya dengan meninggi. #rockygerung
    1588447113629032448 2022-11-04 08:24:38 +0000 <Rgtvchannel_id> Selain mie ayam saya juga nikmati ketoprak di kaki gunung Galunggung.  Ketoprak ini makanan kesukaan Prabu Siliwangi. Dulu namanya ketopras.  #Rockymendation #RockyGerung #RockyGerungOfficial #PoliticsAndBeyond #IndonesiaBerpikir  https://t.co/0GzvS7Ab2f
    1588372858463391744 2022-11-04 03:29:34 +0000 <Rgtvchannel_id> #RockyGerung #RGTVChannelID #PoliticsAndBeyond #IndonesiaBerpikir #JJRizal #LaksaCibinong  https://t.co/DiRaQDfEeY
    1588360504556208128 2022-11-04 02:40:29 +0000 <fajaronline> Partai Demokrat Pastikan Mundur Jika Anies Pilih Luhut sebagai Cawapres  https://t.co/xWUMhGSEor #AniesBaswedan #RockyGerung #YanHarahap
    1588349415864950784 2022-11-04 01:56:25 +0000 <fajaronline> Rocky Gerung Sebut Luhut Pendamping Anies yang Memenuhi Kriteria, AHY dan Aher Kalah  https://t.co/XnzfBj3X5x #AhmadHeryawan #AHY #AniesBaswedan #RockyGerung
    1588338446191382528 2022-11-04 01:12:50 +0000 <fajaronline> Rocky Gerung Bilang Anies Butuh Pendamping yang Bisa Membangun Indonesia dengan Gaya Teknokrat  https://t.co/WRlA51Q9lD #AniesBaswedan #CawapresAnies #Pilpres2024 #RockyGerung
    1588110734784724993 2022-11-03 10:07:59 +0000 <Rgtvchannel_id> #RGTVCHANNELID mengajak #IndonesiaBerpikir  #RockyGerung #RGTVChannelID #PoliticsAndBeyond #IndonesiaBerpikir
    1588019170343792640 2022-11-03 04:04:08 +0000 <eramadanicom> Selengkapnya ditautan berikut :  https://t.co/8ezXkYu8On  #apindo #fahrihamzah #rockygerung #mancanegara #partai #partaigelora #visa #secondhome  https://t.co/l6PbPzknyX
    1587982514488242176 2022-11-03 01:38:29 +0000 <only4yo27523847> Bismillah   https://t.co/0R3ROGuv8b  #panggung #debat #roger roger roger #rockygerung #cacadberbangsa #muak muak muak...
    1587808712424497153 2022-11-02 14:07:51 +0000 <rockygerungcom> Quote:  Kosong ide, modal fanatik, tapi ngotot pujian. Ajaib :))  | #fanatik | #rockygerungcom | #rockygerung
    [!] No more data! Scraping will stop now.
    found 0 deleted tweets in this search.
:::
:::

::: {.cell .markdown id="8f3ni8V7Fmkd"}
#### Ambil Tweet

Setelah proses crawling didapatkan data tweeter diatas, pada data
tersebut terdapat data yang tidak diperlukan. Untuk melakukan
prepocessing hanya memerlukan data tweet dari user, maka dari itu buang
data yang tidak diperlukan dan ambil data tweet yang akan digunakan
dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="FZAniW9RrPZP" outputId="d48c51d2-e077-45e9-9868-dd6d81b7cd07"}
``` {.python}
Tweets_dfs = twint.storage.panda.Tweets_df
Tweets_dfs["tweet"]
```

::: {.output .execute_result execution_count="5"}
    0     Quote:  Negeri yang memburuk karena dikelola m...
    1     Ketum Projo Temui Prabowo Usai Jokowi Berikan ...
    2     Puan dan Mega ke Itaewon, Tragedi Kanjuruhan T...
    3     #RockyGerung #PoliticsAndBeyond #IndonesiaBerp...
    4     Rocky Gerung mengusulkan Anies Baswedan menjad...
    5     10 November. Ada yang harus dikenang, ada yang...
    6     Rocky Gerung Usulkan Luhut Pandjaitan Jadi Caw...
    7     Mantan Ajudan Mengubah Kesaksian di Sidang, Ak...
    8     #RockyGerung #PoliticsAndBeyond #IndonesiaBerp...
    9     Quote:  Memaki boleh.  Tapi jangan cuma itu ke...
    10    Rocky Gerung Puji Buzzer Demokrat Pintar-pinta...
    11    Banyak kisah Rhoma Irama di masa lalu terbongk...
    12    #RockyGerung #RockyGerungOfficial #PoliticsAnd...
    13    Rhoma : "Di dalam negara demokrasi itu harus a...
    14    Banyak jalan menuju Rhoma Bila engkau gagal di...
    15    Politik Identitas adalah suatu gerakan politik...
    16    Populisme Agama adalah suatu pendekatan politi...
    17    #RockyGerung #RockyGerungOfficial #PoliticsAnd...
    18    Rocky Gerung Sebut Ade Armando Memecah Belah S...
    19    Di episode #BisikanRhoma #50 yang sudah tayang...
    20    Rocky Gerung Sebut Jokowi Lebih Nyaman Bertemu...
    21    Rho-Ro Kolaborasi Dua Fenomena  Rocky Gerung b...
    22    #RockyGerung #RockyGerungOfficial #PoliticsAnd...
    23    Quote:  “Kami mulai kedunguan di sini. Kami ak...
    24    @henrysubiakto @PartaiSocmed Guru besar tp ota...
    25    Cari keringat di kaki Gunung Fuji.  Di waktu s...
    26    Cinta dan pengabdian adalah Edelweiss itu send...
    27    Rocky Gerung Sindir Jokowi Kabur dari Istana S...
    28    Quote:  Membaca itu pake otak. Supaya bila nga...
    29    Gus Staquf : "Bermain Identitas Agama Berarti ...
    30    Rocky Gerung Punya Jagoan Pendamping Anies di ...
    31    Rocky Gerung: Kebijakan Second Home Visa Itu K...
    32    #RockyGerung #RGTVChannelID #PoliticsAndBeyond...
    33    Apa? Merendah untuk dipuji? Itu sama halnya de...
    34    Selain mie ayam saya juga nikmati ketoprak di ...
    35    #RockyGerung #RGTVChannelID #PoliticsAndBeyond...
    36    Partai Demokrat Pastikan Mundur Jika Anies Pil...
    37    Rocky Gerung Sebut Luhut Pendamping Anies yang...
    38    Rocky Gerung Bilang Anies Butuh Pendamping yan...
    39    #RGTVCHANNELID mengajak #IndonesiaBerpikir  #R...
    40    Selengkapnya ditautan berikut :  https://t.co/...
    41    Bismillah   https://t.co/0R3ROGuv8b  #panggung...
    42    Quote:  Kosong ide, modal fanatik, tapi ngotot...
    Name: tweet, dtype: object
:::
:::

::: {.cell .markdown id="a-54hzhM4X5s"}
### Upload Data Tweet

Setelah data tweet di dapatkan, simpan data tweet tersebut dalam bentuk
csv, kemudian download dan upload ke github untuk nanti digunakan
sebagai dataset dari proses prepocessing text.
:::

::: {.cell .code id="DmQcczL74a62"}
``` {.python}
Tweets_dfs["tweet"].to_csv("RG.csv",index=False)
```
:::

::: {.cell .markdown id="joV0FyMs4eg8"}
## **Prepocessing Text**

Setelah proses crawling, selanjutnya dilakukan prepocessing text, yaitu
sebuah proses mesin yang digunakan untuk menyeleksi data teks agar lebih
terstruktur dengan melalui beberapa tahapan-tahapan yang meliputi
tahapan case folding, tokenizing, filtering dan stemming. Sebelum
melakukan tahapan-tahapan tersebut, terlebih dahulu kita import data
crawling yang diupload ke github tadi dengan menggunakan library pandas
pada source code berikut.
:::

::: {.cell .code colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="Xac5Jy4J5tg8" outputId="0ef48a9f-beac-47f8-aaaa-3d143555c39f"}
``` {.python}
import pandas as pd 

tweets = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/RG.csv",index_col=False)
tweets
```

::: {.output .execute_result execution_count="7"}
```{=html}
  <div id="df-f251aad2-e06e-4b89-8bf4-dffe79027949">
    <div class="colab-df-container">
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
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anies itu Penantang 👈👉 said #RockyGerung    ht...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#RockyGerung #RGTVChannelid #PolitcsAndBeyond ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KODE DARI OPUNG LUHUT!!  Tidak mau jadi calon ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rocky Gerung Bicara Soal Keberlangsungan Pemer...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#RockyGerung #RGTVChannelid #PolitcsAndBeyond ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Jaman REZIM SKRG kaya bapa tiri di sinetron......</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Sobat akal sehat nantikan video part 2 yang ak...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>Menteri Koordinator bidang Kemaritiman dan Inv...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>PEMBODOHAN YANG BERLANGSUNG DALAM KURIKULUM ET...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Setuju dengan Bung Rocky?!  Saksikan selengkap...</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f251aad2-e06e-4b89-8bf4-dffe79027949')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f251aad2-e06e-4b89-8bf4-dffe79027949 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f251aad2-e06e-4b89-8bf4-dffe79027949');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="_FQhW9mx_h2z"}
Setelah data crawling berhasil di import, selanjutnya lakukan
tahapan-tahapan prepocessing seperti berikut.
:::

::: {.cell .markdown id="-N2T3ds14s6N"}
### Case Folding

Setelah berhassil mengambil dataset, selanjutnya ke proses prepocessing
ke tahapan case folding yaitu tahapan pertama untuk melakukan
prepocessing text dengan mengubah text menjadi huruf kecil semua dengan
menghilangkan juga karakter spesial, angka, tanda baca, spasi serta
huruf yang tidak penting.
:::

::: {.cell .markdown id="Q7_i2mK14vo_"}
#### Merubah Huruf Kecil Semua

Tahapan case folding yang pertama yaitu merubah semua huruf menjadi
huruf kecil semua menggunakan fungsi lower() dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="JhDqytF84y3x" outputId="319e2b30-90b7-4d0a-e9b4-e34c28c60a8c"}
``` {.python}
tweets['tweet'] = tweets['tweet'].str.lower()


tweets['tweet']
```

::: {.output .execute_result execution_count="8"}
    0     anies itu penantang 👈👉 said #rockygerung    ht...
    1     #rockygerung #rgtvchannelid #politcsandbeyond ...
    2     kode dari opung luhut!!  tidak mau jadi calon ...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4     #rockygerung #rgtvchannelid #politcsandbeyond ...
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron......
    96    sobat akal sehat nantikan video part 2 yang ak...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky?!  saksikan selengkap...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="Fdq5WkS942J7"}
#### Menghapus Karakter Spesial

Tahapan case folding selanjutnya ialah menghapus karakter spesial dengan
menggunakan library nltk, untuk menggunakan librarynya terlebih dahulu
install dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="pCaPbvxW44r9" outputId="8ef68aea-c19a-4691-e0aa-9397adfd8d17"}
``` {.python}
#install library nltk
!pip install nltk
```

::: {.output .stream .stdout}
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (3.7)
    Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.7/dist-packages (from nltk) (2022.6.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from nltk) (4.64.1)
    Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from nltk) (1.2.0)
    Requirement already satisfied: click in /usr/local/lib/python3.7/dist-packages (from nltk) (7.1.2)
:::
:::

::: {.cell .markdown id="ahtwYhOQ4-iz"}
Setelah library nltk terinstall kita import librarynya dan buat sebuah
function untuk menghapus karakter spesial tersebut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="uvSaD4zZ47lK" outputId="f2534352-a853-4b22-8265-c499e7128092"}
``` {.python}
import string 
import re #regex library
# import word_tokenize & FreqDist from NLTK

from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist


def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
tweets['tweet'] = tweets['tweet'].apply(remove_special)
tweets['tweet']
```

::: {.output .execute_result execution_count="10"}
    0                           anies itu penantang ?? said
    1                                                      
    2     kode dari opung luhut!! tidak mau jadi calon p...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4                                                      
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron......
    96    sobat akal sehat nantikan video part 2 yang ak...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky?! saksikan selengkapn...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="aLm_vpeH5DSV"}
#### Menghapus Angka

Selanjutnya melakukan penghapusan angka, penghapusan angka disini
fleksibel, jika angka ingin dijadikan fitur maka penghapusan angka tidak
perlu dilakukan. Untuk data tweet ini saya tidak ingin menjadikan angka
sebagai fitur, untuk itu dilakukan penghapusan angka dengan function
berikut
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="0ADN8WB0Yli0" outputId="649c6236-c655-4731-c1e8-7e29c5bb6d21"}
``` {.python}
#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

tweets['tweet'] = tweets['tweet'].apply(remove_number)
tweets['tweet']
```

::: {.output .execute_result execution_count="11"}
    0                           anies itu penantang ?? said
    1                                                      
    2     kode dari opung luhut!! tidak mau jadi calon p...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4                                                      
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron......
    96    sobat akal sehat nantikan video part  yang aka...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky?! saksikan selengkapn...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="jl2FQ1su5M5r"}
#### Menghapus Tanda Baca

Selanjutnya penghapusan tanda baca yang tidak perlu yang dilakukan
dengan function punctuation berikut
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="OmiSTkbP5NnI" outputId="3b8ec72b-21f0-49c4-d5b1-3f6445fdc3b8"}
``` {.python}
#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

tweets['tweet'] = tweets['tweet'].apply(remove_punctuation)
tweets['tweet']
```

::: {.output .execute_result execution_count="12"}
    0                             anies itu penantang  said
    1                                                      
    2     kode dari opung luhut tidak mau jadi calon pre...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4                                                      
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron di...
    96    sobat akal sehat nantikan video part  yang aka...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky saksikan selengkapnya...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="d0YF7OFT5Qzl"}
#### Menghapus Spasi

Selanjutnya melakukan penghapusan spasi dengab menggunakan function
berikut
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="veUMlXci5TSL" outputId="0dfeaef5-da14-47c6-e21b-0daf446cb202"}
``` {.python}
#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

tweets['tweet'] = tweets['tweet'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

tweets['tweet'] = tweets['tweet'].apply(remove_whitespace_multiple)
tweets['tweet']
```

::: {.output .execute_result execution_count="13"}
    0                              anies itu penantang said
    1                                                      
    2     kode dari opung luhut tidak mau jadi calon pre...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4                                                      
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron di...
    96    sobat akal sehat nantikan video part yang akan...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky saksikan selengkapnya...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="cA3VGUt15WqR"}
#### Menghapus Huruf

Selanjutnya melakukan penghapusan huruf yang tidak bermakna dengan
function berikut
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="oCp446D45ZUP" outputId="6d8a9657-a94e-4930-aa15-4029aaed2a43"}
``` {.python}
# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

tweets['tweet'] = tweets['tweet'].apply(remove_singl_char)
tweets['tweet']
```

::: {.output .execute_result execution_count="14"}
    0                              anies itu penantang said
    1                                                      
    2     kode dari opung luhut tidak mau jadi calon pre...
    3     rocky gerung bicara soal keberlangsungan pemer...
    4                                                      
                                ...                        
    95    jaman rezim skrg kaya bapa tiri di sinetron di...
    96    sobat akal sehat nantikan video part yang akan...
    97    menteri koordinator bidang kemaritiman dan inv...
    98    pembodohan yang berlangsung dalam kurikulum et...
    99    setuju dengan bung rocky saksikan selengkapnya...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="PbLjEozU5cJE"}
### Tokenizing

Setelah tahapan case folding selesai, selanjutnya masuk ke tahapan
tokenizing yang merupakan tahapan prepocessing yang memecah kalimat dari
text menjadi kata agar membedakan antara kata pemisah atau bukan. Untuk
melakukan tokenizing dapat menggunakan dengan library nltk dan function
berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="cIjz2Jkf5eqF" outputId="1deabbcf-f613-4f10-9fbc-b48fb602d151"}
``` {.python}
import nltk
nltk.download('punkt')
# NLTK word Tokenize 
```

::: {.output .stream .stderr}
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
:::

::: {.output .execute_result execution_count="15"}
    True
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="LTXPY5cK5hZ-" outputId="5e8cfeac-a276-4974-90be-c9c6a559b4d0"}
``` {.python}
# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

tweets['tweet'] = tweets['tweet'].apply(word_tokenize_wrapper)
tweets['tweet']
```

::: {.output .execute_result execution_count="16"}
    0                         [anies, itu, penantang, said]
    1                                                    []
    2     [kode, dari, opung, luhut, tidak, mau, jadi, c...
    3     [rocky, gerung, bicara, soal, keberlangsungan,...
    4                                                    []
                                ...                        
    95    [jaman, rezim, skrg, kaya, bapa, tiri, di, sin...
    96    [sobat, akal, sehat, nantikan, video, part, ya...
    97    [menteri, koordinator, bidang, kemaritiman, da...
    98    [pembodohan, yang, berlangsung, dalam, kurikul...
    99    [setuju, dengan, bung, rocky, saksikan, seleng...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="O9FCmHZ55jvw"}
### Filtering(Stopword)

Tahapan prepocessing selanjutnya ialah filtering atau disebut juga
stopword yang merupakan lanjutan dari tahapan tokenizing yang digunakan
untuk mengambil kata-kata penting dari hasil tokenizing tersebut dengan
menghapus kata hubung yang tidak memiliki makna.

Proses stopword dapat dilakukan dengan mengimport library stopword dan
function berikut untuk melakukan stopword.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="l8_flTCTaSpW" outputId="e7b50b97-b4a1-4c5b-b03b-ad290c2debd9"}
``` {.python}
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

::: {.output .stream .stderr}
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
:::

::: {.output .execute_result execution_count="17"}
    True
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="asi-yUCR5yht" outputId="6e660acc-d81c-4ec0-ff34-97b46106cfb0"}
``` {.python}
list_stopwords = stopwords.words('indonesian')

# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

tweets['tweet'] = tweets['tweet'].apply(stopwords_removal)

tweets['tweet']
```

::: {.output .execute_result execution_count="18"}
    0                              [anies, penantang, said]
    1                                                    []
    2     [kode, opung, luhut, calon, presiden, acara, k...
    3     [rocky, gerung, bicara, keberlangsungan, pemer...
    4                                                    []
                                ...                        
    95    [jaman, rezim, skrg, kaya, bapa, tiri, sinetro...
    96    [sobat, akal, sehat, nantikan, video, part, ta...
    97    [menteri, koordinator, bidang, kemaritiman, in...
    98        [pembodohan, kurikulum, etika, rocky, gerung]
    99    [setuju, rocky, saksikan, selengkapnya, youtub...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="ei8PlRNj50Yi"}
### Stemming

Tahapan terakhir dari proses prepocessing ialah stemming yang merupakan
penghapusan suffix maupun prefix pada text sehingga menjadi kata dasar.
Proses ini dapat dilakukan dengan menggunakan library sastrawi dan
swifter.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Hae1yyhI52-2" outputId="a30cbb0d-4e92-477e-97a2-56dac1ba5687"}
``` {.python}
!pip install Sastrawi
```

::: {.output .stream .stdout}
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting Sastrawi
      Downloading Sastrawi-1.0.1-py2.py3-none-any.whl (209 kB)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="_RhJYIWh57my" outputId="d58c5ae0-9c8d-4988-9bdd-844a5e885970"}
``` {.python}
!pip install swifter
```

::: {.output .stream .stdout}
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting swifter
      Downloading swifter-1.3.4.tar.gz (830 kB)
    ent already satisfied: pandas>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from swifter) (1.3.5)
    Collecting psutil>=5.6.6
      Downloading psutil-5.9.4-cp36-abi3-manylinux_2_12_x86_64.manylinux2010_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (280 kB)
    ent already satisfied: dask[dataframe]>=2.10.0 in /usr/local/lib/python3.7/dist-packages (from swifter) (2022.2.0)
    Requirement already satisfied: tqdm>=4.33.0 in /usr/local/lib/python3.7/dist-packages (from swifter) (4.64.1)
    Requirement already satisfied: ipywidgets>=7.0.0 in /usr/local/lib/python3.7/dist-packages (from swifter) (7.7.1)
    Requirement already satisfied: cloudpickle>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from swifter) (1.5.0)
    Requirement already satisfied: parso>0.4.0 in /usr/local/lib/python3.7/dist-packages (from swifter) (0.8.3)
    Requirement already satisfied: bleach>=3.1.1 in /usr/local/lib/python3.7/dist-packages (from swifter) (5.0.1)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.7/dist-packages (from bleach>=3.1.1->swifter) (1.15.0)
    Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach>=3.1.1->swifter) (0.5.1)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (21.3)
    Requirement already satisfied: fsspec>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (2022.10.0)
    Requirement already satisfied: toolz>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (0.12.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (6.0)
    Requirement already satisfied: partd>=0.3.10 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (1.3.0)
    Requirement already satisfied: numpy>=1.18 in /usr/local/lib/python3.7/dist-packages (from dask[dataframe]>=2.10.0->swifter) (1.21.6)
    Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (5.1.1)
    Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (7.9.0)
    Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (3.6.1)
    Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (5.3.4)
    Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (0.2.0)
    Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets>=7.0.0->swifter) (3.0.3)
    Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->swifter) (6.0.4)
    Requirement already satisfied: jupyter-client in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets>=7.0.0->swifter) (6.1.12)
    Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (57.4.0)
    Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (4.8.0)
    Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (2.0.10)
    Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (2.6.1)
    Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (4.4.2)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (0.7.5)
    Collecting jedi>=0.10
      Downloading jedi-0.18.1-py2.py3-none-any.whl (1.6 MB)
    ent already satisfied: backcall in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (0.2.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->dask[dataframe]>=2.10.0->swifter) (3.0.9)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=1.0.0->swifter) (2022.6)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=1.0.0->swifter) (2.8.2)
    Requirement already satisfied: locket in /usr/local/lib/python3.7/dist-packages (from partd>=0.3.10->dask[dataframe]>=2.10.0->swifter) (1.0.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython>=4.0.0->ipywidgets>=7.0.0->swifter) (0.2.5)
    Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (5.7.16)
    Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.13.3)
    Requirement already satisfied: jupyter-core>=4.4.0 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (4.11.2)
    Requirement already satisfied: nbconvert<6.0 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (5.6.1)
    Requirement already satisfied: prometheus-client in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.15.0)
    Requirement already satisfied: Send2Trash in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (1.8.0)
    Requirement already satisfied: jinja2<=3.0.0 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (2.11.3)
    Requirement already satisfied: nbformat in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (5.7.0)
    Requirement already satisfied: pyzmq>=17 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (23.2.1)
    Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2<=3.0.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (2.0.1)
    Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.4)
    Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.8.4)
    Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (1.5.0)
    Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.7.1)
    Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from nbconvert<6.0->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.6.0)
    Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.7/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (4.3.3)
    Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.7/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (2.16.2)
    Requirement already satisfied: importlib-metadata>=3.6 in /usr/local/lib/python3.7/dist-packages (from nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (4.13.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=3.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (4.1.1)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=3.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (3.10.0)
    Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.19.2)
    Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (22.1.0)
    Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from jsonschema>=2.6->nbformat->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (5.10.0)
    Requirement already satisfied: ptyprocess in /usr/local/lib/python3.7/dist-packages (from terminado>=0.8.1->notebook>=4.4.1->widgetsnbextension~=3.6.0->ipywidgets>=7.0.0->swifter) (0.7.0)
    Building wheels for collected packages: swifter
      Building wheel for swifter (setup.py) ... e=swifter-1.3.4-py3-none-any.whl size=16323 sha256=794e1c7f4d1b3691e5bb6967ef3705a7c909d2f135adeb78910fa817f6c4e083
      Stored in directory: /root/.cache/pip/wheels/29/a7/0e/3a8f17ac69d759e1e93647114bc9bdc95957e5b0cbfd405205
    Successfully built swifter
    Installing collected packages: jedi, psutil, swifter
      Attempting uninstall: psutil
        Found existing installation: psutil 5.4.8
        Uninstalling psutil-5.4.8:
          Successfully uninstalled psutil-5.4.8
    Successfully installed jedi-0.18.1 psutil-5.9.4 swifter-1.3.4
:::

::: {.output .display_data}
``` {.json}
{"pip_warning":{"packages":["psutil"]}}
```
:::
:::

::: {.cell .code colab="{\"referenced_widgets\":[\"0c9940a7760b48cbb3055943574ec1fe\",\"5128098a6fbc4591a8e84c2381b56772\",\"0bbebdac8e934a66b60a7deec723b78e\",\"6b0ac516c91742babc5184bb32b1f2d3\",\"151d695813e84efc851805a356b5ce3a\",\"25adf6ce3cae4302b4b1c0cf4c7092ae\",\"fde44c08211348b1ac9142fdb03f9ed4\",\"22b4ab598c4d46cfbb9a36e99309a1bf\",\"254c280427c84e97be0c9444385f4308\",\"3f7ff5df96394f3aa6eb843056f91440\",\"d9e4db896f9d4bca8a05f59b333ff51d\"],\"base_uri\":\"https://localhost:8080/\"}" id="fSBDA7B15-lG" outputId="c612d50b-e60e-48fc-f6e9-bce5a0cd0cfa"}
``` {.python}
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in tweets['tweet']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

tweets['tweet'] = tweets['tweet'].swifter.apply(get_stemmed_term)
tweets['tweet']
```

::: {.output .stream .stdout}
    424
    ------------------------
    anies : anies
    penantang : tantang
    said : said
    kode : kode
    opung : opung
    luhut : luhut
    calon : calon
    presiden : presiden
    acara : acara
    kib : kib
    pimpinan : pimpin
    airlangga : airlangga
    hartarto : hartarto
    gimana : gimana
    arti : arti
    rocky : rocky
    gerung : gerung
    bicara : bicara
    keberlangsungan : langsung
    pemerintahan : perintah
    jokowi : jokowi
    pengamat : amat
    politik : politik
    pendapatnya : dapat
    era : era
    menurutnya : turut
    rgtv : rgtv
    channel : channel
    id : id
    mengajak : ajak
    wali : wali
    kota : kota
    solo : solo
    gibran : gibran
    rakabuming : rakabuming
    raka : raka
    mengakui : aku
    menerima : terima
    kritikan : kritik
    bertemu : temu
    tweet : tweet
    sekolah : sekolah
    menandakan : tanda
    pelajar : ajar
    belom : bom
    pemikir : pikir
    quote : quote
    kalangan : kalang
    istana : istana
    menelepon : telepon
    pengen : ken
    berkunjung : kunjung
    silakan : sila
    ucapkan : ucap
    publik : publik
    sisi : sisi
    ekonomi : ekonomi
    dikritik : kritik
    habis : habis
    sadar : sadar
    belajar : ajar
    otak : otak
    kosong : kosong
    dungu : dungu
    mengaku : aku
    bertandang : tandang
    kediaman : diam
    orang : orang
    jenius : jenius
    masukan : masuk
    baca : baca
    selengkapnya : lengkap
    bahas : bahas
    dungudunguan : dungudunguan
    saksikan : saksi
    youtube : youtube
    temukan : temu
    link : link
    akui : aku
    disemprot : semprot
    dibahas : bahas
    xi : xi
    jinping : jinping
    dikudeta : kudeta
    dampaknya : dampak
    indonesia : indonesia
    tantang : tantang
    juara : juara
    kelas : kelas
    terbang : terbang
    one : one
    pride : pride
    mma : mma
    dukung : dukung
    bertarung : tarung
    ring : ring
    duel : duel
    sengit : sengit
    penyandang : sandang
    menang : menang
    awali : awal
    harimu : hari
    semangat : semangat
    selamat : selamat
    senin : senin
    sobat : sobat
    akal : akal
    sehat : sehat
    salam : salam
    berguru : guru
    reaksi : reaksi
    disinggung : singgung
    oligarki : oligarki
    menceritakan : cerita
    momen : momen
    pertemuan : temu
    putra : putra
    aliansi : aliansi
    serikat : serikat
    buruh : buruh
    bersatu : satu
    berama : ama
    quen : quen
    of : of
    longmarct : longmarct
    bergerak : gerak
    puncak : puncak
    negara : negara
    jakarta : jakarta
    sep : sep
    nggak : nggak
    urusan : urus
    unggahan : unggah
    foto : foto
    media : media
    sosial : sosial
    menuai : tuai
    sorotan : sorot
    lantaran : lantar
    penjajagan : penjajagan
    pasangan : pasang
    mas : mas
    bang : bang
    manyambut : manyambut
    pertarungan : tarung
    tamu : tamu
    tebak : tebak
    clue : clue
    pria : pria
    kelahiran : lahir
    magetan : magetan
    jawa : jawa
    timur : timur
    hobi : hobi
    muay : muay
    thai : thai
    berjualan : jual
    bakso : bakso
    heran : heran
    disambangi : sambang
    dibenci : benci
    nicho : nicho
    silalahi : silalahi
    soroti : sorot
    maksa : maksa
    statemen : statemen
    berbahaya : bahaya
    pegiat : giat
    aktivis : aktivis
    menyoroti : sorot
    pernyataan : nyata
    pndah : pndah
    dr : dr
    jkt : jkt
    beliau : beliau
    cm : cm
    pindah : pindah
    merdeka : merdeka
    selatan : selatan
    utara : utara
    bukabukaan : bukabukaan
    obrolan : obrol
    jam : jam
    bareng : bareng
    buzzerp : buzzerp
    visible : visible
    confusion : confusion
    baswedan : baswedan
    legitimasi : legitimasi
    menilai : nilai
    gubernur : gubernur
    dki : dki
    memiliki : milik
    republik : republik
    heboh : heboh
    rumor : rumor
    militer : militer
    china : china
    ditahan : tahan
    rumah : rumah
    tahanan : tahan
    persaingan : saing
    jenderal : jenderal
    laporan : lapor
    tentara : tentara
    akrab : akrab
    kekuasaan : kuasa
    mendekat : dekat
    rg : rg
    didatangi : datang
    diyakini : yakin
    berkantor : kantor
    medan : medan
    kunjungi : kunjung
    pastikan : pasti
    cebongkampret : cebongkampret
    pan : pan
    berminat : minat
    mengasuh : asuh
    mending : mending
    diasuh : asuh
    partai : partai
    amanat : amanat
    nasional : nasional
    peluang : peluang
    selesai : selesai
    jarangjarang : jarangjarang
    lord : lord
    menatap : tatap
    pasca : pasca
    via : via
    part : part
    tonton : tonton
    ditakedown : ditakedown
    klik : klik
    kebebasan : bebas
    freedom : freedom
    natural : natural
    right : right
    konstitusi : konstitusi
    tugas : tugas
    pemerintah : perintah
    melindungi : lindung
    sambangi : sambang
    pembahasannya : bahas
    pesan : pesan
    ramai : ramai
    netizen : netizen
    bersuara : suara
    pertalite : pertalite
    boros : boros
    cepat : cepat
    harganya : harga
    coba : coba
    simak : simak
    videonya : video
    terbagi : bagi
    demo : demo
    golput : golput
    tempo : tempo
    kritik : kritik
    idolakan : idola
    cuitan : cuit
    viral : viral
    menyebut : sebut
    cebong : cebong
    kampret : kampret
    joko : joko
    widodo : widodo
    istilah : istilah
    mengunjungi : unjung
    masuk : masuk
    angin : angin
    temui : temu
    om : om
    pangeran : pangeran
    ngobrol : ngobrol
    tolong : tolong
    cariin : cariin
    cewe : cewe
    doi : doi
    grogi : grogi
    ama : ama
    cowo : cowo
    kekekekkek : kekekekkek
    ketemu : ketemu
    lemes : lemes
    saudara : saudara
    rumahnya : rumah
    kirakira : kirakira
    prediksi : prediksi
    gangguan : ganggu
    terciptanya : cipta
    hoax : hoax
    kebodohan : bodoh
    tim : tim
    bocorkan : bocor
    pembahasan : bahas
    disangka : sangka
    akun : akun
    twitter : twitter
    pribadinya : pribadi
    menanggapi : tanggap
    mementionnya : mementionnya
    dll : dll
    cemas : cemas
    kadrun : kadrun
    cemburu : cemburu
    siapkan : siap
    hadiah : hadiah
    sepatu : sepatu
    nasib : nasib
    pensiun : pensiun
    jabatannya : jabat
    sinyal : sinyal
    diganggu : ganggu
    penguasa : kuasa
    sprindik : sprindik
    sowan : sowan
    tegaskan : tegas
    bapaknya : bapak
    berkawan : kawan
    botol : botol
    minuman : minum
    gagal : gagal
    fokus : fokus
    kabar : kabar
    gembira : gembira
    sayembara : sayembara
    komentar : komentar
    terlucu : lucu
    hadiahnya : hadiah
    tercengang : cengang
    anak : anak
    walikota : walikota
    mengunggah : unggah
    fotonya : foto
    lei : lei
    host : host
    desain : desain
    jelang : jelang
    pilpres : pilpres
    duga : duga
    dipanggil : panggil
    kpk : kpk
    salah : salah
    idola : idola
    janjikan : janji
    nawarin : nawarin
    operasi : operasi
    katarak : katarak
    yaa : yaa
    bhuehuehue : bhuehuehue
    angkat : angkat
    mengomentari : komentar
    insiden : insiden
    penembakan : tembak
    brigadir : brigadir
    dikepalai : palai
    ferdy : ferdy
    sambo : sambo
    berita : berita
    periode : periode
    menko : menko
    singgung : singgung
    big : big
    data : data
    binsar : binsar
    pandjaitan : pandjaitan
    berbincang : bincang
    contohi : contoh
    amerika : amerika
    usul : usul
    sistem : sistem
    pilkada : pilkada
    dievalusi : dievalusi
    tamparan : tampar
    rakyat : rakyat
    parpol : parpol
    serang : serang
    berebut : rebut
    elektabilitas : elektabilitas
    enak : enak
    nontonnya : nontonnya
    marves : marves
    mustahil : mustahil
    kalean : kalean
    mesti : mesti
    nonton : nonton
    menteri : menteri
    koordinator : koordinator
    bidang : bidang
    kemaritiman : maritim
    investasi : investasi
    usil : usil
    bintang : bintang
    podcast : podcast
    jaman : jaman
    rezim : rezim
    skrg : skrg
    kaya : kaya
    bapa : bapa
    tiri : tiri
    sinetron : sinetron
    kasih : kasih
    makan : makan
    bsu : bsu
    blt : blt
    anakrakyat : anakrakyat
    udh : udh
    kekenyangan : kenyang
    trs : trs
    pecutin : pecutin
    nantikan : nanti
    video : video
    tayang : tayang
    rabu : rabu
    wib : wib
    isu : isu
    diperbincangkan : bincang
    pembodohan : bodoh
    kurikulum : kurikulum
    etika : etika
    setuju : tuju
    {'anies': 'anies', 'penantang': 'tantang', 'said': 'said', 'kode': 'kode', 'opung': 'opung', 'luhut': 'luhut', 'calon': 'calon', 'presiden': 'presiden', 'acara': 'acara', 'kib': 'kib', 'pimpinan': 'pimpin', 'airlangga': 'airlangga', 'hartarto': 'hartarto', 'gimana': 'gimana', 'arti': 'arti', 'rocky': 'rocky', 'gerung': 'gerung', 'bicara': 'bicara', 'keberlangsungan': 'langsung', 'pemerintahan': 'perintah', 'jokowi': 'jokowi', 'pengamat': 'amat', 'politik': 'politik', 'pendapatnya': 'dapat', 'era': 'era', 'menurutnya': 'turut', 'rgtv': 'rgtv', 'channel': 'channel', 'id': 'id', 'mengajak': 'ajak', 'wali': 'wali', 'kota': 'kota', 'solo': 'solo', 'gibran': 'gibran', 'rakabuming': 'rakabuming', 'raka': 'raka', 'mengakui': 'aku', 'menerima': 'terima', 'kritikan': 'kritik', 'bertemu': 'temu', 'tweet': 'tweet', 'sekolah': 'sekolah', 'menandakan': 'tanda', 'pelajar': 'ajar', 'belom': 'bom', 'pemikir': 'pikir', 'quote': 'quote', 'kalangan': 'kalang', 'istana': 'istana', 'menelepon': 'telepon', 'pengen': 'ken', 'berkunjung': 'kunjung', 'silakan': 'sila', 'ucapkan': 'ucap', 'publik': 'publik', 'sisi': 'sisi', 'ekonomi': 'ekonomi', 'dikritik': 'kritik', 'habis': 'habis', 'sadar': 'sadar', 'belajar': 'ajar', 'otak': 'otak', 'kosong': 'kosong', 'dungu': 'dungu', 'mengaku': 'aku', 'bertandang': 'tandang', 'kediaman': 'diam', 'orang': 'orang', 'jenius': 'jenius', 'masukan': 'masuk', 'baca': 'baca', 'selengkapnya': 'lengkap', 'bahas': 'bahas', 'dungudunguan': 'dungudunguan', 'saksikan': 'saksi', 'youtube': 'youtube', 'temukan': 'temu', 'link': 'link', 'akui': 'aku', 'disemprot': 'semprot', 'dibahas': 'bahas', 'xi': 'xi', 'jinping': 'jinping', 'dikudeta': 'kudeta', 'dampaknya': 'dampak', 'indonesia': 'indonesia', 'tantang': 'tantang', 'juara': 'juara', 'kelas': 'kelas', 'terbang': 'terbang', 'one': 'one', 'pride': 'pride', 'mma': 'mma', 'dukung': 'dukung', 'bertarung': 'tarung', 'ring': 'ring', 'duel': 'duel', 'sengit': 'sengit', 'penyandang': 'sandang', 'menang': 'menang', 'awali': 'awal', 'harimu': 'hari', 'semangat': 'semangat', 'selamat': 'selamat', 'senin': 'senin', 'sobat': 'sobat', 'akal': 'akal', 'sehat': 'sehat', 'salam': 'salam', 'berguru': 'guru', 'reaksi': 'reaksi', 'disinggung': 'singgung', 'oligarki': 'oligarki', 'menceritakan': 'cerita', 'momen': 'momen', 'pertemuan': 'temu', 'putra': 'putra', 'aliansi': 'aliansi', 'serikat': 'serikat', 'buruh': 'buruh', 'bersatu': 'satu', 'berama': 'ama', 'quen': 'quen', 'of': 'of', 'longmarct': 'longmarct', 'bergerak': 'gerak', 'puncak': 'puncak', 'negara': 'negara', 'jakarta': 'jakarta', 'sep': 'sep', 'nggak': 'nggak', 'urusan': 'urus', 'unggahan': 'unggah', 'foto': 'foto', 'media': 'media', 'sosial': 'sosial', 'menuai': 'tuai', 'sorotan': 'sorot', 'lantaran': 'lantar', 'penjajagan': 'penjajagan', 'pasangan': 'pasang', 'mas': 'mas', 'bang': 'bang', 'manyambut': 'manyambut', 'pertarungan': 'tarung', 'tamu': 'tamu', 'tebak': 'tebak', 'clue': 'clue', 'pria': 'pria', 'kelahiran': 'lahir', 'magetan': 'magetan', 'jawa': 'jawa', 'timur': 'timur', 'hobi': 'hobi', 'muay': 'muay', 'thai': 'thai', 'berjualan': 'jual', 'bakso': 'bakso', 'heran': 'heran', 'disambangi': 'sambang', 'dibenci': 'benci', 'nicho': 'nicho', 'silalahi': 'silalahi', 'soroti': 'sorot', 'maksa': 'maksa', 'statemen': 'statemen', 'berbahaya': 'bahaya', 'pegiat': 'giat', 'aktivis': 'aktivis', 'menyoroti': 'sorot', 'pernyataan': 'nyata', 'pndah': 'pndah', 'dr': 'dr', 'jkt': 'jkt', 'beliau': 'beliau', 'cm': 'cm', 'pindah': 'pindah', 'merdeka': 'merdeka', 'selatan': 'selatan', 'utara': 'utara', 'bukabukaan': 'bukabukaan', 'obrolan': 'obrol', 'jam': 'jam', 'bareng': 'bareng', 'buzzerp': 'buzzerp', 'visible': 'visible', 'confusion': 'confusion', 'baswedan': 'baswedan', 'legitimasi': 'legitimasi', 'menilai': 'nilai', 'gubernur': 'gubernur', 'dki': 'dki', 'memiliki': 'milik', 'republik': 'republik', 'heboh': 'heboh', 'rumor': 'rumor', 'militer': 'militer', 'china': 'china', 'ditahan': 'tahan', 'rumah': 'rumah', 'tahanan': 'tahan', 'persaingan': 'saing', 'jenderal': 'jenderal', 'laporan': 'lapor', 'tentara': 'tentara', 'akrab': 'akrab', 'kekuasaan': 'kuasa', 'mendekat': 'dekat', 'rg': 'rg', 'didatangi': 'datang', 'diyakini': 'yakin', 'berkantor': 'kantor', 'medan': 'medan', 'kunjungi': 'kunjung', 'pastikan': 'pasti', 'cebongkampret': 'cebongkampret', 'pan': 'pan', 'berminat': 'minat', 'mengasuh': 'asuh', 'mending': 'mending', 'diasuh': 'asuh', 'partai': 'partai', 'amanat': 'amanat', 'nasional': 'nasional', 'peluang': 'peluang', 'selesai': 'selesai', 'jarangjarang': 'jarangjarang', 'lord': 'lord', 'menatap': 'tatap', 'pasca': 'pasca', 'via': 'via', 'part': 'part', 'tonton': 'tonton', 'ditakedown': 'ditakedown', 'klik': 'klik', 'kebebasan': 'bebas', 'freedom': 'freedom', 'natural': 'natural', 'right': 'right', 'konstitusi': 'konstitusi', 'tugas': 'tugas', 'pemerintah': 'perintah', 'melindungi': 'lindung', 'sambangi': 'sambang', 'pembahasannya': 'bahas', 'pesan': 'pesan', 'ramai': 'ramai', 'netizen': 'netizen', 'bersuara': 'suara', 'pertalite': 'pertalite', 'boros': 'boros', 'cepat': 'cepat', 'harganya': 'harga', 'coba': 'coba', 'simak': 'simak', 'videonya': 'video', 'terbagi': 'bagi', 'demo': 'demo', 'golput': 'golput', 'tempo': 'tempo', 'kritik': 'kritik', 'idolakan': 'idola', 'cuitan': 'cuit', 'viral': 'viral', 'menyebut': 'sebut', 'cebong': 'cebong', 'kampret': 'kampret', 'joko': 'joko', 'widodo': 'widodo', 'istilah': 'istilah', 'mengunjungi': 'unjung', 'masuk': 'masuk', 'angin': 'angin', 'temui': 'temu', 'om': 'om', 'pangeran': 'pangeran', 'ngobrol': 'ngobrol', 'tolong': 'tolong', 'cariin': 'cariin', 'cewe': 'cewe', 'doi': 'doi', 'grogi': 'grogi', 'ama': 'ama', 'cowo': 'cowo', 'kekekekkek': 'kekekekkek', 'ketemu': 'ketemu', 'lemes': 'lemes', 'saudara': 'saudara', 'rumahnya': 'rumah', 'kirakira': 'kirakira', 'prediksi': 'prediksi', 'gangguan': 'ganggu', 'terciptanya': 'cipta', 'hoax': 'hoax', 'kebodohan': 'bodoh', 'tim': 'tim', 'bocorkan': 'bocor', 'pembahasan': 'bahas', 'disangka': 'sangka', 'akun': 'akun', 'twitter': 'twitter', 'pribadinya': 'pribadi', 'menanggapi': 'tanggap', 'mementionnya': 'mementionnya', 'dll': 'dll', 'cemas': 'cemas', 'kadrun': 'kadrun', 'cemburu': 'cemburu', 'siapkan': 'siap', 'hadiah': 'hadiah', 'sepatu': 'sepatu', 'nasib': 'nasib', 'pensiun': 'pensiun', 'jabatannya': 'jabat', 'sinyal': 'sinyal', 'diganggu': 'ganggu', 'penguasa': 'kuasa', 'sprindik': 'sprindik', 'sowan': 'sowan', 'tegaskan': 'tegas', 'bapaknya': 'bapak', 'berkawan': 'kawan', 'botol': 'botol', 'minuman': 'minum', 'gagal': 'gagal', 'fokus': 'fokus', 'kabar': 'kabar', 'gembira': 'gembira', 'sayembara': 'sayembara', 'komentar': 'komentar', 'terlucu': 'lucu', 'hadiahnya': 'hadiah', 'tercengang': 'cengang', 'anak': 'anak', 'walikota': 'walikota', 'mengunggah': 'unggah', 'fotonya': 'foto', 'lei': 'lei', 'host': 'host', 'desain': 'desain', 'jelang': 'jelang', 'pilpres': 'pilpres', 'duga': 'duga', 'dipanggil': 'panggil', 'kpk': 'kpk', 'salah': 'salah', 'idola': 'idola', 'janjikan': 'janji', 'nawarin': 'nawarin', 'operasi': 'operasi', 'katarak': 'katarak', 'yaa': 'yaa', 'bhuehuehue': 'bhuehuehue', 'angkat': 'angkat', 'mengomentari': 'komentar', 'insiden': 'insiden', 'penembakan': 'tembak', 'brigadir': 'brigadir', 'dikepalai': 'palai', 'ferdy': 'ferdy', 'sambo': 'sambo', 'berita': 'berita', 'periode': 'periode', 'menko': 'menko', 'singgung': 'singgung', 'big': 'big', 'data': 'data', 'binsar': 'binsar', 'pandjaitan': 'pandjaitan', 'berbincang': 'bincang', 'contohi': 'contoh', 'amerika': 'amerika', 'usul': 'usul', 'sistem': 'sistem', 'pilkada': 'pilkada', 'dievalusi': 'dievalusi', 'tamparan': 'tampar', 'rakyat': 'rakyat', 'parpol': 'parpol', 'serang': 'serang', 'berebut': 'rebut', 'elektabilitas': 'elektabilitas', 'enak': 'enak', 'nontonnya': 'nontonnya', 'marves': 'marves', 'mustahil': 'mustahil', 'kalean': 'kalean', 'mesti': 'mesti', 'nonton': 'nonton', 'menteri': 'menteri', 'koordinator': 'koordinator', 'bidang': 'bidang', 'kemaritiman': 'maritim', 'investasi': 'investasi', 'usil': 'usil', 'bintang': 'bintang', 'podcast': 'podcast', 'jaman': 'jaman', 'rezim': 'rezim', 'skrg': 'skrg', 'kaya': 'kaya', 'bapa': 'bapa', 'tiri': 'tiri', 'sinetron': 'sinetron', 'kasih': 'kasih', 'makan': 'makan', 'bsu': 'bsu', 'blt': 'blt', 'anakrakyat': 'anakrakyat', 'udh': 'udh', 'kekenyangan': 'kenyang', 'trs': 'trs', 'pecutin': 'pecutin', 'nantikan': 'nanti', 'video': 'video', 'tayang': 'tayang', 'rabu': 'rabu', 'wib': 'wib', 'isu': 'isu', 'diperbincangkan': 'bincang', 'pembodohan': 'bodoh', 'kurikulum': 'kurikulum', 'etika': 'etika', 'setuju': 'tuju'}
    ------------------------
:::

::: {.output .display_data}
``` {.json}
{"version_major":2,"version_minor":0,"model_id":"0c9940a7760b48cbb3055943574ec1fe"}
```
:::

::: {.output .execute_result execution_count="21"}
    0                                [anies, tantang, said]
    1                                                    []
    2     [kode, opung, luhut, calon, presiden, acara, k...
    3     [rocky, gerung, bicara, langsung, perintah, jo...
    4                                                    []
                                ...                        
    95    [jaman, rezim, skrg, kaya, bapa, tiri, sinetro...
    96    [sobat, akal, sehat, nanti, video, part, tayan...
    97    [menteri, koordinator, bidang, maritim, invest...
    98             [bodoh, kurikulum, etika, rocky, gerung]
    99    [tuju, rocky, saksi, lengkap, youtube, rgtv, c...
    Name: tweet, Length: 100, dtype: object
:::
:::

::: {.cell .markdown id="TbNgmiGg6BPb"}
Setelah tahap stemming proses prepocessing sudah selesai, namun pada
dataset masih belum memiliki kelas atau label untuk itu akan dilakukan
pemberian label atau kelas dengan menggunakan nilai polarity.
:::

::: {.cell .markdown id="F7IYeeSyTp0n"}
## **Labelling Dataset**

Setelah proses prepocesing selesai didapat sebuah dataset yang masih
belum memiliki label, untuk itu pada tahapan ini dataset akan diberikan
kelas atau label yang sesuai. Akan tetapi tahap pelabelan ini akan
memerlukan waktu yang lama jika dilakukan secara manual. Untuk itu pada
tahapan ini saya memberikan kelas atau label pada masing-masing data
secara otomatis dengan menggunakan nilai polarity.
:::

::: {.cell .markdown id="QiXBSbymVPFS"}
### Nilai Polarity

Nilai polarity merupakan nilai yang menunjukkan apakah kata tersebut
bernilai negatif atau positif ataupun netral. Nilai polarity didapatkan
dengan menjumlahkan nilai dari setiap kata dataset yang menunjukkan
bahwa kata tersebut bernilai positif atau negatif ataupun
netral.`<br>`{=html} Didalam satu kalimat atau data,nilai dari kata-kata
didalam satu kalimat tersebut akan dijumlah sehingga akan didapatkan
nilai atau skor polarity. Nilai atau skor tersebutlah yang akan
menentukan kalimat atau data tersebut berkelas positif(pro) atau
negatif(kontra) ataupun netral.`<br>`{=html} Jika nilai polarity yang
didapat lebih dari 0 maka kalimat atau data tersebut diberi label atau
kelas pro. Jika nilai polarity yang didapat kurang dari 0 maka kalimat
atau data tersebut diberi label atau kelas kontra. Sedangkan jika nilai
polarity sama dengan 0 maka kalimat atau data tersebut diberi label
netral.
:::

::: {.cell .markdown id="O_6Pe2NyYKmo"}
### Ambil Nilai Polarity

Sebelum melakukan pemberian label atau kelas dengan menggunakan nilai
polarity, kita ambil nilai polarity dari setiap kata apakah positif atau
negatif. Untuk itu saya mengambil nilai polarity dari github yang di
dapat dari link github berikut <https://github.com/fajri91/InSet> Nilai
lexicon positif dan negatif yang didapat dari github tersebut saya
download kemudian saya upload ke github saya dan kemudian saya ambil
data lexicon positif dan negatif tersebut dengan source code berikut.
:::

::: {.cell .code id="pzGUy7zfgRMt"}
``` {.python}
positive = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/positive.csv")
positive.to_csv('lexpos.csv',index=False)
negative = pd.read_csv("https://raw.githubusercontent.com/Fahrur190125/Data/main/negative.csv")
negative.to_csv('lexneg.csv',index=False)
```
:::

::: {.cell .markdown id="GnAoPh-4acnx"}
### Menentukan Kelas/Label dengan Nilai Polarity

Setelah berhasil mengambil nilai polarity lexicon positif dan negatif
selanjutnya kita tentukan kelas dari masing masing data dengan
menjumlahkan nilai polarity yang didapat dengan ketentuan jika lebih
dari 0 maka memiliki kelas pro, jika kurang dari 0 maka diberi kelas
kontra, dan jika sama dengan 0 maka memiliki kelas netral, dengan source
code berikut.
:::

::: {.cell .code id="Uv2ZYWlmgXTS"}
``` {.python}
# Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)
# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('lexpos.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('lexneg.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'pro'
    elif (score < 0):
        polarity = 'kontra'
    else:
        polarity = 'netral'
    return score, polarity
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ydotQrR9gdwq" outputId="519b90d9-efab-4228-b3e8-019ee4dc7dc5"}
``` {.python}
# Results from determine sentiment polarity of tweets

results = tweets['tweet'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
tweets['polarity_score'] = results[0]
tweets['label'] = results[1]
print(tweets['label'].value_counts())
```

::: {.output .stream .stdout}
    pro       41
    kontra    40
    netral    19
    Name: label, dtype: int64
:::
:::

::: {.cell .markdown id="68i_wlN6bPak"}
Setelah didapat dataset yang sudah memiliki label selanjutnya kita
simpan dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Lio_ACIwgjGA" outputId="0fb90c55-4690-4a4f-eda7-c7ddf6d633cc"}
``` {.python}
# Export to csv file
tweets.to_csv('Prepocessing_label.csv',index=False)

tweets
```

::: {.output .execute_result execution_count="25"}
```{=html}
  <div id="df-ef362c48-dcc2-4653-b9b7-784b7b6e32f0">
    <div class="colab-df-container">
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
      <th>tweet</th>
      <th>polarity_score</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[anies, tantang, said]</td>
      <td>-4</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[]</td>
      <td>0</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[kode, opung, luhut, calon, presiden, acara, k...</td>
      <td>-2</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[rocky, gerung, bicara, langsung, perintah, jo...</td>
      <td>9</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[]</td>
      <td>0</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>[jaman, rezim, skrg, kaya, bapa, tiri, sinetro...</td>
      <td>3</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>96</th>
      <td>[sobat, akal, sehat, nanti, video, part, tayan...</td>
      <td>7</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>97</th>
      <td>[menteri, koordinator, bidang, maritim, invest...</td>
      <td>6</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>98</th>
      <td>[bodoh, kurikulum, etika, rocky, gerung]</td>
      <td>0</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>99</th>
      <td>[tuju, rocky, saksi, lengkap, youtube, rgtv, c...</td>
      <td>2</td>
      <td>pro</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ef362c48-dcc2-4653-b9b7-784b7b6e32f0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ef362c48-dcc2-4653-b9b7-784b7b6e32f0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ef362c48-dcc2-4653-b9b7-784b7b6e32f0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="ix3GeJpb6EcQ"}
## **Term Frequncy(TF)**

Term Frequency(TF) merupakan banyaknya jumlah kemunculan term pada suatu
dokumen. Untuk menghitung nilai TF terdapat beberapa cara, cara yang
paling sederhana ialah dengan menghitung banyaknya jumlah kemunculan
kata dalam 1 dokumen.`<br>`{=html} Sedangkan untuk menghitung nilai TF
dengan menggunakan mesin dapat menggunakan library sklearn dengan source
code berikut.
:::

::: {.cell .code colab="{\"height\":423,\"base_uri\":\"https://localhost:8080/\"}" id="FNBbXRiR6LH3" outputId="a504808e-20b8-410f-d3de-d428d36e8861"}
``` {.python}
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing_label.csv',index_col=False)
dataTextPre.drop("polarity_score", axis=1, inplace=True)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])
dataTextPre
```

::: {.output .execute_result execution_count="26"}
```{=html}
  <div id="df-88006bf5-2456-4d00-b6e3-ba8ed5e61693">
    <div class="colab-df-container">
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
      <th>tweet</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['anies', 'tantang', 'said']</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[]</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['kode', 'opung', 'luhut', 'calon', 'presiden'...</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['rocky', 'gerung', 'bicara', 'langsung', 'per...</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[]</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>['jaman', 'rezim', 'skrg', 'kaya', 'bapa', 'ti...</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>96</th>
      <td>['sobat', 'akal', 'sehat', 'nanti', 'video', '...</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>97</th>
      <td>['menteri', 'koordinator', 'bidang', 'maritim'...</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>98</th>
      <td>['bodoh', 'kurikulum', 'etika', 'rocky', 'geru...</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>99</th>
      <td>['tuju', 'rocky', 'saksi', 'lengkap', 'youtube...</td>
      <td>pro</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-88006bf5-2456-4d00-b6e3-ba8ed5e61693')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-88006bf5-2456-4d00-b6e3-ba8ed5e61693 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-88006bf5-2456-4d00-b6e3-ba8ed5e61693');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="T5M4Uotr6N-j"}
### Matrik VSM(Visual Space Model)

Sebelum menghitung nilai TF, terlebih dahulu buat matrik vsm untuk
menentukan bobot nilai term pada dokumen dengan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iKdRlqLh6RhK" outputId="8d043dfc-0f54-4303-ce7c-d1a3ce0b1d38"}
``` {.python}
matrik_vsm = bag.toarray()
#print(matrik_vsm)
matrik_vsm.shape
```

::: {.output .execute_result execution_count="27"}
    (100, 390)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7N_1nVqo6SCI" outputId="cbab0010-58a8-4416-c2ad-e8f6e9919eeb"}
``` {.python}
matrik_vsm[0]
```

::: {.output .execute_result execution_count="28"}
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
:::
:::

::: {.cell .markdown id="WAIVY1yO6VRf"}
Untuk menampilkan nilai TF yang didapat menggunakan source code berikut
:::

::: {.cell .code id="6CbHMZx76YLP"}
``` {.python}
a=vectorizer.get_feature_names()
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="g_6JDO_06ce-" outputId="5df542b8-7ccd-4beb-be8a-b92bd7ce70bc"}
``` {.python}
print(len(matrik_vsm[:,1]))
#dfb =pd.DataFrame(data=matrik_vsm,index=df,columns=[a])
dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF.to_csv('TF.csv',index=False)
dataTF
```

::: {.output .stream .stdout}
    100
:::

::: {.output .execute_result execution_count="30"}
```{=html}
  <div id="df-2c3f445f-1b86-455b-a860-6e20c115cbb4">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>acara</th>
      <th>airlangga</th>
      <th>ajak</th>
      <th>ajar</th>
      <th>akal</th>
      <th>akrab</th>
      <th>aktivis</th>
      <th>aku</th>
      <th>akun</th>
      <th>aliansi</th>
      <th>...</th>
      <th>viral</th>
      <th>visible</th>
      <th>wali</th>
      <th>walikota</th>
      <th>wib</th>
      <th>widodo</th>
      <th>xi</th>
      <th>yaa</th>
      <th>yakin</th>
      <th>youtube</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>96</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 390 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-2c3f445f-1b86-455b-a860-6e20c115cbb4')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-2c3f445f-1b86-455b-a860-6e20c115cbb4 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-2c3f445f-1b86-455b-a860-6e20c115cbb4');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="LB5l22Qs6fHk"}
### Nilai Term Dokumen

Setelah didapat nilai matrik vsm, selanjutnya tentukan nilai term pada
masing masing dokumen menggunakan source code berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="DWxm9WB16hmN" outputId="8b0b9d54-500a-4ce1-be3f-c5cbab589790"}
``` {.python}
datalabel = pd.read_csv('Prepocessing_label.csv',index_col=False)
TF = pd.read_csv('TF.csv',index_col=False)
dataJurnal = pd.concat([TF, datalabel["label"]], axis=1)
dataJurnal
```

::: {.output .execute_result execution_count="31"}
```{=html}
  <div id="df-32179969-56d7-4372-bfbe-f9249c6c892a">
    <div class="colab-df-container">
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
      <th>acara</th>
      <th>airlangga</th>
      <th>ajak</th>
      <th>ajar</th>
      <th>akal</th>
      <th>akrab</th>
      <th>aktivis</th>
      <th>aku</th>
      <th>akun</th>
      <th>aliansi</th>
      <th>...</th>
      <th>visible</th>
      <th>wali</th>
      <th>walikota</th>
      <th>wib</th>
      <th>widodo</th>
      <th>xi</th>
      <th>yaa</th>
      <th>yakin</th>
      <th>youtube</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>kontra</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>netral</td>
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
      <th>95</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>pro</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>netral</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>pro</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 391 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-32179969-56d7-4372-bfbe-f9249c6c892a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-32179969-56d7-4372-bfbe-f9249c6c892a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-32179969-56d7-4372-bfbe-f9249c6c892a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="KAwbQyXj6iSe"}
### Mengambil Data label

Setelah didapat nilai term pada masing masing dokumen kita ambil data
label pada masing masing dokumen.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="hTsJChBN6k5e" outputId="d3a4ac9a-d6ee-4741-877f-284a62fab8d8"}
``` {.python}
dataJurnal['label'].unique()
```

::: {.output .execute_result execution_count="32"}
    array(['kontra', 'netral', 'pro'], dtype=object)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2aKO8Tah6nyQ" outputId="ea2451f7-9716-436b-9b77-4cadb207da63"}
``` {.python}
dataJurnal.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Columns: 391 entries, acara to label
    dtypes: int64(390), object(1)
    memory usage: 305.6+ KB
:::
:::

::: {.cell .markdown id="2kx32AtJ6qAO"}
### Split Data

Selanjutnya kita split dataset menjadi data training dan testing dengan
source code berikut.
:::

::: {.cell .code id="XXQUBl1l6tX3"}
``` {.python}
### Train test split to avoid overfitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataJurnal.drop(labels=['label'], axis=1),
    dataJurnal['label'],
    test_size=0.15,
    random_state=0)
```
:::

::: {.cell .markdown id="Z7KCLE5TBn54"}
#### Data Training
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Dbe9ME676vzO" outputId="470076a3-e418-4efa-a755-34a15db70052"}
``` {.python}
X_train
```

::: {.output .execute_result execution_count="35"}
```{=html}
  <div id="df-98cef068-84d9-4fcd-916d-f390fa976528">
    <div class="colab-df-container">
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
      <th>acara</th>
      <th>airlangga</th>
      <th>ajak</th>
      <th>ajar</th>
      <th>akal</th>
      <th>akrab</th>
      <th>aktivis</th>
      <th>aku</th>
      <th>akun</th>
      <th>aliansi</th>
      <th>...</th>
      <th>viral</th>
      <th>visible</th>
      <th>wali</th>
      <th>walikota</th>
      <th>wib</th>
      <th>widodo</th>
      <th>xi</th>
      <th>yaa</th>
      <th>yakin</th>
      <th>youtube</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>96</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 390 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-98cef068-84d9-4fcd-916d-f390fa976528')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-98cef068-84d9-4fcd-916d-f390fa976528 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-98cef068-84d9-4fcd-916d-f390fa976528');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="DgtEcpmuBugt"}
#### Data Testing
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="HEEjMfksAEng" outputId="459004d8-7531-49b0-a34c-ee5865864cfa"}
``` {.python}
X_test
```

::: {.output .execute_result execution_count="36"}
```{=html}
  <div id="df-34aa2580-b06f-417f-b020-a97adbb49c8f">
    <div class="colab-df-container">
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
      <th>acara</th>
      <th>airlangga</th>
      <th>ajak</th>
      <th>ajar</th>
      <th>akal</th>
      <th>akrab</th>
      <th>aktivis</th>
      <th>aku</th>
      <th>akun</th>
      <th>aliansi</th>
      <th>...</th>
      <th>viral</th>
      <th>visible</th>
      <th>wali</th>
      <th>walikota</th>
      <th>wib</th>
      <th>widodo</th>
      <th>xi</th>
      <th>yaa</th>
      <th>yakin</th>
      <th>youtube</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 390 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-34aa2580-b06f-417f-b020-a97adbb49c8f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-34aa2580-b06f-417f-b020-a97adbb49c8f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-34aa2580-b06f-417f-b020-a97adbb49c8f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  
```
:::
:::

::: {.cell .markdown id="n1PQcVVjYmoB"}
Setelah didapat matrik VSM, selanjutnya lakukan metode Bagging,
Stacking, dan Random Forest Clasification dengan grid search seperti
berikut.
:::

::: {.cell .markdown id="vieT31JVVc1O"}
## **Bagging Classification**

Bagging merupakan metode yang dapat memperbaiki hasil dari algoritma
klasifikasi machine learning dengan menggabungkan klasifikasi prediksi
dari beberapa model. Hal ini digunakan untuk mengatasi ketidakstabilan
pada model yang kompleks dengan kumpulan data yang relatif kecil.
Bagging adalah salah satu algoritma berbasis ensemble yang paling awal
dan paling sederhana, namun efektif. Bagging paling cocok untuk masalah
dengan dataset pelatihan yang relatif kecil. Bagging mempunyai variasi
yang disebut Pasting Small Votes. cara ini dirancang untuk masalah
dengan dataset pelatihan yang besar, mengikuti pendekatan yang serupa,
tetapi membagi dataset besar menjadi segmen yang lebih kecil. Penggolong
individu dilatih dengan segmen ini, yang disebut bites, sebelum
menggabungkannya melalui cara voting mayoritas.`<br>`{=html}
`<center>`{=html}`<img src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Bagging.png">`{=html}`</center>`{=html}
`<center>`{=html}Gambar Bagging`</center>`{=html}`<br>`{=html} Bagging
mengadopsi distribusi bootstrap supaya menghasilkan base learner yang
berbeda, untuk memperoleh data subset. sehingga melatih base learners.
dan bagging juga mengadopsi strategi aggregasi output base leaner, yaitu
metode voting untuk kasus klasifikasi dan averaging untuk kasus regresi.
Untuk melakukan bagging pada data yang sudah di precocessing dengan
menngunakan libary skikit learn seperti berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="57UTch-jbcgx" outputId="a0db5758-f79a-459d-d8ab-084470ea32b9"}
``` {.python}
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
  
# load the data
X = X_train
Y = y_train
  
seed = 20
#kfold = model_selection.KFold(n_splits = 3,random_state = seed)
  
# initialize the base classifier
base_cls = DecisionTreeClassifier()

#Menyimpan Hasil Nilai Base Classifier
base_classifier=[]
hasilNilai_classifier=[]

for i in range (1, 50):
  # no. of base classifier
  num_trees = i
    
  # bagging classifier
  model = BaggingClassifier(base_estimator = base_cls,
                            n_estimators = num_trees,
                            random_state = seed)
    
  results = model_selection.cross_val_score(model, X, Y)

  #Nilai base classifier dan hasil nilai classifier disimpan dan akan ditampilkan di grafik
  base_classifier.append(i)
  hasilNilai_classifier.append(results.mean())

  print("accuracy :")
  print(results.mean())
```

::: {.output .stream .stdout}
    accuracy :
    0.4470588235294118
    accuracy :
    0.4235294117647059
    accuracy :
    0.5294117647058825
    accuracy :
    0.5294117647058824
    accuracy :
    0.5058823529411764
    accuracy :
    0.5764705882352942
    accuracy :
    0.5294117647058824
    accuracy :
    0.5058823529411764
    accuracy :
    0.49411764705882355
    accuracy :
    0.5176470588235295
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.5058823529411764
    accuracy :
    0.49411764705882355
    accuracy :
    0.5058823529411766
    accuracy :
    0.48235294117647065
    accuracy :
    0.49411764705882355
    accuracy :
    0.48235294117647065
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.5058823529411766
    accuracy :
    0.49411764705882355
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647065
    accuracy :
    0.5058823529411766
    accuracy :
    0.49411764705882355
    accuracy :
    0.5058823529411766
    accuracy :
    0.5058823529411766
    accuracy :
    0.5058823529411766
    accuracy :
    0.48235294117647065
    accuracy :
    0.49411764705882355
    accuracy :
    0.48235294117647065
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.49411764705882355
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647065
    accuracy :
    0.45882352941176474
    accuracy :
    0.48235294117647065
    accuracy :
    0.47058823529411764
    accuracy :
    0.48235294117647065
    accuracy :
    0.48235294117647054
    accuracy :
    0.47058823529411764
    accuracy :
    0.5058823529411766
:::
:::

::: {.cell .markdown id="bOdFSe-Ra1DL"}
Menampilkan data hasil akurasi yang di dapat dari metode bagging dengan
melakukan perulangan yang digrafikkan menggunakan Plot dari library
python sebagai berikut.
:::

::: {.cell .code colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="Clwwm8LVazZ8" outputId="5357c7b9-45b0-4b17-e452-0c9fd37ac590"}
``` {.python}
import matplotlib.pyplot as plt

plt.plot(base_classifier, hasilNilai_classifier)
plt.title('Hasil Nilai Bagging Classifier')
plt.xlabel('Nilai Estimator')
plt.ylabel('Hasil Akurasi')
plt.grid(True)
plt.show()
```

::: {.output .display_data}
![](vertopal_e77f8e028f43456d81c91f487fc386b0/28948eaa94c0753c0b6fc0b6054f220ca3d44ee6.png)
:::
:::

::: {.cell .markdown id="UA24bmx6Vj2f"}
## **Stacking Classification**

Stacking merupakan cara untuk mengkombinasi beberapa model, dengan
konsep meta learner. dipakai setelah bagging dan boosting. tidak seperti
bagging dan boosting, stacking memungkinkan mengkombinasikan model dari
tipe yang berbeda. Ide dasarnya adalah untuk train learner tingkat
pertama menggunakan kumpulan data training asli, dan kemudian
menghasilkan kumpulan data baru untuk melatih learner tingkat kedua, di
mana output dari learner tingkat pertama dianggap sebagai fitur masukan
sementara yang asli label masih dianggap sebagai label data training
baru. Pembelajar tingkat pertama sering dihasilkan dengan menerapkan
algoritma learning yang berbeda.

Dalam fase training pada stacking, satu set data baru perlu dihasilkan
dari classifier tingkat pertama. Jika data yang tepat yang digunakan
untuk melatih classifier tingkat pertama juga digunakan untuk
menghasilkan kumpulan data baru untuk melatih classifier tingkat kedua.
proses tersebut memiliki risiko yang tinggi yang akan mengakibatkan
overfitting. sehingga disarankan bahwa contoh yang digunakan untuk
menghasilkan kumpulan data baru dikeluarkan dari contoh data training
untuk learner tingkat pertama, dan prosedur crossvalidasi.`<br>`{=html}
`<center>`{=html}`<img src="https://upload.wikimedia.org/wikipedia/commons/d/de/Stacking.png">`{=html}`</center>`{=html}`<center>`{=html}Gambar
Stacking`</center>`{=html}`<br>`{=html} Berikut source code untuk
melakukan klasisikasi dengan algoritma Stackingt Classifikcaion
menggunakan library scikit-learn
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="-Sp3_MGrlDWR" outputId="fd0a19d6-ef55-4166-af8d-a97e27f04206"}
``` {.python}
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=20, random_state=42),'rf1', RandomForestClassifier(n_estimators=20, random_state=42)),
    ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=RandomForestClassifier()
)

clf.fit(X_train, y_train).score(X_test, y_test)
```

::: {.output .execute_result execution_count="39"}
    0.8
:::
:::

::: {.cell .markdown id="Ub_lncNTVm1W"}
## **Random Forest Classification**

Random forest (RF) adalah suatu algoritma yang digunakan pada
klasifikasi data dalam jumlah yang besar. Klasifikasi random forest
dilakukan melalui penggabungan pohon (tree) dengan melakukan training
pada sampel data yang dimiliki. Penggunaan pohon (tree) yang semakin
banyak akan mempengaruhi akurasi yang akan didapatkan menjadi lebih
baik. Penentuan klasifikasi dengan random forest diambil berdasarkan
hasil voting dari tree yang terbentuk. Pemenang dari tree yang terbentuk
ditentukan dengan vote terbanyak. Pembangunan pohon (tree) pada random
forest sampai dengan mencapai ukuran maksimum dari pohon data. Akan
tetapi,pembangunan pohon random forest tidak dilakukan pemangkasan
(pruning) yang merupakan sebuah metode untuk mengurangi kompleksitas
ruang. Pembangunan dilakukan dengan penerapan metode random feature
selection untuk meminimalisir kesalahan. Pembentukan pohon (tree) dengan
sample data menggunakan variable yang diambil secara acak dan
menjalankan klasifikasi pada semua tree yang terbentuk. Random forest
menggunakan Decision Tree untuk melakukan proses seleksi. Pohon yang
dibangun dibagi secara rekursif dari data pada kelas yang sama.
Pemecahan (split) digunakan untuk membagi data berdasarkan jenis atribut
yang digunakan. Pembuatan decision tree pada saat penentuan
klasifikasi,pohon yang buruk akan membuat prediksi acak yang saling
bertentangan. Sehingga,beberapa decision tree akan menghasilkan jawaban
yang baik. Random forest merupakan salah satu cara penerapan dari
pendekatan diskriminasi stokastik pada klasifikasi. Proses Klasifikasi
akan berjalan jika semua tree telah terbentuk.Pada saat proses
klasifikasi selesai dilakukan, inisialisasi dilakukan dengan sebanyak
data berdasarkan nilai akurasinya. Keuntungan penggunaan random forest
yaitu mampu mengklasifiksi data yang memiliki atribut yang tidak
lengkap,dapat digunakan untuk klasifikasi dan regresi akan tetapi tidak
terlalu bagus untuk regresi, lebih cocok untuk pengklasifikasian data
serta dapat digunakan untuk menangani data sampel yang banyak. Proses
klasifikasi pada random forest berawal dari memecah data sampel yang ada
kedalam decision tree secara acak. Setelah pohon terbentuk,maka akan
dilakukan voting pada setiap kelas dari data sampel. Kemudian,
mengkombinasikan vote dari setiap kelas kemudian diambil vote yang
paling banyak.Dengan menggunakan random forest pada klasifikasi data
maka, akan menghasilkan vote yang paling baik.`<br>`{=html}
`<center>`{=html}`<img src='https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png'>`{=html}`</center>`{=html}`<center>`{=html}Gambar
Random Forest`</center>`{=html}`<br>`{=html} Berikut source code untuk
melakukan klasisikasi dengan algoritma Random Forest Classifikcaion
menggunakan library scikit-learn
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="N4RsMmTtnAM5" outputId="a577054e-60c5-4caf-c8db-6ab74bc4eec5"}
``` {.python}
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

::: {.output .stream .stdout}
    Accuracy: 0.6666666666666666
:::
:::

::: {.cell .markdown id="jwOuiapaaNK-"}
## **Grid Search**

Grid Search adalah sebuah function yang terdapat pada library
Scikit-Learn. Function ini dapat membantu untuk mengulang melalui
hyperparameter yang telah ditentukan dan menyesuaikan estimator (model)
Anda pada data set pelatihan. Pada kali ini saya akan menggunakan Grid
Search untuk membantu menemukan nilai estimator atau nilai yang terbaik
sehingga nilai dari base classifier mendapatkan hasil akurasi yang
terbaik pada metode Bagging dan Random Forest Classification.
:::

::: {.cell .markdown id="oFX2fC7bea5u"}
### Bagging Classification dengan menggunakan Grid Search

Penggunaan Grid Search pada metode Bagging Classification untuk
menemukan nilai estimator terbaik sehingga menghasilkan akurasi yang
terbaik dapat dilakukan sebagai berikut.
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="l2HImXjTei8R" outputId="8114dccb-ec36-41ea-93e5-ffb6be47fef1"}
``` {.python}
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'base_estimator__max_depth' : [4, 8, 12, 16, 20]
}

X = X_train
Y = y_train

clf = GridSearchCV(BaggingClassifier(DecisionTreeClassifier(),
                                     n_estimators = 50, max_features = 0.5), param_grid)
 
results = model_selection.cross_val_score(clf, X, Y)
print("accuracy :")
print(results.mean())
```

::: {.output .stream .stdout}
    accuracy :
    0.5058823529411764
:::
:::

::: {.cell .markdown id="J4C_Lrz-ugrm"}
### Random Forest Classification dengan menggunakan Grid Search

Penggunaan Grid Search pada metode Random Forest Classification untuk
menemukan nilai estimator terbaik sehingga menghasilkan akurasi yang
terbaik dapat dilakukan sebagai berikut.
:::

::: {.cell .code id="evByGxBzuf7R"}
``` {.python}
from sklearn.model_selection import GridSearchCV

hyper_params = {'max_depth': [3, 5, 10, 15, 20],
                'max_features': [3, 5, 7, 11, 15],
                'min_samples_leaf': [20, 50, 100, 200, 400],
                'n_estimators': [10, 25, 50, 80, 100]
                }
```
:::

::: {.cell .code id="jMqq6UyIvh6p"}
``` {.python}
#Grid search
model_cv = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=hyper_params,
                        verbose=1,
                        cv=5,
                        n_jobs=1,
                        return_train_score=True)
```
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="OtaK-7pvv_Hc" outputId="1470eaf8-cd59-4222-dff9-7f0d3474cefe"}
``` {.python}
model_cv.fit(X_train, y_train)
```

::: {.output .stream .stdout}
    Fitting 5 folds for each of 625 candidates, totalling 3125 fits
:::

::: {.output .execute_result execution_count="44"}
    GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=1,
                 param_grid={'max_depth': [3, 5, 10, 15, 20],
                             'max_features': [3, 5, 7, 11, 15],
                             'min_samples_leaf': [20, 50, 100, 200, 400],
                             'n_estimators': [10, 25, 50, 80, 100]},
                 return_train_score=True, verbose=1)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="csirdWJuwFMh" outputId="a6db0aea-dbdf-4833-a3fd-9627a3737e66"}
``` {.python}
GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=1,
             param_grid={'max_depth': [3, 5, 10, 15, 20],
                         'max_features': [3, 5, 7, 11, 15],
                         'min_samples_leaf': [20, 50, 100, 200, 400],
                         'n_estimators': [10, 25, 50, 80, 100]},
             return_train_score=True, verbose=1)
```

::: {.output .execute_result execution_count="45"}
    GridSearchCV(cv=5, estimator=RandomForestClassifier(), n_jobs=1,
                 param_grid={'max_depth': [3, 5, 10, 15, 20],
                             'max_features': [3, 5, 7, 11, 15],
                             'min_samples_leaf': [20, 50, 100, 200, 400],
                             'n_estimators': [10, 25, 50, 80, 100]},
                 return_train_score=True, verbose=1)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="JiuKsvcvwICL" outputId="6da9c0e7-5665-417b-a41b-ab48b2185b79"}
``` {.python}
model_cv.best_score_
```

::: {.output .execute_result execution_count="46"}
    0.48235294117647054
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="MO0Rl6Icrd5j" outputId="15e1190f-679c-4af6-c2a9-661f85618746"}
``` {.python}
model_cv.best_estimator_
```

::: {.output .execute_result execution_count="48"}
    RandomForestClassifier(max_depth=10, max_features=11, min_samples_leaf=20,
                           n_estimators=50)
:::
:::

::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Dj1Bl6mIrjDW" outputId="f71f01bf-5049-4e64-b1da-3c8dfcdd3893"}
``` {.python}
#Pengimplementasian best estimator hasil dari GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
RandomForestClassifier(max_depth=10, max_features=11, min_samples_leaf=20,
                       n_estimators=50)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# prediction on test set
y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

::: {.output .stream .stdout}
    Accuracy: 0.4666666666666667
:::
:::

::: {.cell .markdown id="MXX5RlATTVtZ"}
## **Kesimpulan**

Berdasar dari hasil atau nilai akurasi yang di dapat dari semua metode
ensemble learning, metode Stacking Classification menghasilkan nilai
akurasi yang paling baik dengan nilai akurasi sebesar 80% dibandingkan
dengan metode Random Forest Classification sebesar 66% dan metode
Bagging Classification sebesar 57%. Dan dari hasil akurasi yang
diperoleh dari metode Bagging dan Random Forest Classification dengan
menggunakan Grid Search memperoleh hasil atau nilai akurasi yang lebih
buruk dibandingkan tanpa menggunakan Grid Search.`<br>`{=html}

Sehingga dapat disimpulkan bahwa penggunaan Grid Searh pada metode
Bagging dan Random Forest Classification pada data Twitteer dengan
pencarian \'\#rockygerung\' menunjukkan bahwa penggunaan Grid Searh
tidak dapat meningkatkan nilai akurasi yang diperoleh, akan tetapi
membuat nilai akurasi yang di dapat semakin buruk. Oleh karena itu
penggunaan Grid Search pada metode Bagging dan Random Forest
Classification tidak begitu berpengaruh terhadap peningkatan akurasi
yang diperoleh.
:::
