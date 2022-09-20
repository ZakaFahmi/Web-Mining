#!/usr/bin/env python
# coding: utf-8

# # data-crawling-twitter

# In[1]:


get_ipython().system('pip install tweepy')


# importkan data atau librari yang di butuhkan

# In[2]:


import tweepy
import json
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import csv
import nltk
from nltk.tokenize import word_tokenize
import re


# masukkan api tweet developer anda lakukan crowling data anda

# In[3]:


consumer_key = 'QaSkYai54p3YJ3e2DNNcQPdVH'
consumer_secret = '82a7qI4suRJ0aR1TjkiVSRTBKnmA9n6Cunwc79QcJUUy4vnVLQ'
access_token = '1098381907-rNHpAcYmnIt7pb5K8bwhbh5nPQAfU244dr4FyS4'
access_secret = 'eZL092xX7CW9t9d0S5c3QCcia3qGY275lHG8zbo2GWxxY'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
    
# Setup access API
def connectOAuth():
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Connecting tweet with JSON
def process_or_store(tweet):
    print(json.dumps(tweet))
    
# Create API Object
api = connectOAuth()

# This class to create class from  Stream Listener
# The result exported in file python.json
class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('hastag.json','a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data : %s"%str(e))
        return True
   
    def on_error(self, status):
        print(status)
        return True


# ##data-perayapan-twitter 
# 
# ini ditujukan untuk menampilkan data
