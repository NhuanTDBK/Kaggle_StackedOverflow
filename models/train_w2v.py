# -*- encoding: utf-8 -*-

import pandas as pd 
import numpy as np 
import re
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec

# Load the data.
print "load data"
train_bio = pd.read_csv("../input_light/biology.csv",encoding='utf-8')
train_cooking = pd.read_csv("../input_light/cooking.csv",encoding='utf-8')
train_crypto = pd.read_csv("../input_light/crypto.csv",encoding='utf-8')
train_dyi = pd.read_csv("../input_light/diy.csv",encoding='utf-8')
train_robotic = pd.read_csv("../input_light/robotics.csv",encoding='utf-8')
train_travel = pd.read_csv("../input_light/travel.csv",encoding='utf-8')
test_df = pd.read_csv("../input_light/test.csv",encoding='utf-8')

df_list = [train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel,test_df]



for df in df_list:
    df['text'] = df['title'] +" "+  df['content']
    
texts =  pd.concat([_['text'] for _ in df_list],ignore_index=True)


class Texts(object):
    def __init__(self, df):
        self.df = df 
    
    def __iter__(self):
            for item in self.df:
                words = str(item).split(" ")
                yield words

print "start learning"

text_iter = Texts(texts)
from gensim.models.phrases import Phrases , Phraser
tri = Phraser.load("./trigram.model")
model = Word2Vec(tri[text_iter], size=100, window=5, min_count=1, workers=8)
model.save("./w2v_all.model")
