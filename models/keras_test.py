#//Author: Vuongnm at vuongnguyen710@gmail.com
# //                            _
# //                         _ooOoo_
# //                        o8888888o
# //                        88" . "88
# //                        (| -_- |)
# //                        O\  =  /O
# //                     ____/`---'\____
# //                   .'  \\|     |//  `.
# //                  /  \\|||  :  |||//  \
# //                 /  _||||| -:- |||||_  \
# //                 |   | \\\  -  /'| |   |
# //                 | \_|  `\`---'//  |_/ |
# //                 \  .-\__ `-. -'__/-.  /
# //               ___`. .'  /--.--\  `. .'___
# //            ."" '<  `.___\_<|>_/___.' _> \"".
# //           | | :  `- \`. ;`. _/; .'/ /  .' ; |
# //           \  \ `-.   \_\_`. _.'_/_/  -' _.' /
# // ===========`-.`___`-.__\ \___  /__.-'_.'_.-'================

# from __future__ import print_function
#this script show that tags can be predicted with high f1 score on just title of the text
#vanilla lstm model 

import json
import os
import pandas as pd 
import numpy as np 
import re
import ast

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
import keras 
# from keras.engine import Input
# from keras.layers import Embedding, merge
# from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import base_filter
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, Dropout,GlobalMaxPooling1D
from sklearn.preprocessing import MultiLabelBinarizer

print "load data"
train_bio = pd.read_csv("../input_light/biology.csv",encoding='utf-8')
train_cooking = pd.read_csv("../input_light/cooking.csv",encoding='utf-8')
train_crypto = pd.read_csv("../input_light/crypto.csv",encoding='utf-8')
train_dyi = pd.read_csv("../input_light/diy.csv",encoding='utf-8')
train_robotic = pd.read_csv("../input_light/robotics.csv",encoding='utf-8')
train_travel = pd.read_csv("../input_light/travel.csv",encoding='utf-8')
test_df = pd.read_csv("../input_light/test.csv",encoding='utf-8')

df_list = [train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel]


df_train = pd.DataFrame(data=[],columns=train_bio.columns)
df_test = pd.DataFrame(data=[],columns=train_bio.columns)

df = pd.DataFrame(data=[],columns=train_bio.columns)

for dfi in df_list:
    msk = np.random.rand(len(dfi)) < 0.8
    train = dfi[msk]
    test =  dfi[~msk]

    df_train = pd.concat([df_train,train]).reset_index(drop=True)
    df_test  = pd.concat([df_test,test]).reset_index(drop=True)
    df = pd.concat([df,dfi]).reset_index(drop=True)



batch_size = 32
nb_filter = 250
filter_length = 3 #most tags is from 1 to 3 word
hidden_dims = 250

tokenizer = Phraser.load("./trigram.model")

def title_tokens(row):
    toks = tokenizer[row['title'].split(" ")]
    return toks






X = train_cooking['title']
X = [ i.encode("utf-8") for i in X]

tk = keras.preprocessing.text.Tokenizer(nb_words=None, filters=base_filter(), lower=True, split=" ") 
tk.fit_on_texts(X)
X = tk.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen=30)

Y = train_cooking['tags'].values
Y = [item.replace("-","_").split(" ") for item in Y]
all_tags = list(set(sum(Y,[])))
transformer = MultiLabelBinarizer()
Y = transformer.fit_transform(Y)

model = Sequential()

#keras built-in embedding, should be replaced with pre-loaded gensim w2v
model.add(Embedding( len(tk.word_index.keys())+1, 100, dropout=0.2))

#lstm is great, also yield a decent result 
# model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun


model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(GlobalMaxPooling1D())

model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(len(all_tags)))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam', lr=0.1, #test learning rate, too high for adam
              metrics=['fmeasure']) #must use f mesure, accuracy sucks



model.fit(X, Y, batch_size=batch_size, nb_epoch=200)
