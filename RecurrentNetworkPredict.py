
# coding: utf-8

# In[28]:

import numpy as np
import pandas as pd
import nltk
from gensim.models import Word2Vec, Phrases
from nltk.tag.stanford import StanfordPOSTagger
# import matplotlib.pyplot as plt
import string
import gensim
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.models import load_model
from utils.tools import *


# In[2]:

dir_name = 'input_not_lowercase/'
train_bio = pd.read_csv('%s/biology.csv'%dir_name,encoding='utf-8')
# train_cooking = pd.read_csv("%s/cooking.csv"%dir_name,encoding='utf-8')
# train_crypto = pd.read_csv("%s/crypto.csv"%dir_name,encoding='utf-8')
# train_dyi = pd.read_csv("%s/diy.csv"%dir_name,encoding='utf-8')
# train_robotic = pd.read_csv("%s/robotics.csv"%dir_name,encoding='utf-8')
# train_travel = pd.read_csv("%s/travel.csv"%dir_name,encoding='utf-8')
# # # test_df = pd.read_csv("input_light/test.csv",encoding='utf-8')

# # df_list = train_bio
# df_list = pd.concat([train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel])
# df_list['doc'] = df_list['title'] + ' ' + df_list['content']
# del train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel


# In[3]:

trigram = Phrases.load('trigram_model_new.model')
# w2v = Word2Vec.load('pretrain/w2v_model_dim300')
tagger = StanfordPOSTagger('../stanford-postagger-full-2016-10-31/models/english-bidirectional-distsim.tagger','../stanford-postagger-full-2016-10-31/stanford-postagger.jar')


# In[4]:

def split_token(doc_str):
    str_concat = nltk.word_tokenize(' '.join(doc_str.splitlines()))
    result = trigram[str_concat]
    return result
def doc2vec(doc_str):
    str_concat = ' '.join(doc_str.splitlines()).split(' ')
    result =[ w2v[v.lower()] for v in str_concat if w2v.vocab.has_key(v.lower())]
    return np.mean(result,axis=0)
def get_candidate_tags(doc_str):
    POS_tag = tagger.tag(trigram[split_token(content)])
    result = []
    for tag in POS_tag:
        if tag[-1] in 'NN,NNS,NNP':
            if tag[-1] not in string.punctuation:
                result.append(tag[0])
    return set(result)


# In[5]:




# In[47]:

dict2word = {w: idx for idx,w in enumerate(trigram.vocab)}


# In[ ]:

labels = []
training = []
for idx in range(len(train_bio)):
    tokens = [dict2word[t] for t in trigram[split_token(train_bio.title.iloc[idx])]]
    tags = train_bio.tags.iloc[idx].split()
    tag2rank = {t: idx for t, idx in zip(tags,range(len(tags)))}
    targets = [tag2rank[t] if t in tags else -1 for t in tokens]
    training.append(tokens)
    labels.append(targets)


# In[53]:

from sklearn.model_selection import train_test_split


# In[54]:

X_train,X_val,y_train,y_val = train_test_split(training,labels,train_size=0.8,random_state=4111)


# In[57]:

np.savez("is13/demo_bio",X_train=X_train,X_val=X_val,y_train=y_train,y_val=y_val,dict2word=dict2word)


# In[ ]:

dat = np.load('is13/demo_bio.npz')['dict2word']


# In[ ]:




# In[ ]:




# In[ ]:



