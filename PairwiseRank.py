
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as snb
import nltk
from gensim.models import Word2Vec, Phrases
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import re
import string
import gensim
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:

dir_name = 'input_not_lowercase/'
train_bio = pd.read_csv('%s/biology.csv'%dir_name,encoding='utf-8')
train_cooking = pd.read_csv("%s/cooking.csv"%dir_name,encoding='utf-8')
train_crypto = pd.read_csv("%s/crypto.csv"%dir_name,encoding='utf-8')
train_dyi = pd.read_csv("%s/diy.csv"%dir_name,encoding='utf-8')
train_robotic = pd.read_csv("%s/robotics.csv"%dir_name,encoding='utf-8')
train_travel = pd.read_csv("%s/travel.csv"%dir_name,encoding='utf-8')
# # test_df = pd.read_csv("input_light/test.csv",encoding='utf-8')

# df_list = train_bio
df_list = pd.concat([train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel])
df_list['doc'] = df_list['title'] + ' ' + df_list['content']
del train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel


# In[3]:

trigram = Phrases.load('trigram_all_vocab_tokenize.model')
w2v = Word2Vec.load('pretrain/w2v_model_dim300')


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
    POS_tag = nltk.pos_tag(trigram[split_token(content)])
    result = []
    for tag in POS_tag:
        if tag[-1] in 'NN,NNS,NNP,JJ':
            if tag[-1] not in string.punctuation:
                result.append(tag[0])
    return set(result)


# In[57]:

sample_dat = df_list
negative_sampling = 4
training = []


# In[58]:

for idx in range(len(sample_dat)):
    content = sample_dat['doc'].iloc[idx]
    tag = sample_dat['tags'].iloc[idx].split(' ')
    content2vec = doc2vec(content)
    content_tokens = split_token(content)
    count_negative = 0
    for candidate in get_candidate_tags(content):
        label = -1 
        if candidate in tag:
            label = 1
    #         else:
    #             if count_negative > negative_sampling:
    #                 continue
    #             count_negative +=1
        if w2v.vocab.has_key(candidate):
            tag2vec = w2v[candidate]
            idx = content_tokens.index(candidate)
            delta = 1.0/(idx+1) * (content2vec-tag2vec)
            training.append([delta,label])
#             target.append(label)


# In[59]:

df = pd.DataFrame(training,columns=['vec','label'])


# In[60]:

labels_count = df.label.value_counts()


# In[61]:

ratio = labels_count.iloc[1]*1.0/labels_count.iloc[0]


# In[62]:

df_negative = df[df.label<0].sample(frac=ratio)
df_positive = df[df.label>0]
df_balanced = shuffle(df_negative.append(df_positive))


# In[63]:

X = df_balanced['vec'].tolist()
y = df_balanced['label'].tolist()
X_train,X_val,y_train,y_val = train_test_split(X,y,train_size=0.8,random_state=4111)


# In[64]:

np.savez("pairwise_rank",X=X,y=y)


# In[65]:

linear_model = LogisticRegression(C=10000,warm_start=True,penalty='l2',max_iter=100)
linear_model.fit(X_train,y_train)
linear_model.score(X_val,y_val)


# In[151]:

# sentences = [nltk.word_tokenize(s) for s in df_list['doc']]
# bigram = Phrases(sentences,min_count=36,delimiter='-')
# trigram = Phrases(bigram[sentences],min_count=20,delimiter='-')
# trigram.save('trigram_all_vocab_tokenize.model')


# In[ ]:



