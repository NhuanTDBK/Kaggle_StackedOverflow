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


def clean_sentence(sentence):
    sentence = sentence.lower()
    string = re.sub(r"\'s", "", sentence)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub("&gt","",string)
    string = re.sub("&lt","",string)
    string = re.sub("&amp","",string)
    string = re.sub(r"[\"'-@\(\)\[\]%\^*+=!\]\[\{\};:\,\.]", " ",string)

    #string = re.sub(r"[=\"\'\?/!@#$%^&\*\_\\(\)\[\]\{\}`~\n\t\+\$\^],\."," ",string)
    
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    # print string
    return string

def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


for df in df_list:
    df['text'] = df['text'] + " "+  df['content']
    
texts =  pd.concat([_['text'] for _ in df_list],ignore_index=True)


class Texts(object):
    def __init__(self, df):
       self.df = df 
 
    def __iter__(self):
        for item in self.df:
            blob = TextBlob(item)
            for sentence in blob.sentences:
                # print sentence
                cleaned = clean_sentence(str(sentence))
                words = cleaned.split(" ")
                
                yield words
print "start learning"
text_iter = Texts(texts)
from gensim.models.phrases import Phrases
bigram = Phrases(text_iter)
trigram  = Phrases(bigram[text_iter])
trigram.save("./trigram.model")
#model = Word2Vec(text_iter, size=100, window=5, min_count=1, workers=8)
#model.save("./w2v_all.model")
# test_ratio = 0.3

# for df in df_list:
#     msk = np.random.rand(len(df)) < test_ratio
#     test = df[msk]
#     train = df[~msk]



# tags = reduce(lambda x,y: x + " " + y, train_bio['tags'])
# tags = list(set(tags.split(" ")))
# tags = map(lambda x: x.replace("-", ' '),tags)
# text_title = reduce(lambda x,y: x + " " + y, train_bio['title'])
# text_content = reduce(lambda x,y: x + " " + y, train_bio['content'])

# text = text_title + " " + text_content



# for df in df_list:
#     tags = reduce(lambda x,y: x + " " + y, df['tags'])
#     tags = list(set(tags.split(" ")))
#     tags = map(lambda x: x.replace("-", ' '),tags)
#     text_title = reduce(lambda x,y: x + " " + y, df['title'])
#     text_content = reduce(lambda x,y: x + " " + y, df['content'])
#     text = text_title + " " + text_content
#     counter = 0
#     for i in tags:
#         rex = "[^\w]"+i+"[^\w]"
#         if len (re.findall(rex,text)) > 0:
#             counter  += 1
#     print counter * 1.0 / len(tags)

