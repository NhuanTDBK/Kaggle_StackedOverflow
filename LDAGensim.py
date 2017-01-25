
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as snb
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import re
import gensim
from sklearn.externals import joblib


# In[ ]:

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
del train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel


# In[ ]:

dir_name = 'input_light/'
train_bio = pd.read_csv('%s/biology.csv'%dir_name,encoding='utf-8')
train_cooking = pd.read_csv("%s/cooking.csv"%dir_name,encoding='utf-8')
train_crypto = pd.read_csv("%s/crypto.csv"%dir_name,encoding='utf-8')
train_dyi = pd.read_csv("%s/diy.csv"%dir_name,encoding='utf-8')
train_robotic = pd.read_csv("%s/robotics.csv"%dir_name,encoding='utf-8')
train_travel = pd.read_csv("%s/travel.csv"%dir_name,encoding='utf-8')
# # test_df = pd.read_csv("input_light/test.csv",encoding='utf-8')

# df_list = train_bio
df_list_light = pd.concat([train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel])
del train_bio,train_cooking,train_crypto,train_dyi,train_robotic,train_travel


# In[ ]:

# df_list = pd.read_csv("input_light/total_dat_reformat.csv")
df_list['doc'] = df_list['title'] +' '+df_list['content']
df_list_light['doc'] = df_list_light['title'] +' '+df_list_light['content']


# In[ ]:

# df_list['tags'] = df_list['tags'].map(lambda d: d.replace("-","_"))


# In[ ]:

reg_alphabet = "[a-zA-Z]{3,}"
def tokenizer_words(demo,arr=True):
    a = []
    for f in demo.split(' '):
        if bool(re.match(reg_alphabet,f)):
            a.append(f)
    if arr==True:
        return a
    else:
        return ' '.join(a)
def remove_blank_space(token_str):
    result = []
    for t in token_str.split():
        if t != '':
            if '\n' in t:
                result.append('\n')
            result.append(t)
    return result
def get_keyword(tokens):
    result = nltk.pos_tag(nltk.word_tokenize(' '.join(tokens)))
    keywords = []
    for keyword, pos in result:
        if pos in 'NN,NNS,NNP':
#             print "a"
            keywords.append((keyword,pos))
    return keywords
def get_trigram(dat):
    sentences = [nltk.word_tokenize(w) for w in dat]
    bigram = gensim.models.Phrases(sentences,delimiter='-',min_count=35)
    trigram = gensim.models.Phrases(bigram[sentences],delimiter='-')
#         quadgram = Phrases(trigram[sentences],delimiter='-')
#         pentagram = Phrases(quadgram[sentences],delimiter='-')
    return trigram
def get_ngram_tokenizer(ngram,token):
    result = nltk.word_tokenize(' '.join(ngram[remove_blank_space(token)]))
    return result


# In[ ]:

sentence_stream = [nltk.word_tokenize(senten) for senten in df_list['doc']]
bigram = gensim.models.Phrases(sentence_stream,min_count=40,threshold=2)

trigram = gensim.models.Phrases(bigram[sentence_stream])


# # Sklearn LDA model

# In[ ]:

# sentence_stream_str = [tokenizer_words(senten,arr=False) for senten in df_list['doc']]


# In[ ]:

# ngram_tokenizer = lambda d: bigram[d.split(' ')]
ngram_tokenizer = lambda d: get_ngram_tokenizer(trigram,d)


# In[ ]:

tfIdfVect = CountVectorizer(tokenizer=ngram_tokenizer)


# In[ ]:

corpus = tfIdfVect.fit_transform(df_list['doc'])


# In[ ]:

lda = joblib.load("lda_14.model")


# In[ ]:

# n_topics = 14
# lda = LatentDirichletAllocation(n_jobs=1,n_topics=n_topics,learning_method='online')
# lda.fit(corpus)


# In[ ]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# In[ ]:

n_top_words = 40
tf_feature_names = tfIdfVect.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# In[ ]:

predicted =[]
actual = []
word2topic = np.array(lda.components_).T
dat = df_list_light
candidates = []
max_negative = 10
for idx in range(len(dat)):
#     idx = 0
#     print idx
    item = dat['title'].iloc[idx]
    content = dat['content'].iloc[idx]
    # print get_keyword(ngram_tokenizer(item))
    # print get_keyword(ngram_tokenizer(content))
    tags = dat['tags'].iloc[idx].split(' ')
    actual.append(tags)
    #Keyword title
    doc2vec = corpus[idx].toarray()[0]
    title_token = trigram[remove_blank_space(item)]
    
    term_tf_title = {word: doc2vec[tfIdfVect.vocabulary_.get(word.replace('-','_'))] for word in title_token if tfIdfVect.vocabulary_.has_key(word)}
    #Keyword content
    doc2vec_content = corpus[idx].toarray()[0]
    content_token = get_keyword(trigram[remove_blank_space(content)])
    term_tf_content = {word: doc2vec_content[tfIdfVect.vocabulary_.get(word.replace('-','_'))] for word in content_token if tfIdfVect.vocabulary_.has_key(word)}
    #LDA factor title
    topic_id = np.argmax(lda.transform(tfIdfVect.transform([item])))
    topic_lda = {word: word2topic[tfIdfVect.vocabulary_[word.replace('-','_')],topic_id] for word in title_token if tfIdfVect.vocabulary_.has_key(word)}
    max_val = max(topic_lda.itervalues(), key=lambda k:k)
    # Extend terms
    terms = term_tf_title.keys()
    terms.extend(term_tf_content.keys())
    terms.extend(topic_lda.keys())

    # sorted(term_tf_title.items(), key = lambda x: x[1], reverse=True)[0:10]

    # sorted(term_tf_content.items(), key = lambda x: x[1], reverse=True)[0:10]
    counter_negative = 0

    terms_w = {key: 0 for key in terms}
    for key in terms_w:
        title_w = term_tf_title.get(key,0.1)
        content_w = term_tf_content.get(key,0.2)
        lda_ld = topic_lda.get(key,0.1)/max_val
        label = 0
        if key.replace('-','_') in tags:
            label = 1
        elif counter_negative > max_negative:
            continue
        counter_negative +=1
        candidates.append([title_w,content_w,lda_ld,label])
# candidates = sorted(terms_w.items(), key = lambda x: x[1], reverse=True)
# print candidates
# predicted.append(map(str,np.array(candidates)[:,0].tolist()))


# In[ ]:

np.savez("candidates",X=candidates)

