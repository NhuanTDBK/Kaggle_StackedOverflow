
# coding: utf-8

# In[1]:

from gensim.models.phrases import Phrases
import pandas as pd
import nltk
import os
import seaborn as sb


# In[97]:

listdir = [r.split('.')[0] for r in os.listdir('input_raw/')]
corpus = []
def remove_blank_space_batch(batch):
    results = []
    for lst in batch:
        result = [item for item in lst if item != '']
        results.append(result)
    return results


# In[ ]:

for filename in listdir:
#     filename = "robotics"
    #     print filename
    dat = pd.read_csv("input_raw/%s.csv"%filename)
    dat.dropna(how='any',inplace=True)
    remove_blank_space = lambda d: str(d).rstrip().replace('\n','')
    dat['title'] = dat['title'].map(remove_blank_space)
    dat['content'] = dat['content'].map(remove_blank_space)
    # dat['doc'] = dat['title'] +' '+dat['content'] + ' ' + dat['tags']
    dat['doc'] = dat['title'] +' '+dat['content']

    def get_trigram(dat):
            sentences = [nltk.word_tokenize(w) for w in dat]
            bigram = Phrases(sentences,delimiter='-')
            trigram = Phrases(bigram[sentences],delimiter='-')
    #         quadgram = Phrases(trigram[sentences],delimiter='-')
    #         pentagram = Phrases(quadgram[sentences],delimiter='-')
            return trigram

    trigram = get_trigram(dat['doc'])
    sentences = [trigram[sent.rstrip().split(' ')] for sent in dat['doc']]
    corpus.append(remove_blank_space_batch(sentences))


# In[24]:

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Word2Vec(sentences, workers=num_workers,
           size=num_features, min_count = min_word_count,
           window = context, sample = downsampling, seed=1)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "w2v_model"
model.save(model_name)


# In[ ]:



