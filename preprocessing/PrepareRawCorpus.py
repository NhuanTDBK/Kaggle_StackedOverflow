
# coding: utf-8

# In[1]:

from gensim.models.phrases import Phrases
from gensim.models import Word2Vec
import pandas as pd
import nltk
import os
import seaborn as sb


# In[97]:

listdir = [r.split('.')[0] for r in os.listdir('../input_raw/')]
corpus = []
# Remove double-blank space in corpus
def remove_blank_space_batch(batch):
    results = []
    for lst in batch:
        result = [item for item in lst if item != '']
        results.append(result)
    return results

for filename in listdir:
#     filename = "robotics"
    #     print filename
    dat = pd.read_csv("../input_raw/%s.csv"%filename)
    dat.dropna(how='any',inplace=True)
    remove_blank_space = lambda d: str(d).rstrip().replace('\n','')
    dat['title'] = dat['title'].map(remove_blank_space)
    dat['content'] = dat['content'].map(remove_blank_space)
    # dat['doc'] = dat['title'] +' '+dat['content'] + ' ' + dat['tags']
    dat['doc'] = dat['title'] +' '+dat['content']
    if 'tags' in dat.columns:
	dat['doc'] = dat['doc'] + ' ' +dat['tags']
    def get_trigram(dat):
            sentences = [nltk.word_tokenize(w) for w in dat]
            bigram = Phrases(sentences,delimiter='-')
            trigram = Phrases(bigram[sentences],delimiter='-')
    #         quadgram = Phrases(trigram[sentences],delimiter='-')
    #         pentagram = Phrases(quadgram[sentences],delimiter='-')
            return trigram

    trigram = get_trigram(dat['doc'])
    sentences = [trigram[sent.rstrip().split(' ')] for sent in dat['doc']]
    corpus.extend(remove_blank_space_batch(sentences))
# In[24]:
# Set values for various parameters
num_features = 512   # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 5          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
training_algo = 1
# Initialize and train the model (this will take some time)
print "Training Word2Vec model..."
model = Word2Vec(corpus,sg=training_algo, workers=num_workers,iter=20,
           size=num_features,window = context, sample = downsampling, seed=1)
# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)
# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "w2v_model_dim%s_iter_20"%num_features
model.save(model_name)


# In[ ]:



