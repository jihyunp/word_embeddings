from datetime import datetime
from load_data import *
from gensim.models import Word2Vec
import cPickle as cp
import os

model_name = 'rcv1_skipgram'
data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
result_dir = '/extra/jihyunp0/research/word_embeddings/result/' + model_name
skip_gram = 1  # 0 for CBOW

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# Load list[list[str]], which is a list of sentence,
# and the sentence consists of words as list.
sentences = load_sentences(data_dir)
print('Data Loaded')
print('Number of sentences: ' + str(len(sentences)))
print('Number of tokens: ' + str(sum(map(len, sentences))))
print(datetime.now())

print('Train started')
print(datetime.now())
model = Word2Vec(sentences, window=5, min_count=20, size=200, workers=8, sg=skip_gram)
print('finished')
print(datetime.now())

print('saving model')
model_fname = result_dir + '/gensim_word2vec_2008_2015.model'
model.save(model_fname)

print('saving vectors in text')
vectors_fname = result_dir + '/gensim_word2vec_2008_2015.vectors'
vocab_fname = result_dir + '/gensim_word2vec_2008_2015.vocabs'
model.save_word2vec_format(vectors_fname, fvocab= vocab_fname, binary=False)
print(datetime.now())


print('Saving Sentences.. This will take a long time.. ')
sentences_file = result_dir + '/sentences.pkl'
cp.dump(sentences, open(sentences_file, 'wb'), protocol=cp.HIGHEST_PROTOCOL)
print('Done')
print(datetime.now())
