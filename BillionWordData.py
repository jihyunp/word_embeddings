import json
import os
import collections
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from random import sample
from gensim.models import Word2Vec
import urllib
import tarfile
import cPickle as cp
from Word2VecUtils import Word2VecData, parse_string, load_sentences_in_text_file


class BillionWordData(Word2VecData):

    def __init__(self, data_dir='./data', word2vec_file='./word2vec.vectors', binary=True,
                 vdim=200, window=5, sg=True, min_count=20, workers=5):
        """
        Parameters
        ----------
        data_dir: str
            Root data directory that the data will be downloaded and unpacked.
        word2vec_file : str
            xx.vectors file that you want to load the w2v model from,
            or the file you want to save the vectors to.
            If the file does not exist, it will start training and save the
            .vectors file in this location. It will also save .vocab and .model.pkl files as well.
        binary : bool
            Default True.
            Save/Load binary vectors
        vdim : int
            Dimension of the vectors
        window : int
            Word2vec window length. Default 5. 5 previous and 5 later words are considered.
        sg : bool
            Skipgram if True (default)
            CBOW if False
        min_count : int
            Minimum count of words that will be considered for training.
            Default 20.
        workers : int
            Number of threads
            Default 5.
        """

        Word2VecData.__init__(self, data_dir)
        self.window, self.min_count, self.vdim = window, min_count, vdim
        self.workers, self.sg = workers, sg
        self._load_data(data_dir)
        self._load_word2vec(word2vec_file, binary)

    def _load_data(self, data_dir):
        # The data is almost already parsed
        t1 = datetime.now()

        def _download_and_uncompress(url):
            file_name = url.split('/')[-1]
            file_path = os.path.join(self.data_dir, file_name)
            folder_path = os.path.join(self.data_dir, file_name.split('.')[0])

            if not os.path.isfile(file_path):
                print("Downloading " + file_name + " from " + url + " ...")
                urllib.urlretrieve(url, file_path)

            if not os.path.isdir(folder_path):
                if file_name.split('.')[-1] == 'gz':
                    print("Un-compressing data " + file_name)
                    if file_name.split('.')[-2] == 'tar':
                        tar = tarfile.open(file_path, "r:gz")
                        tar.extractall(self.data_dir)
                        tar.close()
            else:
                print('Data directory already exists. If you want to re-extract data, delete the folder: '
                       + folder_path + '\n')

        # First see if it exists, else download the data
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        data_url = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
        _download_and_uncompress(data_url)
        file_name = data_url.split('/')[-1]
        folder_path = os.path.join(data_dir, file_name.split('.')[0])

        data = []
        for dir, subdir, files in os.walk(folder_path):
            for fname in files:
                if fname.startswith('news'):
                    fpath = os.path.join(dir, fname)
                    tmpdata = load_sentences_in_text_file(fpath)
                    data.extend(tmpdata)
                    self.n_docs += 1
                    self.n_sents += len(tmpdata)
                    self.n_words += sum(map(len, tmpdata))

        self.sentences = data
        print('Took %.2f seconds to load the data.' % (datetime.now() - t1).seconds)
        print('Number of sentences: ' + str(self.n_sents))
        print('Number of tokens: ' + str(self.n_words) + '\n')

    def _train_word2vec(self, word2vec_file, binary):
        print('Training Started')
        t1 = datetime.now()
        print(t1)
        self.w2v_model= Word2Vec(self.sentences, window=self.window, min_count=self.min_count, size=self.vdim,
                                 workers=self.workers, sg=self.sg)
        print('Took %.2f seconds to train the model.' % (datetime.now() - t1).seconds)
        vectors_fname = word2vec_file
        fname = word2vec_file.split('.vectors')[0]
        model_fname = fname + '.pkl'
        vocab_fname = fname + '.vocab'

        # If the result folder does not exist, generate one.
        res_folder = os.path.dirname(vectors_fname)
        if not os.path.isdir(res_folder):
            os.makedirs(res_folder)

        print('Saving vectors and vocabs')
        self.w2v_model.save_word2vec_format(vectors_fname, fvocab=vocab_fname, binary=binary)
        print(datetime.now())

        print('Saving model as pickle')
        cp.dump(self.w2v_model, open(model_fname, 'wb'), protocol=cp.HIGHEST_PROTOCOL)
        print('Done saving')
        print(datetime.now())






