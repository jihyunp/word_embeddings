import os, re
from datetime import datetime
from gensim.models import Word2Vec
import urllib
import bz2
import cPickle as cp
from Word2VecUtils import Word2VecData, parse_string, \
    load_sentences_in_text_file, save_output_vectors


class WikiData(Word2VecData):

    def __init__(self, data_dir='./data', word2vec_file='./word2vec.vectors', binary=True,
                 vdim=200, window=5, sg=True, min_count=20, workers=5):

        Word2VecData.__init__(self, data_dir=data_dir)
        self.window, self.min_count, self.vdim = window, min_count, vdim
        self.workers, self.sg = workers, sg
        self._load_data(data_dir)
        self._load_word2vec(word2vec_file, binary)

    def _load_data(self, data_dir):
        t1 = datetime.now()
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        data_url = "http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        self._download_and_uncompress(data_url)
        self._preprocess_data()  # Generates self.sentences, self.n_sents, self.n_words

        print('Took %.2f seconds to load the data.' % (datetime.now() - t1).seconds)
        print('Number of sentences: ' + str(self.n_sents))
        print('Number of tokens: ' + str(self.n_words) + '\n')

    def _download_and_uncompress(self, url):
        file_name = url.split('/')[-1]
        file_path = os.path.join(self.data_dir, file_name)
        uncompressed_file_path = re.sub("\.bz2$", "", file_path)

        if not os.path.isfile(file_path):
            print("Downloading " + file_name + " from " + url + " ...")
            print("This might take a long time. (13.19GB)")
            urllib.urlretrieve(url, file_path)

        if not os.path.exists(uncompressed_file_path):
            if file_name.split('.')[-1] == 'bz2':
                print("Un-compressing data " + file_name)
            bzfile = bz2.BZ2File(file_path)
            uncompressed_data = bzfile.readlines()
            f = open(uncompressed_file_path, 'w')
            f.writelines(uncompressed_data)
            f.close()
        else:
            print('Uncompressed data already exists. If you want to re-extract data, delete the file: '
                   + uncompressed_file_path + '\n')

        self.uncompressed_file_path = uncompressed_file_path

    def _preprocess_data(self):
        # Run the perl script ------> Later convert the script into python so it save time!
        import subprocess
        t1 = datetime.now()
        if not os.path.exists('./wikifil_jp.pl'):
            raise('ERROR: Script wikifil_jp.pl does not exist! ')
        outfile_path = re.sub(".xml", "-parsed.txt", self.uncompressed_file_path)
        if not os.path.exists(outfile_path):
            print('Parsing the xml file with perl script.. This might take upto 2 hours.')
            with open(outfile_path, 'w') as outfile:
                command = ['perl', './wikifil_jp.pl' ]
                subprocess.call(command, stdout=outfile)
            print('Parsing done.')
        self.sentences = load_sentences_in_text_file(outfile_path, sep=". ")
        self.n_sents = len(self.sentences)
        self.n_words = sum(map(len, self.sentences))

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
        outputv_fname = fname + '.output.vectors'

        # If the result folder does not exist, generate one.
        res_folder = os.path.dirname(vectors_fname)
        if not os.path.isdir(res_folder):
            os.makedirs(res_folder)

        print('Saving vectors and vocabs')
        self.w2v_model.save_word2vec_format(vectors_fname, fvocab=vocab_fname, binary=binary)
        # Saving output vectors
        save_output_vectors(self.w2v_model, outputv_fname, binary=binary)
        print(datetime.now())

        print('Saving model as pickle')
        cp.dump(self.w2v_model, open(model_fname, 'wb'), protocol=cp.HIGHEST_PROTOCOL)
        print('Done saving')
        print(datetime.now())