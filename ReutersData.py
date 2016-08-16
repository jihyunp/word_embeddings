import gzip
import os
import urllib
import re

import numpy as np
from scipy.sparse import find

# from ProcessData import ProcessData
# from ProcessData import _split_data

class ReutersData():

    def __init__(self, orig_data_dir, output_dir, train_valid_split=(1, 0), sup_unsup_split=(1, 0),
                 train_test_split=(1, 0), shuffle=False, random_seed=1234):
        '''

        Parameters
        ----------
        orig_data_dir
        output_dir
        train_valid_split
        sup_unsup_split
        train_test_split
        shuffle
        random_seed
        '''

        self.orig_data_dir = orig_data_dir
        self.output_dir = output_dir

        self.shuf = shuffle
        self.random_seed = random_seed

        if self.shuf:
            self.shuffled_idx = None

        self.train_raw_text = None
        self.train_sup_raw_text = None
        self.test_raw_text = None
        self.valid_raw_text = None
        self.unsup_raw_text = None

        self.train_docids = None
        self.train_sup_docids = None
        self.valid_docids = None
        self.test_docids = None
        self.unsup_docids = None

        self.train_x = None
        self.train_y = None

        self.train_sup_x = None
        self.train_sup_y = None
        self.valid_x = None
        self.valid_y = None
        self.test_x = None
        self.test_y = None
        self.unsup_x = None

        self.train_valid_split = train_valid_split
        self.sup_unsup_split = sup_unsup_split
        self.train_test_split = train_test_split
        self.vocab = None

        self.binary = False

        self.topic_raw_data = None
        self.topic_new_data = None

    def _download_and_uncompress(self, url):
        '''
        Used inside the 'get_raw_data' function if needed.

        Returns
        -------
        String, or a list of string

        '''
        file_name = url.split('/')[-1]
        file_path = os.path.join(self.orig_data_dir, file_name)

        if not os.path.isfile(file_path):
            print("Downloading " + file_name + " from " + url + " ...")
            urllib.urlretrieve(url, file_path)

        if file_name.split('.')[-1] == 'gz':
            print("Un-compressing data " + file_name)
            zip = gzip.open(file_path, 'rb')
            content = zip.read()
        else:
            f = open(file_path, 'r')
            content = f.read()
        return content


    def get_raw_data(self):

        test_urls = ['http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz']
        # test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz')
        # test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz')
        # test_urls.append('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz')
        train_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'

        # 103 RCV1 Topics categories
        topics_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt'

        # Topic hierarchy
        topic_hier_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig'

        # specifies which Topic categories each RCV1-v2 document belongs to.
        topic_doc_url = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz'

        if not os.path.isdir(self.orig_data_dir):
            os.makedirs(self.orig_data_dir)

        # Get test raw data
        print('Processing test data')
        test_data = []
        test_did = []
        # Download the dataset
        for url in test_urls:
            content = self._download_and_uncompress(url)
            content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
            for doc in content_arr:
                id_and_doc = doc.strip().split(' .W ')
                test_data.append(id_and_doc[1])
                test_did.append(int(id_and_doc[0]))

        # Get the train raw data
        print('Processing train data')
        train_data = []
        train_did = []
        content = self._download_and_uncompress(train_url)
        content_arr = re.sub('\n', ' ', content.strip()).split('.I ')[1:]
        for doc in content_arr:
            id_and_doc = doc.strip().split(' .W ')
            train_data.append(id_and_doc[1])
            train_did.append(int(id_and_doc[0]))


        # Get topic data
        print('Processing topic information')
        td_content = self._download_and_uncompress(topic_doc_url)
        td_content_arr = td_content.strip().split('\n')
        tid2topic = []
        topic2tid = {}
        did2tid = {}
        tid = 0
        for line in td_content_arr:
            topic = line.split(' ')[0]
            docid = int(line.split(' ')[1])
            if topic not in tid2topic:
                tid2topic.append(topic)
                topic2tid[topic] = tid
                did2tid[docid] = tid
                tid += 1
            else:
                tid2 = topic2tid[topic]
                did2tid[docid] = tid2

        # Get topic hierarchies
        th_content = self._download_and_uncompress(topic_hier_url)
        th_content_arr = th_content.strip().split('\n')
        topic_child2parent = {}
        topic_desc = {}
        for line in th_content_arr:
            sp_line = re.split('\s+', line)
            parent = sp_line[1]
            child = sp_line[3]
            child_dsc = sp_line[5]
            topic_child2parent[child] = parent
            topic_desc[child] = child_dsc

        self.topic_raw_data = (tid2topic, topic2tid, did2tid, topic_child2parent, topic_desc)

        self.train_raw_text = train_data
        self.train_docids = train_did
        self.test_raw_text = test_data
        self.test_docids = test_did
        self.unsup_raw_text = unsup_data
        self.unsup_docids = unsup_did




    def map_parent_categories(self):

        print('Mapping child to parent categories (topics).. Generating new set of \'topic_raw\'')
        old_id2cat, old_cat2id, old_did2cid, topic_child2parent, old_topic_desc = self.topic_raw_data

        newid = 0
        oldtid2newtid = []
        new_id2cat = []
        new_cat2id = {}
        new_topic_desc = []
        new_did2tid = {}

        for child in topic_child2parent:
            parent = topic_child2parent[child]
            if parent == 'Root':
                new_id2cat.append(child)
                new_cat2id[child] = newid
                new_topic_desc.append(old_topic_desc[child])
                newid += 1

        for oldid, child in enumerate(old_id2cat):
            parent = child
            while parent not in new_id2cat:
                tmp = topic_child2parent[child]
                if tmp == 'None' or tmp == 'Root':
                    break
                parent = tmp
                child = parent
            newid2 = new_cat2id[parent]
            oldtid2newtid.append(newid2)

        # get did2tid
        for did in old_did2cid:
            old_cid = old_did2cid[did]
            newtid = oldtid2newtid[old_cid]
            new_did2tid[did] = newtid

        self.topic_new_data = (new_id2cat, new_cat2id, new_did2tid, topic_child2parent, new_topic_desc)


    def get_matrices(self):

        print('Generating BOW matrices for train/test set')
        from sklearn.feature_extraction.text import CountVectorizer

        # Get BOWs for train/test
        train_text = self.train_raw_text
        train_did = self.train_docids
        unsup_text = self.unsup_raw_text
        unsup_did = self.unsup_docids
        test_text = self.test_raw_text
        test_did = self.test_docids

        count_vec = CountVectorizer()
        count_vec = count_vec.fit(train_text + unsup_text)
        train_bow = count_vec.transform(train_text)
        unsup_bow = count_vec.transform(unsup_text)
        test_bow = count_vec.transform(test_text)
        vocab = count_vec.vocabulary_
        vocab_rev = {y: x for x, y in vocab.iteritems()}

        # Change vocab order
        sorted_vocab_idx = np.argsort(np.array(train_bow.sum(axis=0))[0])[::-1]
        self.train_x = train_bow[:, sorted_vocab_idx]
        self.unsup_x = unsup_bow[:, sorted_vocab_idx]
        self.test_x = test_bow[:, sorted_vocab_idx]
        new_vocab = [vocab_rev[i] for i in sorted_vocab_idx]
        self.vocab = new_vocab

        print('Generating labels for train/test set')
        # Get topic labels for train/test set
        did2tid = self.topic_new_data[2]
        self.train_y = np.array([did2tid[did] for did in train_did], dtype=np.int16)
        self.test_y = np.array([did2tid[did] for did in test_did], dtype=np.int16)

        # SPlit the data
        self.train_sup_x, self.valid_x = _split_data(self.train_x, self.train_valid_split)
        self.train_sup_y, self.valid_y = _split_data(self.train_y, self.train_valid_split)
        self.train_sup_docids, self.valid_docids = _split_data(self.train_docids, self.train_valid_split)

    def save_and_print_data(self):
        # from copy import copy
        import cPickle as cp

        train_folder = os.path.join(self.output_dir, 'train')
        test_folder = os.path.join(self.output_dir, 'test')
        valid_folder = os.path.join(self.output_dir, 'valid')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)
        if not os.path.isdir(valid_folder):
            os.makedirs(valid_folder)

        train_sup_bow_file = os.path.join(train_folder, 'labeledBow.feat')
        unsup_bow_file = os.path.join(train_folder, 'unsupBow.feat')
        valid_bow_file = os.path.join(valid_folder, 'labeledBow.feat')
        test_bow_file = os.path.join(test_folder, 'labeledBow.feat')

        cp_file = os.path.join(self.output_dir, 'dataset.pkl')
        dataset = ((self.train_sup_x, self.train_sup_y), (self.valid_x, self.valid_y), (self.unsup_x, []), (self.test_x, self.test_y))
        print('Saving X matrices and label arrays to dataset.pkl ..')
        cp.dump(dataset, open(cp_file, 'wb'), protocol=cp.HIGHEST_PROTOCOL)

        self.print_svmlight_format(self.train_sup_x, self.train_sup_y, train_sup_bow_file)
        self.print_svmlight_format(self.unsup_x, np.zeros(self.unsup_x.shape[0], dtype=np.int8), unsup_bow_file, unsup=True)
        self.print_svmlight_format(self.valid_x, self.valid_y, valid_bow_file)
        self.print_svmlight_format(self.test_x, self.test_y, test_bow_file)

        # Printing vocabulary
        self.print_vocab(self.output_dir + '/vocab')

    def print_topics(self, output_file):

        id2cat = self.topic_new_data[0]
        id2desc = self.topic_new_data[4]

        print('Printing topic (category) information')
        outfile = open(output_file, 'w')
        id = 0
        for cat, desc in zip(id2cat, id2desc):
            outfile.write('{}\t{}\t{}\n'.format(id, cat, desc))
        outfile.close()

    def _print_text(self, text_data, doc_ids, output_file):

        outfile = open(output_file, 'w')
        for did, text in zip(doc_ids, text_data):
            line = '_*' + str(did) + ' ' + text + '\n'
            outfile.write(line.encode("utf-8"))
        outfile.close()

    def print_text_file(self, output_folder=None):
        '''
        This is for https://github.com/hiyijian/doc2vec
        '''
        if output_folder is not None:
            if output_folder != self.output_dir:
                self.output_dir = output_folder

        train_folder = os.path.join(self.output_dir, 'train')
        valid_folder = os.path.join(self.output_dir, 'valid')
        test_folder = os.path.join(self.output_dir, 'test')

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)
        if not os.path.isdir(valid_folder):
            os.makedirs(valid_folder)
        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        print('Printing text file in doc2vec form')
        # Now printing text
        train_output_file = os.path.join(train_folder, 'train_sup_text.txt')
        self._print_text(self.train_sup_raw_text, self.train_sup_docids, train_output_file)

        unsup_output_file = os.path.join(train_folder, 'train_unsup_text.txt')
        self._print_text(self.unsup_raw_text, self.unsup_docids, unsup_output_file)

        valid_output_file = os.path.join(valid_folder, 'valid_sup_text.txt')
        self._print_text(self.valid_raw_text, self.valid_docids, valid_output_file)

        test_output_file = os.path.join(test_folder, 'test_text.txt')
        self._print_text(self.test_raw_text, self.test_docids, test_output_file)



if __name__ == "__main__":

    obj = ReutersData('./rcv1-v2_orig', './rcv1-v2')
    obj.get_raw_data()
    # obj.map_parent_categories()
    # obj.get_matrices()
    # obj.save_and_print_data()
    # obj.print_topics(obj.output_dir + '/rcv1.topics')
    # obj.print_text_file()

