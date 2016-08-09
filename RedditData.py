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


class RedditData():

    def __init__(self, data_dir='./reddit_data_MH',
                 word2vec_file='./word2vec_reddit_2008_2015_20/gensim_word2vec_2008_2015_orig.vectors'):
        t1 = datetime.now()
        self.data_dir = data_dir
        self.json_data, self.subreddit2did, self.link2did = self._load_reddit_data(data_dir)
        self._get_subreddit_info()
        self._update_stats()
        self._load_word2vec(word2vec_file)
        print('Took %.2f seconds to load.' % (datetime.now() - t1).seconds)

    def _update_stats(self):
        self.n_comments = len(self.json_data)
        self.n_subreddits = len(self.subreddit2did)
        self.n_posts = len(self.link2did)

    def _get_subreddit_info(self):
        self.sid2subreddit = self.subreddit2did.keys()
        self.subreddit2sid = {y:x for x, y in enumerate(self.sid2subreddit)}

    def _load_word2vec(self, word2vec_file):
        self.w2v_model = Word2Vec.load_word2vec_format(word2vec_file, binary=False)

    def _load_reddit_data(self, data_dir):
        data = self._load_json(data_dir)
        return self._load_subreddit_link_info(data)

    def _load_subreddit_link_info(self, json_data):
        """
        Used inside of '_load_reddit_data()'
        Extract the subreddit & link information
        Also update the json data (exclude the data that are shorter than 15 words)
        :param json_data:
        :return:
        """
        print('Extracting the subreddit and link information')
        subreddit2idx = defaultdict(list)
        link2idx = defaultdict(list)
        new_data = []
        idx = 0
        for d in json_data:
            body = d['body']
            if body == '[deleted]':
                continue
            if len(body.split()) < 15:
                continue
            new_data.append(d)
            subred, linkid = self.extract_subreddit_link_info(d)
            subreddit2idx[subred].append(idx)
            link2idx[linkid].append(idx)
            idx += 1
        return new_data, subreddit2idx, link2idx

    def _load_json(self, data_dir='./reddit_data_MH'):
        """
        Used inside of '_load_reddit_data()'
        :param data_dir:
        :return:
        """
        data = []
        for dir, subdir, files in os.walk(data_dir):
            for fname in files:
                if fname.startswith('RC'):
                    fpath = os.path.join(dir, fname)
                    print('Loading json file ' + fpath)
                    tmpdata = self.load_single_json(fpath)
                    data.extend(tmpdata)
        return data

    @staticmethod
    def extract_subreddit_link_info(single_data):
        """
        For a single json data (in dict), extract its subreddit and link_id (orig post id).
        :param single_data:
        :return:
        """
        subreddit = single_data["subreddit"]
        link_id = single_data["link_id"]
        return (subreddit, link_id)

    def load_single_json(self, file_name):
        """
        Return a list of dictionaries from one file

        :param file_name:
        :return: list[dict]
        """
        f = open(file_name, 'r')
        data = []
        for line in f:
            data.append(json.loads(line))
        return data

    def get_random_doc_within_subreddit(self, subreddit):
        idxs = self.subreddit2did[subreddit]
        idx = idxs[sample(range(len(idxs)), 1)[0]]
        data = self.json_data[idx]
        doc = data["body"]
        return self.get_parsed_doc(doc)

    def get_random_doc_within_post(self, link_id):
        idxs = self.link2did[link_id]
        idx = idxs[sample(range(len(idxs)), 1)[0]]
        data = self.json_data[idx]
        doc = data["body"]
        return self.get_parsed_doc(doc)

    def get_random_doc_outside_subreddit(self, subreddit):
        subred_id = self.subreddit2sid[subreddit]
        other_sids = range(self.n_subreddits)
        other_sids.remove(subred_id)
        sid = sample(other_sids, 1)[0]
        return self.get_random_doc_within_subreddit(self.sid2subreddit[sid])

    @staticmethod
    def get_parsed_doc(doc, remove_stopwords=True):
        tmpstr = doc
        tmpstr = tmpstr.lower()
        tmpstr = re.sub('\[deleted\]', ' ', tmpstr)
        tmpstr = re.sub('\\n', ' ', tmpstr)
        tmpstr = re.sub('\\\'', "'", tmpstr)
        tmpstr = re.sub('[^a-z0-9\-.\' ]', ' ', tmpstr)
        tmpstr = re.sub(' ---*', ' ', tmpstr)
        tmpstr = re.sub(r'\s+', ' ', tmpstr)
        tmpstr = re.sub(r'\.\.+', '. ', tmpstr)  # Remove multiple periods
        tmpstr = re.sub(" \'", " ", tmpstr)
        tmpstr = re.sub("\' ", " ", tmpstr)
        tmpstr = re.sub("\'$", "", tmpstr)
        tmpstr = re.sub(" -", " ", tmpstr)
        tmpstr = re.sub("- ", " ", tmpstr)
        tmpstr = re.sub("-$", "", tmpstr)

        wordlist = tmpstr.split()
        if remove_stopwords:
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
            wordlist = [w for w in wordlist if w not in stop_words]
        return wordlist

    def get_wmdistance(self, doc1, doc2):
        return self.w2v_model.wmdistance(doc1, doc2)

