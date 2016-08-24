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
        self.data_dir = data_dir
        self.n_docs, self.n_subreddits, self.n_posts = None, None, None
        self.w2v_model = None
        self.json_data = None
        self.subreddit2did, self.link2did = None, None
        self.name2did, self.did2name = None, None

        t1 = datetime.now()
        self._load_reddit_data(data_dir)
        self._get_subreddit_info()
        self._update_stats()
        self._load_word2vec(word2vec_file)
        print('Took %.2f seconds to load.' % (datetime.now() - t1).seconds)

    def _update_stats(self):
        self.n_docs = len(self.json_data)
        self.n_subreddits = len(self.subreddit2did)
        self.n_posts = len(self.link2did)

    def _get_subreddit_info(self):
        self.sid2subreddit = self.subreddit2did.keys()
        self.subreddit2sid = {y:x for x, y in enumerate(self.sid2subreddit)}

    def _load_word2vec(self, word2vec_file):
        self.w2v_model = Word2Vec.load_word2vec_format(word2vec_file, binary=False)

    def _load_reddit_data(self, data_dir):
        data = self._load_json(data_dir)
        self._load_subreddit_link_info(data)

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
        idx2name = []
        name2idx = defaultdict(int)
        new_data = []
        idx = 0
        for d in json_data:
            body = d['body']
            if body == '[deleted]':
                continue
            if len(body.split()) < 15:
                continue
            new_data.append(d)
            name, subred, linkid = self.extract_subreddit_link_info(d)
            subreddit2idx[subred].append(idx)
            link2idx[linkid].append(idx)
            name2idx[name] = idx
            idx2name.append(name)
            idx += 1

        self.json_data = new_data
        self.subreddit2did, self.link2did = subreddit2idx, link2idx
        self.name2did = name2idx
        self.did2name = idx2name

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
        name = single_data["name"]
        subreddit = single_data["subreddit"]
        link_id = single_data["link_id"]
        return (name, subreddit, link_id)

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

    def get_random_doc_within_subreddit(self, subreddit, name_to_exclude=None):
        idxs = self.subreddit2did[subreddit]
        if len(idxs) == 1:
            return []
        else:
            idxs_copy = list(idxs)
            if name_to_exclude is not None:
                did_to_exclude = self.name2did[name_to_exclude]
                del(idxs_copy[idxs_copy.index(did_to_exclude)])
            idx = idxs_copy[sample(range(len(idxs_copy)), 1)[0]]
            doc= self.json_data[idx]["body"]
            return self.parse_string(doc)

    def get_random_doc_within_post(self, link_id, name_to_exclude):
        idxs = self.link2did[link_id]
        if len(idxs) == 1:
            return []
        else:
            idxs_copy = list(idxs)
            did_to_exclude = self.name2did[name_to_exclude]
            del(idxs_copy[idxs_copy.index(did_to_exclude)])
            idx = idxs_copy[sample(range(len(idxs_copy)), 1)[0]]
            doc = self.json_data[idx]["body"]
            return self.parse_string(doc)

    def get_random_doc_outside_subreddit(self, subreddit):
        subred_id = self.subreddit2sid[subreddit]
        other_sids = range(self.n_subreddits)
        other_sids.remove(subred_id)
        sid = sample(other_sids, 1)[0]
        return self.get_random_doc_within_subreddit(self.sid2subreddit[sid], None)

    @staticmethod
    def parse_string(doc, remove_stopwords=True):
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

    def get_three_scores(self, n_test=1000):
        n_data = self.n_docs
        test_idx = sample(range(n_data), n_test)

        within_subreddit = []
        within_post = []
        random_doc = []

        for idx in test_idx:
            single_data = self.json_data[idx]
            name, subreddit, link_id = self.extract_subreddit_link_info(single_data)
            sent_to_compare = self.parse_string(single_data['body'])
            if len(sent_to_compare) < 2:
                continue
            doc1 = self.get_random_doc_within_subreddit(subreddit, name)
            doc2 = self.get_random_doc_within_post(link_id, name)
            doc3 = self.get_random_doc_outside_subreddit(subreddit)
            if len(doc1) < 2:
                continue
            if len(doc2) < 2:
                continue
            if len(doc3) < 2:
                continue
            within_subreddit.append(self.get_wmdistance(sent_to_compare, doc1))
            within_post.append(self.get_wmdistance(sent_to_compare, doc2))
            rdist = self.get_wmdistance(sent_to_compare, doc3)
            random_doc.append(rdist)

        return within_subreddit, within_post, random_doc


def plot_score_histogram(score, label, filename):
    plt.clf()
    hist_res = plt.hist(score)
    xmax = max(hist_res[1])
    ymax = max(hist_res[0])
    mean_val = round(np.mean(score), 2)
    std_val = round(np.std(score), 2)
    plt.text(xmax*0.9, ymax*0.9, 'Mean: '+ str(mean_val)+'\nStd: '+ str(std_val))
    plt.xlabel('WMD Score')
    plt.ylabel('Number of posts/comments')
    plt.title('Word Mover Distance '+ label)
    plt.savefig(filename)


def plot_three_scores_hist(score_list, label_list, filename):
    plt.clf()
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    for i, score in enumerate(score_list):
        ax = axes[i]
        xmax = max(hist_res[1])
        xmin = min(hist_res[1])
        ymax = max(hist_res[0])
        ymin = min(hist_res[0])
        xlen = xmax-xmin
        hist_res = ax.hist(score, bins=xlen, range=(xmin, xmax), align='mid', alpha=0.8)
        mean_val = round(np.mean(score), 2)
        std_val = round(np.std(score), 2)
        ax.text(xmax * 0.85, ymax * 0.85, 'Mean: ' + str(mean_val) + '\nStd: ' + str(std_val))
        ax.set_title('Word Mover Distance ' + label_list[i])
    plt.xlabel('WMD Score')
    axes[1].set_ylabel('Number of posts/comments')
    plt.savefig(filename)