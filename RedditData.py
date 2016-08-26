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
from Word2VecUtils import Word2VecData, parse_string


class RedditData(Word2VecData):

    def __init__(self, data_dir='./reddit_data_MH',
                 word2vec_file='./word2vec_reddit_2008_2015_20/gensim_word2vec_2008_2015_orig.vectors', binary=False):
        Word2VecData.__init__(self, data_dir)

        self.n_docs, self.n_subreddits, self.n_posts = None, None, None
        self.json_data = None
        self.subreddit2did, self.link2did = None, None
        self.name2did, self.did2name = None, None

        t1 = datetime.now()
        self._load_reddit_data(data_dir)
        self._get_subreddit_info()
        self._update_stats()
        self._load_word2vec(word2vec_file, binary)
        print('Took %.2f seconds to load.' % (datetime.now() - t1).seconds)

    def _update_stats(self):
        self.n_docs = len(self.json_data)
        self.n_subreddits = len(self.subreddit2did)
        self.n_posts = len(self.link2did)

    def _get_subreddit_info(self):
        self.sid2subreddit = self.subreddit2did.keys()
        self.subreddit2sid = {y:x for x, y in enumerate(self.sid2subreddit)}

    def _load_word2vec(self, word2vec_file, binary):
        if not os.path.exists(word2vec_file):
            print('word2vec_file ' + word2vec_file + ' does not exist. Training first..')
            self._train_word2vec(word2vec_file, binary)
        else:
            if binary:
                print('Loading binary word2vec vectors')
            else:
                print('Loading text word2vec vectors')
            self.w2v_model = Word2Vec.load_word2vec_format(word2vec_file, binary=binary)

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

    def _train_word2vec(self, word2vec_file, binary):
        print('_train_word2vec(): Currently this function does nothing. Training should be done separately.')
        ## word_embeddings/Word2Vec/train_gensim_word2vec_reddit.py
        pass

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
            return parse_string(doc)

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
            return parse_string(doc)

    def get_random_doc_outside_subreddit(self, subreddit):
        subred_id = self.subreddit2sid[subreddit]
        other_sids = range(self.n_subreddits)
        other_sids.remove(subred_id)
        sid = sample(other_sids, 1)[0]
        return self.get_random_doc_within_subreddit(self.sid2subreddit[sid], None)


    def get_wmdistance(self, doc1, doc2):
        return self.w2v_model.wmdistance(doc1, doc2)

    def get_three_scores(self, n_test=1000):
        """
        Randomly select a comment, and then randomly select another three comments that are
        within the same subreddit, within the same post, and a random post outside the subreddit
        that the selected comment is in.
        After that, calculate three WMD (Word mover distances) from the selected comment to the
        three comments. Return the scores with comments.

        :param n_test:
        :return:  Returns 4 lists
            1) list of scores within subreddit
            2) list of scores within thread/post
            3) list of scores with random doc
            4) list of [comment, comment used in 1), comment used in 2), comment used in 3)]
        """
        n_data = self.n_docs
        test_idx = sample(range(n_data), n_test)

        within_subreddit = []
        within_post = []
        random_doc = []
        comments = []

        for idx in test_idx:
            single_data = self.json_data[idx]
            name, subreddit, link_id = self.extract_subreddit_link_info(single_data)
            sent_to_compare = parse_string(single_data['body'])
            if len(sent_to_compare) < 3:
                continue
            doc1 = self.get_random_doc_within_subreddit(subreddit, name)
            doc2 = self.get_random_doc_within_post(link_id, name)
            doc3 = self.get_random_doc_outside_subreddit(subreddit)
            if len(doc1) < 3:
                continue
            if len(doc2) < 3:
                continue
            if len(doc3) < 3:
                continue
            within_subreddit.append(self.get_wmdistance(sent_to_compare, doc1))
            within_post.append(self.get_wmdistance(sent_to_compare, doc2))
            rdist = self.get_wmdistance(sent_to_compare, doc3)
            random_doc.append(rdist)
            comments.append([sent_to_compare, doc1, doc2, doc3])

        return within_subreddit, within_post, random_doc, comments

    def get_most_and_least_similar_comments(self, n_test=100):
        """

        Parameters
        ----------
        n_test

        Returns
        -------
        list[tup[list[int, float, str]]]
        list of tuples.
        each tuple is ([random_sent_idx, 0, comment_to_compare],
                       [most_similar_idx, most_similar_score, most_similar_comment],
                       [least_similar_idx, least_similar_score, least_similar_coment])
            within the post for a randomly selected comment.

        """
        n_data = self.n_docs
        test_idx = sample(range(n_data), n_test)
        result = []

        for idx in test_idx:
            single_data = self.json_data[idx]
            name, subreddit, link_id = self.extract_subreddit_link_info(single_data)
            sent_to_compare = parse_string(single_data['body'])
            if len(sent_to_compare) < 5:
                continue
            idxs_in_same_post = self.link2did[link_id]
            if len(idxs_in_same_post) < 3:
                continue
            else:
                idxs_copy= list(idxs_in_same_post)
                did_to_exclude = self.name2did[name]
                del (idxs_copy[idxs_copy.index(did_to_exclude)])
                tmp_scores = {}
                tmp_docs = {}
                for idx2 in idxs_copy:
                    doc = parse_string(self.json_data[idx2]['body'])
                    score = self.get_wmdistance(sent_to_compare, doc)
                    tmp_scores[score] = idx2
                    tmp_docs[idx2] = doc
                sorted_score = np.sort(tmp_scores.keys())
                most_similar_score = sorted_score[0]
                most_similar_idx = tmp_scores[most_similar_score]
                most_similar_comment = tmp_docs[most_similar_idx]
                least_similar_score = sorted_score[-1]
                least_similar_idx = tmp_scores[least_similar_score]
                least_similar_comment = tmp_docs[least_similar_idx]
                result.append(([idx, 0, ' '.join(sent_to_compare)],
                               [most_similar_idx, most_similar_score, ' '.join(most_similar_comment)],
                               [least_similar_idx, least_similar_score, ' '.join(least_similar_comment)]))

        return result

