import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime
from six import iteritems, itervalues, string_types
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
        ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim.models import Word2Vec
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc


class Word2VecData():
    """
    Abstract class for Word2vecData
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.w2v_model = None
        self.sentences = None
        self.n_docs, self.n_sents, self.n_words = 0, 0, 0

    def _update_stats(self):
        pass

    def _load_data(self, data_dir):
        pass

    def _load_word2vec(self, word2vec_file, binary):
        if not os.path.exists(word2vec_file):
            print('word2vec_file ' + word2vec_file + ' does not exist. Training first..')
            self._train_word2vec(word2vec_file, binary)
        else:
            if binary:
                print('Loading binary word2vec vectors')
            else:
                print('Loading text word2vec vectors')
            t1 = datetime.now()
            self.w2v_model = Word2Vec.load_word2vec_format(word2vec_file, binary=binary)
            print('Took %.2f seconds to load the word2vec model.' % (datetime.now() - t1).seconds)

    def _train_word2vec(self, word2vec_file, binary):
        pass


def save_output_vectors(model, fname='syn1.vectors', binary=False):
    """
    Save output (syn1neg) vectors, given the Word2Vec model

    Parameters
    ----------
    model
    fname
    binary

    Returns
    -------
    None

    """
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % model.syn1neg.shape))
        # store in sorted order: most frequent words at the top
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            row = model.syn1neg[vocab.index]
            if binary:
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


def load_sentences_in_text_file(file_name, sep=" . "):
    print('Loading file ' + file_name)
    with open(file_name, 'r') as f:
        tmpstr = f.read()

    tmpstr= tmpstr.lower()
    tmpstr = re.sub('\[deleted\]', ' ', tmpstr)
    tmpstr = re.sub('\\n', ' ', tmpstr)
    tmpstr = re.sub('\\\'', "'", tmpstr)
    tmpstr = re.sub('[^a-z0-9\-.\' ]', ' ', tmpstr)
    tmpstr = re.sub(' ---*', ' ', tmpstr)
    tmpstr = re.sub(r'\s+', ' ', tmpstr)
    tmpstr = re.sub(r'\.\.+', '. ', tmpstr)  # Remove multiple periods
    tmpstr = re.sub(" \.", " ", tmpstr)
    tmpstr = re.sub(" \'", " ", tmpstr)
    tmpstr = re.sub("\' ", " ", tmpstr)
    tmpstr = re.sub("\'$", "", tmpstr)
    tmpstr = re.sub(" -", " ", tmpstr)
    tmpstr = re.sub("- ", " ", tmpstr)
    tmpstr = re.sub("-$", "", tmpstr)

    tmpstr = tmpstr.strip()
    sent_list = tmpstr.split(sep)
    word_sent_list = map(lambda x: x.split(), sent_list)

    return word_sent_list


def load_sentences_in_large_text_file(file_name, sep=" . "):
    """
    Needs to be fixed.
    Parameters
    ----------
    file_name
    sep

    Returns
    -------

    """
    import csv
    print('Loading file ' + file_name)
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=sep)
        tmpstr = f.read()

    tmpstr= tmpstr.lower()
    tmpstr = re.sub('\[deleted\]', ' ', tmpstr)
    tmpstr = re.sub('\\n', ' ', tmpstr)
    tmpstr = re.sub('\\\'', "'", tmpstr)
    tmpstr = re.sub('[^a-z0-9\-.\' ]', ' ', tmpstr)
    tmpstr = re.sub(' ---*', ' ', tmpstr)
    tmpstr = re.sub(r'\s+', ' ', tmpstr)
    tmpstr = re.sub(r'\.\.+', '. ', tmpstr)  # Remove multiple periods
    tmpstr = re.sub(" \.", " ", tmpstr)
    tmpstr = re.sub(" \'", " ", tmpstr)
    tmpstr = re.sub("\' ", " ", tmpstr)
    tmpstr = re.sub("\'$", "", tmpstr)
    tmpstr = re.sub(" -", " ", tmpstr)
    tmpstr = re.sub("- ", " ", tmpstr)
    tmpstr = re.sub("-$", "", tmpstr)

    tmpstr = tmpstr.strip()
    sent_list = tmpstr.split(sep)
    word_sent_list = map(lambda x: x.split(), sent_list)

    return word_sent_list


def parse_string(doc, remove_stopwords=True):
    tmpstr = doc
    tmpstr = tmpstr.lower()
    tmpstr = re.sub('\[deleted\]', ' ', tmpstr)
    tmpstr = re.sub('\\n', ' ', tmpstr)
    tmpstr = re.sub('\\\'', "'", tmpstr)
    tmpstr = re.sub('[^a-z0-9\-\' ]', ' ', tmpstr)
    tmpstr = re.sub(' ---*', ' ', tmpstr)
    tmpstr = re.sub(r'\s+', ' ', tmpstr)
    tmpstr = re.sub(r'\.\.+', '. ', tmpstr)  # Remove multiple periods
    tmpstr = re.sub(r"\'\'*", "'", tmpstr)
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



def plot_score_histogram(score, label, filename):
    dirpath= os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
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

    dirpath= os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    plt.clf()
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    xmax = np.ceil(np.max(score_list))
    xmin = np.floor(np.min(score_list))
    xlen = xmax-xmin
    if xlen < 10:
        binlen = 20
    else:
        binlen = xlen
    for i, score in enumerate(score_list):
        ax = axes[i]

        hist_res = ax.hist(score, bins=binlen, range=(xmin, xmax), align='mid', alpha=0.8)
        # hist_res = ax.hist(score, bins=xlen, align='mid', alpha=0.8)
        ymax = max(hist_res[0])
        ymin = min(hist_res[0])
        mean_val = round(np.mean(score), 2)
        std_val = round(np.std(score), 2)
        ax.text(xmax * 0.85, ymax * 0.85, 'Mean: ' + str(mean_val) + '\nStd: ' + str(std_val))
        ax.set_title('Word Mover Distance ' + label_list[i])
    plt.xlabel('WMD Score')
    axes[1].set_ylabel('Number of posts/comments')
    plt.savefig(filename)



def plot_three_scores_plot1(score_list, label_list, filename):

    dirpath= os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    plt.clf()
    fig, axes = plt.subplots()
    xmax = len(score_list[0])
    xmin = 0
    xlen = xmax-xmin
    ymax = np.ceil(np.max(score_list))

    for i, score in enumerate(score_list):
        axes.plot(score, '.-')
        mean_val = round(np.mean(score), 2)
        std_val = round(np.std(score), 2)
        # axes.text(xmax * 0.85, ymax * 0.85, 'Mean: ' + str(mean_val) + '\nStd: ' + str(std_val))
        # axes.set_title('Word Mover Distance ' + label_list[i])
    plt.legend(label_list)
    plt.xlabel('Comment Index')
    axes.set_ylabel('WMD Score')
    plt.savefig(filename)




def plot_three_scores_plot(score_list, label_list, filename):

    dirpath= os.path.dirname(filename)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

    plt.clf()
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    xmax = len(score_list[0])
    xmin = 0
    xlen = xmax-xmin

    for i, score in enumerate(score_list):
        ymax = np.ceil(np.max(score))
        ax = axes[i]
        ax.plot(score, '.-')
        mean_val = round(np.mean(score), 2)
        std_val = round(np.std(score), 2)
        ax.text(xmax * 0.85, ymax * 0.85, 'Mean: ' + str(mean_val) + '\nStd: ' + str(std_val))
        ax.set_title('Word Mover Distance ' + label_list[i])
    plt.xlabel('Comment Index')
    axes[1].set_ylabel('WMD Score')
    plt.savefig(filename)


def most_similar(model, synnorm, positive=[], negative=[], topn=10, restrict_vocab=None):

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in positive
    ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in negative
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in model.vocab:
            mean.append(weight * synnorm[model.vocab[word].index])
            all_words.add(model.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    limited = synnorm if restrict_vocab is None else synnorm[:restrict_vocab]
    dists = dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(model.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def most_similar_two(model, synnorm1, synnorm2, positive=[], negative=[], topn=10, restrict_vocab=None):

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in positive
    ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
        for word in negative
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        elif word in model.vocab:
            mean.append(weight * synnorm1[model.vocab[word].index])
            all_words.add(model.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

    limited = synnorm2 if restrict_vocab is None else synnorm2[:restrict_vocab]
    dists = dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    result = [(model.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    return result[:topn]


def print_most_similar_words(word, model, synnorm, N=10):
    print("\n--- " + str(N) + " most similar words of '"+ word +"' ---" )
    res = most_similar(model, synnorm, positive=[word], topn=N)
    for item in res:
        print(item[0]+ ' : ' + str(item[1]))


def print_most_similar_words_two(word, model, synnorm1, synnorm2, N=10):
    print("\n--- " + str(N) + " most similar words of '"+ word +"' ---" )
    res = most_similar_two(model, synnorm1, synnorm2, positive=[word], topn=N)
    for item in res:
        print(item[0]+ ' : ' + str(item[1]))

