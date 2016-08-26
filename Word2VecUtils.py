import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime
from gensim.models import Word2Vec

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



def load_sentences_in_text_file(file_name):
    """
    Made for test purpose
    :param file_name:
    :return:
    """
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
    tmpstr = re.sub(" \'", " ", tmpstr)
    tmpstr = re.sub("\' ", " ", tmpstr)
    tmpstr = re.sub("\'$", "", tmpstr)
    tmpstr = re.sub(" -", " ", tmpstr)
    tmpstr = re.sub("- ", " ", tmpstr)
    tmpstr = re.sub("-$", "", tmpstr)

    sent_list = tmpstr.split(' . ')
    word_sent_list = map(lambda x: x.split(), sent_list)

    return word_sent_list


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

