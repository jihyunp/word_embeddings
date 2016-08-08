import json
import os
import collections
import re
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime


def load_pretrained_word2vec(data_dir='/Users/jihyun/Documents/jihyun/research/word_embeddings/pre-trained/word2vec_CBOW/500K_subset'):
    import csv
    print('Loading the pre-trained word2vec data .. ')
    wvec_file = os.path.join(data_dir, 'word_wvecs.txt')
    reader = csv.reader(open(wvec_file, 'r'), delimiter=' ')

    dictionary = {}
    reverse_dictionary = []
    # wvecs = {}
    wvecs = np.zeros((500000, 300))
    for idx, line in enumerate(reader):
        word = line[0]
        dictionary[word] = idx
        reverse_dictionary.append(word)
        # wvecs[word] = np.array(line[1:], dtype=np.float)
        wvecs[idx,:] = np.array(line[1:], dtype=np.float)

    return (dictionary, reverse_dictionary, wvecs)




def load_json(data_dir='./reddit_data_MH'):
    data = []

    for dir, subdir, files in os.walk(data_dir):
        for fname in files:
            if fname.startswith('RC'):
                fpath = os.path.join(dir, fname)
                print(fpath)
                tmpdata = load_single_json(fpath)
                data.extend(tmpdata)
    return data




def load_single_json(file_name):
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

def load_sentences(data_dir='./reddit_data_MH'):
    """

    :param data_dir:
    :return: list[list[str]]
            List of sentences, which is a list of words.
    """
    list_of_sents= []

    for dir, subdir, files in os.walk(data_dir):
        for fname in files:
            if fname.startswith('RC'):
                fpath = os.path.join(dir, fname)
                list_of_sents += load_sentences_in_single_text(fpath)
    return list_of_sents


def load_sentences_in_single_text(file_name):
    """
    Made for test purpose
    :param file_name:
    :return:
    """
    print('Loading file ' + file_name)
    data = load_single_json(file_name)
    str_list = map(lambda x: x['body'], data)
    tmpstr = ' '.join(str_list)

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

    sent_list = tmpstr.split('. ')
    word_sent_list = map(lambda x: x.split(), sent_list)

    return word_sent_list



def load_words(data_dir='./reddit_data_MH'):
    """

    :param data_dir:
    :return: list[str]
            List of words
    """
    tmpdata = []

    for dir, subdir, files in os.walk(data_dir):
        for fname in files:
            if fname.startswith('RC'):
                fpath = os.path.join(dir, fname)
                print(fpath)
                tmpdata += load_words_in_single_text(fpath)
    return tmpdata


def load_words_in_single_text(file_name):
    """
    :param file_name:
    :return:
    """
    print('Loading file ' + file_name)
    data = load_single_json(file_name)
    str_list = map(lambda x: x['body'], data)
    tmpstr = ' '.join(str_list)

    tmpstr= tmpstr.lower()
    tmpstr = re.sub('\[deleted\]', ' ', tmpstr)
    tmpstr = re.sub('\\n', ' ', tmpstr)
    tmpstr = re.sub('[^a-z0-9\-\' ]', ' ', tmpstr)
    tmpstr = re.sub(' ---*', ' ', tmpstr)
    tmpstr = re.sub(r'\s+', ' ', tmpstr)

    split_data = tmpstr.split(' ')
    return split_data


def build_dataset(words, vocabulary_size=50000):
    print('Building dataset')
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
    data_index = 0
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels



def load_data_for_doc_class(data_dir='./reddit_data_MH', num_cats=None):

    json_data = load_json(data_dir)

    # Subreddit = target
    target_names= []
    target_dict = {}
    target = []
    data = []

    idx = 0
    for item in json_data:
        yname = item['subreddit']
        body = item['body']

        if body == '[deleted]':
            continue
        if len(body.split()) < 15:
            continue

        if yname not in target_names:
            # Add it to the dictionary
            target_dict[yname] = idx
            target_names.append(yname)
            idx += 1

        target.append(target_dict[yname])
        data.append(body)

    if num_cats is not None:
        tot_data_size = len(target)
        from collections import Counter
        Counter(target).most_common(num_cats)
        cats_to_include = map(lambda x: x[0], Counter(target).most_common(5))
        target_names = [target_names[i] for i in cats_to_include]
        oldid2newid = {y:x for x, y in enumerate(cats_to_include)}
        idx_newy = [(i, oldid2newid[target[i]]) for i in range(tot_data_size) if target[i] in cats_to_include]
        idx = map(lambda x: x[0], idx_newy)
        target = map(lambda x: x[1], idx_newy)
        data = [data[i] for i in idx]

    return data, np.array(target, dtype=np.int), target_names






def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)










if __name__ == "__main__":

    fname = './reddit_data_MH/2015/RC_2015-01'
    vocab_size = 50000

    # Step 1: Build the dictionary and replace rare words with UNK token.

    # Let's just load one file
    # data = load_json(fname)
    # words = load_words_in_single_text(fname)
    words = load_words('./reddit_data_MH/2015')

    print('Data Size: ' + str(len(words)) + ' tokens')

    # Step 2: Build the dictionary and replace rare words with UNK token.

    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=vocab_size)
    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


    # Step 3: Function to generate a training batch for the skip-gram model.

    # batch, labels = generate_batch(data, batch_size=8, num_skips=2, skip_window=1)
    # for i in range(8):
    #     print(batch[i], reverse_dictionary[batch[i]],
    #           '->', labels[i, 0], reverse_dictionary[labels[i, 0]])



    print('')
    print('Data loaded')
    print(datetime.now())
    print('')





