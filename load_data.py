import json
import os
import collections
import re
import numpy as np
import random

def load_json(data_dir='./reddit_data_MH'):
    data = []
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in os.listdir(subdir_path):
                fpath = os.path.join(subdir_path, fname)
                tmpdata= load_single_json(fpath)
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


def load_text(data_dir='./reddit_data_MH'):
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
                tmpdata += load_single_text(fpath)
    return tmpdata


def load_single_text(file_name):
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
    tmpstr = re.sub('[^a-z0-9\-\' ]', ' ', tmpstr)
    tmpstr = re.sub(' ---*', ' ', tmpstr)
    tmpstr = re.sub(r'\s+', ' ', tmpstr)

    split_data = tmpstr.split(' ')
    return split_data


def build_dataset(words, vocabulary_size=50000):
    print('Building dataset')
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
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


if __name__ == "__main__":

    fname = './reddit_data_MH/2015/RC_2015-01'

    # Step 1: Build the dictionary and replace rare words with UNK token.

    # Let's just load one file
    # data = load_json(fname)
    words = load_single_text(fname)
    # words2015 = load_text('./reddit_data_MH/2015')

    print('Data Size: ' + str(len(words)) + ' tokens')

    # Step 2: Build the dictionary and replace rare words with UNK token.

    data, count, dictionary, reverse_dictionary = build_dataset(words)
    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


    # Step 3: Function to generate a training batch for the skip-gram model.
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


    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])













