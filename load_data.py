import json
import os


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
    print('Json file ' + file_name + ' loaded.')
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
    tmpstr = ""
    data = load_single_json(file_name)
    str_list = map(lambda x: x['body'], data)
    tmpstr = ' '.join(str_list)
    split_data = tmpstr.split(' ')
    return split_data



if __name__ == "__main__":

    fname = './reddit_data_MH/2015/RC_2015-01'
    # Let's just load one file
    # data = load_json(fname)
    # words = load_single_text(fname)
    words2015 = load_text('./reddit_data_MH/2015')



