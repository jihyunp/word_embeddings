from gensim import utils, matutils
from six import iteritems, itervalues, string_types

# Assume that you've trained a word2vec model as 'model'.


def save_syn1neg_vectors(model, fname='syn1.vectors', binary=False):
    # Save output vectors

    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % model.syn1neg.shape))
        # store in sorted order: most frequent words at the top
        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):
            row = model.syn1neg[vocab.index]
            if binary:
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))

