# coding: utf-8
import numpy as np
import pandas as pd
import gensim
from scipy import sparse
import copy


def get_data(sdata="./samples.txt", stw="./stopwords.txt", val_split=0.2):
    texts = []
    labels = []
    label_dict = {
        0: 'joy',
        1: 'hate',
        2: 'love',
        3: 'sorrow',
        4: 'anxiety',
        5: 'surprise',
        6: 'anger',
        7: 'expect',
    }

    def parsing():
        df_ = pd.read_csv(sdata, sep=',', error_bad_lines=True)
        for i, col in df_.iterrows():
            tweet = col['Tweet'].strip()
            lbs = col['label'].split('|')
            lb_ = []
            # for lb in lbs:
            for key in range(len(label_dict)):
                # lab = label_dict[key]
                # lab_onehot = int(col[lab])
                lb_.append([0, 1] if lbs[key] != 0 else [1, 0])
            labels.append(lb_)
            msg = list(tweet.split())
            texts.append(msg)

    parsing()

    dictionary = gensim.corpora.Dictionary(texts)
    bow_dictionary = copy.deepcopy(dictionary)

    stopWords = []
    for word in open(stw).readlines():
        stopWords.append(word.strip())
    stopWords = list(set(stopWords))
    # bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, stopWords)))
    len_1_words = list(filter(lambda w: len(w) == 1, bow_dictionary.values()))
    # bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))
    bow_dictionary.filter_extremes(no_below=2, no_above=0.7, keep_n=None)
    bow_dictionary.compactify()

    def get_(text_doc, seq_dictionary, bow_dictionary, ori_labels):
        seq_doc = []

        row = []
        col = []
        value = []
        row_id = 0
        m_labels = []

        for d_i, doc in enumerate(text_doc):
            if len(bow_dictionary.doc2bow(doc)) < 3:
                continue
            for i, j in bow_dictionary.doc2bow(doc):
                row.append(row_id)
                col.append(i)
                value.append(j)
            row_id += 1
            wids = list(map(seq_dictionary.token2id.get, doc))
            wids = np.array(list(filter(lambda x: x is not None, wids))) + 1
            m_labels.append(ori_labels[d_i])
            seq_doc.append(wids)

        lens = list(map(len, seq_doc))

        bow_doc = sparse.coo_matrix((value, (row, col)), shape=(row_id, len(bow_dictionary)))

        print("total %d sentences, avg len: %d, max len: %d" % (len(seq_doc), np.array(lens).mean(), np.max(lens)))
        return seq_doc, bow_doc, m_labels

    seq_title, bow_title, label_title = get_(texts, dictionary, bow_dictionary, labels)

    # shuf data
    indices = np.arange(len(seq_title))
    np.random.shuffle(indices)
    seq_title = np.array(seq_title)[indices]

    # splitting
    nb_test_samples = int(val_split * len(seq_title))
    seq_title_train = seq_title[:-nb_test_samples]
    seq_title_test = seq_title[-nb_test_samples:]

    bow_title = bow_title.tocsr()
    bow_title = bow_title[indices]
    bow_title_train = bow_title[:-nb_test_samples]
    bow_title_test = bow_title[-nb_test_samples:]

    label_title = np.array(label_title)[indices]
    label_title_train = label_title[:-nb_test_samples]
    label_title_test = label_title[-nb_test_samples:]

    return bow_title_train, bow_title_test, seq_title_train, seq_title_test, label_title_train, label_title_test, \
           dictionary, bow_dictionary, label_dict
