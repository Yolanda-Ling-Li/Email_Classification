import re
import os
import sys
import importlib
importlib.reload(sys)
import multiprocessing
import numpy as np
import pandas as pd
from keras.models import Sequential
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense


class Config():
    max_len = 256
    vocabulary_dim = 32
    min_count = 5
    batch_size = 32
    keep_prob = 0.2
    num_epoch = 60
    class_size = 15
    unit_size = 150
    valid_split = 0
    window_size = 8
    n_iterations = 5
    cpu_count = multiprocessing.cpu_count()
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")
    log_dir_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data/logs")
    input_data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data/under_sample_data.csv")



def w2v_load_data(input_data_dir):
    print("Initialize...")
    cleanData = pd.read_csv(input_data_dir)
    x_data = []
    y_data = []

    for i in range(len(cleanData["x"])):
        x_data += [re.split(' ', ' '.join(re.split(' +|\n+', cleanData["x"][i])).strip())]
        y_data += [cleanData["y"][i]]

    # m = []
    # for i in range(len(x_data)):
    #     m += [len(x_data[i])]
    # m = np.array(m)
    # print("%.3f  %.3f  %.3f  %.3f" % (np.mean(m), np.std(m), np.median(m), np.percentile(m, 75)))

    X_train, X_test, y_train, y_test = train_test_split(x_data, np.array(y_data), train_size=0.8, random_state=42, stratify=np.array(y_data))
    return X_train, X_test, y_train, y_test


def create_dictionaries(model=None, data=None):
    if (data is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        return w2indx, w2vec
    else:
        print('No data provided...')


def word2vec_model(data, config):
    model = Word2Vec(size=config.vocabulary_dim, min_count=config.min_count, window=config.window_size,
                     workers=config.cpu_count, iter=config.n_iterations,
                     sorted_vocab=True)  # ,max_vocab_size=config.max_vocab_size
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(os.path.join(config.data_dir, "Word2vec_model.pkl"))
    index_dict, word_vectors = create_dictionaries(model=model, data=data)
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, config.vocabulary_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    x_index = np.zeros((len(data), config.max_len))
    for i, the_text in enumerate(data):
        for j, word in enumerate(the_text):
            if j < config.max_len and word in index_dict:
                x_index[i][j] = index_dict[word]

    return n_symbols, embedding_weights, x_index


def word2vec_shallow_model(data, config):
    model = Word2Vec(size=1, min_count=config.min_count, window=config.window_size,
                     workers=config.cpu_count, iter=config.n_iterations,
                     sorted_vocab=True)  # ,max_vocab_size=config.max_vocab_size
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(os.path.join(config.data_dir, "Word2vec_model.pkl"))
    index_dict, word_vectors = create_dictionaries(model=model, data=data)

    x_data = np.zeros((len(data), config.max_len))  # padding
    for i, the_text in enumerate(data):
        for j, word in enumerate(the_text):
            if j < config.max_len and word in word_vectors:
                x_data[i][j] = word_vectors[word]

    return x_data


def w2v_cnn_lstm_model(n_symbols,embedding_weights,config):
    # embedding layer

    model = Sequential()
    model.add(Embedding(output_dim=config.vocabulary_dim, input_dim=n_symbols,
                        weights=[embedding_weights], input_length=config.max_len,
                        name="Word2Vec"))
    model.add(LSTM(units=config.unit_size, dropout=config.keep_prob, recurrent_dropout=config.keep_prob, name="LSTM"))
    model.add(Dense(config.class_size, activation='softmax'))

    return model

