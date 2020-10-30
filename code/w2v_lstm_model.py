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
    max_len = 256  # 句长
    vocabulary_dim = 32  # 词向量维数
    min_count = 5  # 过滤频数小于5的词语
    batch_size = 32
    keep_prob = 0.2  # 防过拟合
    num_epoch = 100  # 100
    class_size = 15
    unit_size = 150
    window_size = 8  # 窗口大小
    n_iterations = 5  # 迭代次数，默认为5 #定义词向量模型
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


# 创建词语字典，并返回每个词语的索引，词向量，以及每个文本所对应的词语索引
def create_dictionaries(model=None, data=None):
    if (data is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)  # 把该查询文档（词集）更改为（词袋模型）即：字典格式，key是单词，value是该单词在该文档中出现次数。
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有的词语的索引
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有词语的词向量

        return w2indx, w2vec
    else:
        print('No data provided...')


# 创建词语字典，并返回词典大小，权重矩阵以及每个词语索引矩阵
def word2vec_model(data, config):
    model = Word2Vec(size=config.vocabulary_dim, min_count=config.min_count, window=config.window_size,
                     workers=config.cpu_count, iter=config.n_iterations,
                     sorted_vocab=True)  # ,max_vocab_size=config.max_vocab_size
    model.build_vocab(data)  # 建立词典，必须步骤，不然会报错
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)  # 训练词向量模型
    model.save(os.path.join(config.data_dir, "Word2vec_model.pkl"))  # 保存词向量模型
    index_dict, word_vectors = create_dictionaries(model=model, data=data)
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, config.vocabulary_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    x_index = np.zeros((len(data), config.max_len)) # 转化为词向量代替词语的向量矩阵
    for i, the_text in enumerate(data):
        for j, word in enumerate(the_text):
            if j < config.max_len and word in index_dict:
                x_index[i][j] = index_dict[word]

    return n_symbols, embedding_weights, x_index


def word2vec_shallow_model(data, config):
    model = Word2Vec(size=1, min_count=config.min_count, window=config.window_size,
                     workers=config.cpu_count, iter=config.n_iterations,
                     sorted_vocab=True)  # ,max_vocab_size=config.max_vocab_size
    model.build_vocab(data)  # 建立词典，必须步骤，不然会报错
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)  # 训练词向量模型
    model.save(os.path.join(config.data_dir, "Word2vec_model.pkl"))  # 保存词向量模型
    index_dict, word_vectors = create_dictionaries(model=model, data=data)

    x_data = np.zeros((len(data), config.max_len))  # 转化为词向量代替词语的向量矩阵
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
                        name="Word2Vec"))  # Adding Input Length Embedding层只能作为模型的第一层
    model.add(LSTM(units=config.unit_size, dropout=config.keep_prob, recurrent_dropout=config.keep_prob, name="LSTM"))
    model.add(Dense(config.class_size, activation='softmax'))

    return model

