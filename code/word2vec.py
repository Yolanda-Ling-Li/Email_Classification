import re
import os
import multiprocessing
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


class Config():
    max_len = 512  # 句长
    vocabulary_dim = 64    #词向量维数
    min_count = 5  # 过滤频数小于5的词语
    max_vocab_size = 100000 #如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。
    window_size = 8  # 窗口大小
    n_iterations = 5 # 迭代次数，默认为5 #定义词向量模型
    cpu_count = multiprocessing.cpu_count()
    model_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),"data")
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))),"data/Q3.csv")



def w2v_load_data(data_dir, model_dir):
    print("Initialize...")
    cleanData = pd.read_csv(data_dir)
    x_data = []
    y_data = []

    for i in range(len(cleanData["x"])):
        x_data += [' '.join(re.split(' +|\n+', cleanData["x"][i])).strip()]
        y_data += [cleanData["y"][i]]

    np.save(os.path.join(model_dir,'y_data.npy'), np.array(y_data))  # y_data存为npy
    return x_data, y_data


#创建词语字典，并返回每个词语的索引，词向量，以及每个文本所对应的词语索引
def create_dictionaries(vocab_dim,max_len,model=None, data=None):

    if (data is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True) #把该查询文档（词集）更改为（词袋模型）即：字典格式，key是单词，value是该单词在该文档中出现次数。
        w2indx = {v: k+1 for k, v in gensim_dict.items()} #所有的词语的索引
        w2vec = {word: model.wv[word] for word in w2indx.keys()} #所有词语的词向量

        def parse_dataset(thedata):    #Words become index
            new_index=[]

            for sentence in thedata:
                new_txt_index = []
                for word in sentence:
                    if word in w2indx:
                        new_txt_index.append(w2indx[word])
                    else:
                        new_txt_index.append(0)
                new_index.append(new_txt_index)
            return new_index

        x_index = parse_dataset(data) #每个句子所含词语对应的词语索引
        x_index = sequence.pad_sequences(x_index, maxlen=max_len)  # maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
        return w2indx, w2vec,x_index
    else:
        print('No data provided...')


#创建词语字典，并返回词典大小，权重矩阵以及每个词语索引矩阵
def word2vec_model(data, config):
    model = Word2Vec(size=config.vocabulary_dim,min_count=config.min_count,window=config.window_size,workers=config.cpu_count,iter=config.n_iterations,sorted_vocab=True) #,max_vocab_size=config.max_vocab_size
    model.build_vocab(data) #建立词典，必须步骤，不然会报错
    model.train(data,total_examples=model.corpus_count,epochs=model.epochs) #训练词向量模型
    model.save(os.path.join(config.model_dir,  "Word2vec_model.pkl")) #保存词向量模型
    index_dict, word_vectors, x_index = create_dictionaries(vocab_dim=config.vocabulary_dim,max_len=config.max_len,model=model,data=data)
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, config.vocabulary_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]

    x_data = np.zeros((n_symbols, config.max_len)) #转化为词向量代替词语的向量矩阵
    for i, the_text in enumerate(data):
        for j, word in enumerate(the_text):
            if word in word_vectors:
                x_data[i][j] = np.mean(word_vectors[word])

    return n_symbols, embedding_weights, x_index, x_data


def train_w2v():
    config = Config()
    x_data, y_data = w2v_load_data(config.data_dir, config.model_dir)

    print('Training a Word2vec model...')
    n_symbols, embedding_weights, x_index, x_data = word2vec_model(data=x_data,config=config)

    return n_symbols, embedding_weights, x_index





def main():
    train_w2v()


#当.py文件被直接运行时将被运行，当.py文件以模块形式被导入时不被运行。
if __name__ == "__main__":
    main()
