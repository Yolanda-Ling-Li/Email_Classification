import os
from chardet.universaldetector import UniversalDetector
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tfidf import *
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import collections


def get_encode_info(file):
    with open(file, 'rb') as f:
        detector = UniversalDetector()
        for line in f.readlines():
            detector.feed(line)
            if detector.done:
                break
        detector.close()
        return detector.result['encoding']


def convert_encode2utf8(file):
    f = open(file, 'rb')
    file_content = f.read()
    encode_info = get_encode_info(file)
    if encode_info == 'utf-8':
        return
    file_decode = file_content.decode(encode_info, 'ignore')
    file_encode = file_decode.encode('utf-8')
    f.close()
    f = open(file, 'wb')
    f.write(file_encode)
    f.close()


# read email for specific sender
def read_name(idx, name, x, y, path):
    allList = os.walk(path + '\\' + name)
    for root, dirs, files in allList:
        for file in files:
            file_name = root + '/' + file

            # transform encoding to utf-8
            convert_encode2utf8(file_name)

            f = open(file_name, 'r')
            x.append(f.read())
            f.close()
            y.append(idx)


# read all data
def read_dataSet():
    # data set path
    path = r'..\maildir'
    names = os.listdir(path)
    x = []
    y = []

    for idx, name in enumerate(names):
        read_name(idx, name, x, y, path)

    return x, y, names


# remove special symbol
def remove_sp_symbol(text):
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', text)


# remove special symbol, low idf words and stop words
def extract_feature(x):
    x_featured = []
    stop_words = set(stopwords.words('english'))
    idf_low = get_idf_low(x)
    for text in x:
        textList = remove_sp_symbol(text).lower().split(" ")
        for idx, t in enumerate(textList):
            if t in idf_low or t in stop_words or t == 'x':
                textList[idx] = ''
        x_featured.append(' '.join(textList))
    return x_featured


# export featured data, return all class names
def export_featured_data():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")

    x, y, y_class = read_dataSet()
    x = extract_feature(x)
    with open(data_dir + '/data.txt', 'w') as f:
        f.write("\n*\n".join(x))
    data = list(zip(x, y))
    df = pd.DataFrame(data=data, columns=['x', 'y'])
    df.to_csv(data_dir + '/data.csv')
    d = pd.DataFrame(data=y_class, columns=['y_class'])
    d.to_csv(data_dir + '/y_class.csv')
    return y_class


# split classes with different number of samples
def count_classes():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")
    data = pd.read_csv(data_dir + '/data.csv')
    classes = data['y'].value_counts()
    samples = classes.to_numpy()[::-1]
    cur = 0
    plot_y = []
    plot_x = []
    for i in range(10):
        count = 0
        bound = (i + 1) * 3000
        while cur < len(samples) and samples[cur] <= bound:
            count += 1
            cur += 1
        plot_y.append(count)
        plot_x.append(str(int(bound / 1000) - 3) + 'k-' + str(int(bound / 1000)) + 'k')
    plt.barh(plot_x, plot_y)
    plt.ylabel('Number of samples')
    plt.xlabel('Number of classes')
    plt.show()

    # 103 classes <= 3000 samples
    # 47 classes > 3000 samples
    class_list = classes.axes[0]
    high = class_list[:47]
    mid = class_list[47:98]
    low = class_list[98:]
    return high, mid, low


# random sample data, 5 classes for each interval
def sampling():
    high, mid, low = count_classes()
    sample_number = 5
    h = np.random.choice(high, sample_number, replace=False)
    m = np.random.choice(mid, sample_number, replace=False)
    l = np.random.choice(low, sample_number, replace=False)
    return h, m, l

def rename_class(data, y_class):
    y_list = list(y_class.keys())
    for idx, row in data.iterrows():
        data.loc[idx, 'y'] = y_list.index(row['y'])
    y_renamed = {}
    for idx, y in enumerate(y_list):
        y_renamed[idx] = y_class[y]
    return data, y_renamed

# get sample data
def get_sample_data():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")
    h, m, l = sampling()
    y_class = pd.read_csv(data_dir + '/y_class.csv')['y_class'].to_numpy()
    data = pd.read_csv(data_dir + '/data.csv')
    high = data.loc[data['y'].isin(h)]
    mid = data.loc[data['y'].isin(m)]
    low = data.loc[data['y'].isin(l)]
    sample_data = pd.concat([high, mid, low])
    y = {i: y_class[i] for i in np.concatenate((h, m, l))}
    x_renamed, y_renamed = rename_class(sample_data, y)
    return x_renamed, y_renamed


# data under sampling
def under_sampled_data():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")
    data, y_class = get_sample_data()
    x = data.loc[:, :'y']
    y = data['y']
    rus = RandomUnderSampler(random_state=30)
    data_rus, y_rus = rus.fit_resample(x, y)
    df = pd.DataFrame(data=data_rus, columns=['x', 'y'])
    df.to_csv(data_dir + '/under_sample_data.csv')
    # counter = collections.Counter(y_rus)
    d = pd.DataFrame.from_dict(y_class, orient='index', columns=['y_class'])
    d.to_csv(data_dir + '/under_sample_y_class.csv')
    return data_rus, y_rus, y_class


def main():
    # y_class = export_featured_data()
    # train_w2v()
    np.random.seed(30)
    data_rus, y_rus, y_class = under_sampled_data()


if __name__ == "__main__":
    main()
