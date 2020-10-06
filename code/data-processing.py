import os
from chardet.universaldetector import UniversalDetector
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from tfidf import *


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


def read_name(idx, name, x, y, path):
    allList = os.walk(path + '\\' + name)
    for root, dirs, files in allList:
        for file in files:
            file_name = root + '/' + file

            # transform encoding to utf-8
            convert_encode2utf8(file_name)

            f = open(file_name, 'rb')
            x.append(f.read())
            f.close()
            y.append(idx)


def read_dataSet():
    # data set path
    path = r'C:\Users\ericb\Desktop\maildir'
    names = os.listdir(path)
    x = []
    y = []

    for idx, name in enumerate(names):
        read_name(idx, name, x, y, path)

    return x, y, names


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


def export_featured_data():
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")

    x, y, y_class = read_dataSet()
    x = extract_feature(x)
    data = list(zip(x, y))
    df = pd.DataFrame(data=np.asarray(data))
    df.to_csv(data_dir + '/data.csv')
    return y_class


def main():
    y_class = export_featured_data()
    print(y_class)
    # train_w2v()


if __name__ == "__main__":
    main()
