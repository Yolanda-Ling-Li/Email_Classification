import os
from chardet.universaldetector import UniversalDetector
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords


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


def read_dataSet():
    names = os.listdir(r'C:\Users\ericb\Desktop\maildir')
    X = []
    Y = []

    for idx, name in enumerate(names):
        read_name(idx, name, X, Y)

    return X, Y


def read_name(idx, name, X, Y):
    allList = os.walk(r'C:\Users\ericb\Desktop\maildir\\' + name)
    for root, dirs, files in allList:
        for file in files:
            file_name = root + '/' + file

            # transform encoding to utf-8
            convert_encode2utf8(file_name)

            f = open(file_name, 'rb')
            X.append(f.read())
            f.close()
            Y.append(idx)


# f = open('/Users/bibi/Documents/CSC522/project/maildir/presto-k/junk_e_mail/1.', 'rb')
# s = f.read()
# f.close()

# xTest = []
#
# al = os.walk(r'C:\Users\ericb\Desktop\maildir/presto-k/junk_e_mail/')
# for root, dirs, files in al:
#     for file in files:
#         file_name = root + '/' + file
#         f = open(file_name, 'r')
#         xTest.append(f.read())
#         f.close()


def get_idf(X):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_vectorizer.fit_transform(X)
    tfidf_tokens = tfidf_vectorizer.get_feature_names()

    idf = tfidf_vectorizer.idf_
    idf_list = zip(tfidf_tokens, idf)

    idf_list_sorted = sorted(idf_list, key=lambda x: x[1])
    return idf_list_sorted


def get_idf_low():
    idf_low = pd.read_csv('../idf_low.csv')
    idf_low_set = set(idf_low['0'])
    return idf_low_set


def remove_sp_symbol(text):
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', text)


# remove special symbol, low idf words and stop words
def extract_feature(X):
    x_feature = []
    stop_words = set(stopwords.words('english'))
    idf_low = get_idf_low()
    for text in X:
        textList = remove_sp_symbol(text).lower().split(" ")
        for idx, t in enumerate(textList):
            if t in idf_low or t in stop_words or t == 'x':
                textList[idx] = ''
        x_feature.append(' '.join(textList))
    return x_feature


qewx = extract_feature(xTest)
print(qewx[0])

# x, y = read_dataSet()
# idf = get_idf(x)
# print(np.asarray(idf[:20]))
# [['bcc' '1.0']
#  ['cc' '1.0']
#  ['charset' '1.0']
#  ['content' '1.0']
#  ['date' '1.0']
#  ['encoding' '1.0']
#  ['evans' '1.0']
#  ['filename' '1.0']
#  ['folder' '1.0']
#  ['id' '1.0']
#  ['javamail' '1.0']
#  ['message' '1.0']
#  ['mime' '1.0']
#  ['origin' '1.0']
#  ['plain' '1.0']
#  ['subject' '1.0']
#  ['text' '1.0']
#  ['thyme' '1.0']
#  ['transfer' '1.0']
#  ['type' '1.0']]

# print(np.asarray(idf[len(idf) - 21:]))
# [['zzp' '13.463428233647901']
#  ['zzsygxgx' '13.463428233647901']
#  ['zzsyjrxe' '13.463428233647901']
#  ['zztmky' '13.463428233647901']
#  ['zzv70asactaaqa5aeaamc' '13.463428233647901']
#  ['zzvoge' '13.463428233647901']
#  ['zzwyriqeuzgdapy59k830iuldikj7cgbt6v6dpd4720dkomzivepjdkb1rf6sdzeabikjctikmk0'
#   '13.463428233647901']
#  ['zzxsctwljcm9z' '13.463428233647901']
#  ['zzz1' '13.463428233647901']
#  ['zzz2' '13.463428233647901']
#  ['zzz30q0fqqc' '13.463428233647901']
#  ['zzzc00l1b0c' '13.463428233647901']
#  ['zzzehjtdapc' '13.463428233647901']
#  ['zzzipe' '13.463428233647901']
#  ['zzzl7wxn4pc' '13.463428233647901']
#  ['zzzoqjdwcpc' '13.463428233647901']
#  ['zzzp3wq0npc' '13.463428233647901']
#  ['zzzq61skaoc' '13.463428233647901']
#  ['zzztalk' '13.463428233647901']
#  ['zzzugorq00c' '13.463428233647901']
#  ['åkesson' '13.463428233647901']]

# idf_df = pd.DataFrame(data=np.asarray(idf))
# idf_df.to_csv('../idf.csv')
