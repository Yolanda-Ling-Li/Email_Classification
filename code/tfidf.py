import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def get_idf(x):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidf_vectorizer.fit_transform(x)
    tfidf_tokens = tfidf_vectorizer.get_feature_names()

    idf = tfidf_vectorizer.idf_
    idf_list = zip(tfidf_tokens, idf)

    idf_list_sorted = sorted(idf_list, key=lambda a: a[1])
    return idf_list_sorted


def get_idf_low(x):
    data_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "data")
    idf = get_idf(x)
    idf_df = pd.DataFrame(data=np.asarray(idf))
    idf_df.to_csv(data_dir + '/idf.csv')
    idf_df.head(36).loc[:, '0': '1'].to_csv('../idf_low.csv')
    idf_low = pd.read_csv(data_dir + '/idf_low.csv')
    idf_low_set = set(idf_low['0'])
    return idf_low_set

# first 20 idf
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

# last 20 idf
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
