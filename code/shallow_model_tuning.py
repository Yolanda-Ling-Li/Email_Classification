import numpy as np
import pandas as pd
import re
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def range1():
    for n in 0.1, 1:
        yield n


def range2():
    for n in 3, 6, 9:
        yield n


def range3():
    for n in 50, 150, 250:
        yield n


cleanData = pd.read_csv("data.csv")
x_data = np.zeros((len(cleanData["x"]), 256))
y_data = np.zeros(len(cleanData["x"]))
for i in range(len(cleanData["x"])):
    one_x = re.split(' +', ' '.join(re.split(' +|\n+', cleanData["x"][i].replace("[", "").replace("]", ""))).strip())
    for j, x in enumerate(one_x):
        x_data[i][j] = float(x)
    y_data[i] = int(cleanData["y"][i])
ros = RandomOverSampler(random_state=0)
F_data, L_data = ros.fit_sample(x_data, y_data)
print(F_data.shape)
print(L_data.shape)


print('---------------------------')
print('hyperparameter tuning results for SVM')
print('max score, corresponding hyperparameter: C, gamma')
score_1 = []
i_1 = []
for i in range1():
    for j in range1():
        clf1 = SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(estimator=clf1, X=F_data, y=L_data, cv=5)
        scores = scores.mean()
        score_1.append(scores)
        i_1.append((i, j))
print(max(score_1), i_1[score_1.index(max(score_1))])

print('---------------------------')
print('hyperparameter tuning results for KNN')
print('max score, corresponding hyperparameter: n_neighbors')
score_2 = []
i_2 = []
for i in range4():
    clf2 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf2, F_data, L_data, cv=5)
    scores = scores.mean()
    score_2.append(scores)
    i_2.append(i)
print(max(score_2), i_2[score_2.index(max(score_2))])

print('---------------------------')
print('hyperparameter tuning results for RF')
print('max score, corresponding hyperparameter: n_estimators, max_depth')
score_3 = []
i_3 = []
for i in range3():
    clf3 = RandomForestClassifier(n_estimators=i, random_state=0)
    scores = cross_val_score(clf3, F_data, L_data, cv=5)
    scores = scores.mean()
    score_3.append(scores)
    i_3.append(i)
print(max(score_3), i3[score_3.index(max(score_3))])
