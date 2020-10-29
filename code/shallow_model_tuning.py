import numpy as np
import pandas as pd
import re
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


Data = pd.read_csv("new_data.csv")

x_data = np.zeros((len(Data["x"]), 256))
y_data = np.zeros(len(Data["x"]))
for i in range(len(Data["x"])):
    one_x = re.split(' +', ' '.join(re.split(' +|\n+', Data["x"][i].replace("[", "").replace("]", ""))).strip())
    for j, x in enumerate(one_x):
        x_data[i][j] = float(x)
    y_data[i] = int(Data["y"][i])

print(x_data.shape)
print(y_data.shape)
print(Counter(y_data))



print('---------------------------')
print('hyperparameter tuning results for SVM')
print('max score, corresponding hyperparameter: C, gamma')
score_1 = []
i_1 = []
for i in [0.01, 0.1, 1, 10]:
    for j in [0.01, 0.1, 1, 10]:
        clf1 = SVC(C=i, gamma=j, probability=True)
        scores = cross_val_score(estimator=clf1, X=x_data, y=y_data, cv=5)
        scores = scores.mean()
        score_1.append(scores)
        i_1.append((i, j))
print(max(score_1), i_1[score_1.index(max(score_1))])


print('---------------------------')
print('hyperparameter tuning results for KNN')
print('max score, corresponding hyperparameter: n_neighbors')
score_2 = []
i_2 = []
for i in [3, 6, 9, 12]:
    clf2 = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf2, x_data, y_data, cv=5)
    scores = scores.mean()
    score_2.append(scores)
    i_2.append(i)
print(max(score_2), i_2[score_2.index(max(score_2))])


print('---------------------------')
print('hyperparameter tuning results for RF')
print('max score, corresponding hyperparameter: n_estimators, max_depth')
score_3 = []
i3 = []
for i in [10, 50, 100, 250, 500]:
    for j in [10, 50, 100]:
        for k in [10, 20, 30, 40, 50]:
            clf3 = RandomForestClassifier(n_estimators=i, max_depth=j, max_leaf_nodes=k, random_state=0)
            scores = cross_val_score(clf3, x_data, y_data, cv=5)
            scores = scores.mean()
            score_3.append(scores)
            i3.append((i, j, k))
print(max(score_3), i3[score_3.index(max(score_3))])
