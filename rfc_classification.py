import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
import scipy as sp
import time

import os
os.chdir('G:/academics/ML2 Project/drug_discovery')

data=pd.read_csv('processed.csv')
data.iloc[:,-1]=data.iloc[:,-1].astype('category')

train, test=train_test_split(data,test_size=0.3)
train_f=train.iloc[:,1:2049]
test_f=test.iloc[:,1:2049]
train_y=train.iloc[:,-1]
test_y=test.iloc[:,-1]


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
for n in range(1,502,50):
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train_f.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_f.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
print(best_n, min_score)

print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []

for m in range(1,502,50):
    print("the max depth : {0}".format(m))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train_f.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train_f.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))
print(best_m, min_score)


plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(range(1,502,50), scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range(1,502,50), scores_m)
plt.ylabel('score')
plt.xlabel('max depth')


#model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
#model = RandomForestClassifier(n_estimators=best_n, max_depth=100)
#model = RandomForestClassifier(n_estimators=best_n, max_depth=500)
model = RandomForestClassifier(n_estimators=401, max_depth=401)
model.fit(train_f, train_y)
print(model.score(test_f,test_y))
plt.show()

#test accuracy 0.840092699884 n_estimators=100 max_depth=100
#test accuracy2 0.278485901893 n_estimators=100 max_depth=1
#test accuracy3 0.838933951333 n_estimators=401 max_depth=401
#test accuracy3 0.839706450367 n_estimators=351 max_depth=251