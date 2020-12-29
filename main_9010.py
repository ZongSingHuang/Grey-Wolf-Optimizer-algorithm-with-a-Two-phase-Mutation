# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from BHPSOGWO import BHPSOGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import functools
import time

num_neig = 5
g = 100
p = 10
times = 20
w1 = 0.99
w2 = 0.01
skf = StratifiedKFold(n_splits=10, shuffle=True)
np.random.seed(42)

# name_list = ['BreastCancer', 'BreastEW', 'Congress', 'Exactly', 'Exactly2', 
#               'HeartEW', 'Ionosphere', 'KrVsKpEW', 'Lymphography', 'M-of-n',
#               'PenglungEW', 'Sonar', 'SpectEW', 'Tic-tac-toe', 'Vote', 
#               'WaveformEW', 'Wine', 'Zoo']
name_list = ['BreastCancer', 'BreastEW', 'Congress', 
              'HeartEW', 'Ionosphere', 'Lymphography',
              'PenglungEW', 'Sonar', 'SpectEW', 'Vote', 'Wine', 'Zoo']
X_all = []
y_all = []
table = np.zeros((7, len(name_list)))
table[3, :] = np.ones(len(name_list))*np.inf
table[4, :] = -np.ones(len(name_list))*np.inf
tmep_for_std = np.zeros((times, len(name_list)))
tmep_for_loss = np.zeros((g, len(name_list)))

# 讀資料
for idx, name in enumerate(name_list):
    data = pd.read_csv(name+'.csv', header=None).values
    X, y = data[:, :-1], data[:, -1]
    
    X_all.append(X)
    y_all.append(y)

def fitness(x, X, y):
    if x.ndim==1:
        x = x.reshape(1, -1)
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            score = cross_val_score(KNeighborsClassifier(n_neighbors=num_neig), X[:, x[i, :].astype(bool)], y, cv=skf)
            loss[i] = w1*(1-score.mean()) + w2*(np.sum(x[i, :])/X.shape[1])
        else:
            loss[i] = np.inf
    return loss

for k in range(times):
    for i in range(len(name_list)):
        start = time.time()
        
        loss_func = functools.partial(fitness, X=X_all[i], y=y_all[i])
        optimizer = BHPSOGWO(fit_func=loss_func, 
                             num_dim=X_all[i].shape[1], num_particle=p, max_iter=g)
        optimizer.opt()
        
        acc = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_all[i][:, optimizer.X_alpha.astype(bool)], y_all[i], cv=skf).mean()
        selected = np.sum(optimizer.X_alpha)
        loss = w1*(1-acc) + w2*selected/len(optimizer.X_alpha)
        
        table[0, i] += selected # for mean selected
        table[1, i] += acc # for mean acc
        table[2, i] += loss # for mean loss
        table[6, i] += time.time()-start # for mean time
        
        tmep_for_std[k, i] = loss
        tmep_for_loss[:, i] += optimizer.gBest_curve
        print(i)
    print('time:'+str(k))

table[0, :] = table[0, :] / times
table[1, :] = table[1, :] / times
table[2, :] = table[2, :] / times
table[3, :] = tmep_for_std.min()
table[4, :] = tmep_for_std.max()
table[5, :] = np.std(tmep_for_std, axis=0)
table[6, :] = table[6, :] / times
table = pd.DataFrame(table)
table = np.round(table, 2)
table.columns=name_list
table.index = ['mean feature selected', 'mean accuracy', 'mean fitness', 'best fitness', 'worst fitness', 'std fitness', 'time']

tmep_for_loss /= times
tmep_for_loss = pd.DataFrame(tmep_for_loss)
tmep_for_loss = np.round(tmep_for_loss, 2)
tmep_for_loss.columns=name_list