# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1016/j.advengsoft.2013.12.007
https://seyedalimirjalili.com/gwo
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class TMGWO():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, 
                 x_max=1, x_min=0, a_max=2, a_min=0):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.Mp = 0.5
        
        self._iter = 0
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)
        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = None
        self.X_beta = None
        self.X_delta = None

        self.X = np.random.choice(2, size=[self.num_particle, self.num_dim])
        
        self.update_score()
        
        self._itter = self._iter + 1

        
    def opt(self):
        while(self._iter<self.max_iter):
            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            
            for i in range(self.num_particle):
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_alpha - self.X[i, :])
                X1 = self.X_alpha - A*D
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_beta - self.X[i, :])
                X2 = self.X_beta - A*D
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_delta - self.X[i, :])
                X3 = self.X_delta - A*D
                
                self.X[i, :] = np.mean([X1, X2, X3])
                
                # conti 2 binary
                self.X[i, :] = np.random.uniform() >= 1/(1+np.exp(-1*self.X[i, :]))
            
            self.update_score()
            
            # mutation
            fitness = self.fit_func(self.X_alpha)
            Xmutated1 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):
                r = np.random.uniform()
                
                if r<self.Mp and self.X_alpha[i]==1:
                    Xmutated1[i] = 0
                    fitness_mutated = self.fit_func(Xmutated1)
                    if fitness_mutated<fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated1.copy()
                        
            fitness = self.fit_func(self.X_alpha)
            Xmutated2 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):
                r = np.random.uniform()
                
                if r<self.Mp and self.X_alpha[i]==0:
                    Xmutated2[i] = 1
                    fitness_mutated = self.fit_func(Xmutated2)
                    if fitness_mutated<fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated2.copy()
                        
            # update X_alpha
            self.score_alpha = self.fit_func(self.X_alpha)
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
    
    def update_score(self):
        score_all = self.fit_func(self.X)
        for idx, score in  enumerate(score_all):
            if score<self.score_alpha:
                self.score_alpha = score.copy()
                self.X_alpha = self.X[idx, :].copy()
                
            if score>self.score_alpha and score<self.score_beta:
                self.score_beta = score.copy()
                self.X_beta = self.X[idx, :].copy()
            
            if score>self.score_alpha and score>self.score_beta and score<self.score_delta:
                self.score_delta = score.copy()
                self.X_delta = self.X[idx, :].copy()
        
        self.gBest_X = self.X_alpha.copy()
        self.gBest_score = self.score_alpha.copy()
        self.gBest_curve[self._iter] = self.score_alpha.copy()