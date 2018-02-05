# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:50:48 2017

@author: shrey
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

y1=y[:49]

# find the mean vector
mean_vector=[]
grp=np.split(X,3)
for arr in grp:
    mean=arr.mean(axis=0)
    mean_vector.append(mean)

scatter_w=np.zeros((X.shape[1],X.shape[1]))
for g,m in zip(grp,mean_vector):
    sw=np.zeros((X.shape[1],X.shape[1]))
    for row in g:
        row,m=row.reshape((X.shape[1],1)),m.reshape((X.shape[1],1))
        sw+=np.dot((row-m),(row-m).T)
    scatter_w+=sw  
    
mean_x=np.mean(X,axis=0)  
mean_x=mean_x.reshape((X.shape[1],1)) 

scatter_b= np.zeros((X.shape[1],X.shape[1]))
for m in mean_vector:
    m=m.reshape((X.shape[1],1))
    scatter_b+=len(X)/3*np.dot((m-mean_x),(m-mean_x).T)
    
e_val,e_vec=np.linalg.eig(np.dot((np.linalg.inv(scatter_w)),scatter_b))  
e_val=np.real(e_val)
e_vec=np.real(e_vec)
aa =e_val.argsort()[::-1] 
e_val=e_val[aa]
e_vec=e_vec[:,aa] 
    
W=e_vec[:,:2]

y=np.dot(W.T,X.T)
y=y.T
plt.scatter(y[:,0],y[:,1])
        
    
    
    