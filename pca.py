import numpy as np
import matplotlib.pyplot as plt

def pca(data,std=True):# method for PCA
    x_array=np.array(data)

    data_x=x_array.astype(float)

    if std==True:  # normalize the data
        
            data_mean=data_x.mean(axis=0)
    
            data_mc=data_x-data_mean
            pca_data=data_mc
    else:
        pca_data=data_x
    
    data_cov=np.cov(pca_data.T)   # find the covariance matrix
        
    e_val,e_vec=np.linalg.eig(data_cov)  # find eigenvalues and eigenvectors of the covariance matrix
 
    aa =e_val.argsort()[::-1]   # find index in the increasing order of eigenvalues
    e_val=e_val[aa]
    e_vec=e_vec[:,aa]
    scores = np.matmul(pca_data, e_vec[:,[0,1]]) # transform the original data to the new vector space

    var_sum=e_val.sum()
    
    e_var=e_val/var_sum
    plt.plot(e_var)   # plot the variance shown by different principle components
    return e_val,e_vec,scores,e_var,pca_data
