import numpy as np
import func
def pca(X):
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        M = np.dot(X,func.Transpose(X))
        e,EV = np.linalg.eigh(M)
        tmp = func.Transpose(np.dot(func.Transpose(X),EV))
        V = tmp[::-1]
        S = np.sqrt(e[::-1])

        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data] # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X