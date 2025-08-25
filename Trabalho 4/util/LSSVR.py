"""Least Squares Support Vector Regression."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process import kernels
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from scipy.sparse import linalg

__all__ = ['LSSVR', 'RegENN_LSSVR', 'RegCNN_LSSVR', 'DiscENN_LSSVR', 'MI_LSSVR', 'AM_LSSVR', 'NL_LSSVR']

class LSSVR(BaseEstimator, RegressorMixin):
    def __init__(self, C=100, kernel='rbf', gamma=0.1, ):
        self.supportVectors      = None
        self.supportVectorLabels = None
        self.C = C
        self.gamma = gamma
        self.kernel= kernel
        self.idxs  = None
        self.K = None
        self.bias = None 
        self.alphas = None

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, x_train, y_train):
        # self.idxs can be used to select points as support vectors,
        # so you need another algorithm or criteria to choose them
        if type(self.idxs) == type(None):
            self.idxs=np.ones(x_train.shape[0], dtype=bool)

        self.supportVectors      = x_train[self.idxs, :]
        self.supportVectorLabels = y_train[self.idxs]

        K = self.kernel_func(self.kernel, x_train, self.supportVectors, self.gamma)

        self.K = K
        OMEGA = K
        OMEGA[self.idxs, np.arange(OMEGA.shape[1])] += 1/self.C

        D = np.zeros(np.array(OMEGA.shape) + 1)

        D[1:,1:] = OMEGA
        D[0, 1:] += 1
        D[1:,0 ] += 1

        n = len(self.supportVectorLabels) + 1
        t = np.zeros(n)
        
        t[1:n] = self.supportVectorLabels
    
        # sometimes this function breaks
        try:
            z = linalg.lsmr(D.T, t)[0]
        except:
            z = np.linalg.pinv(D).T @ t.ravel()

        self.bias   = z[0]
        self.alphas = z[1:]
        self.alphas = self.alphas[self.idxs]

        return self

    def predict(self, x_test):
        K = self.kernel_func(self.kernel, x_test, self.supportVectors, self.gamma)

        return (K @ self.alphas) + self.bias
        # return np.sum(K * (np.tile(self.alphas, (K.shape[0], 1))), axis=1) + self.bias

    def kernel_func(self, kernel, u, v, gamma):
        if kernel == 'linear':
            k = np.dot(u, v.T)
        if kernel == 'rbf':
            k = rbf_kernel(u, v, gamma=gamma)
            # temp = kernels.RBF(length_scale=(1/gamma))
            # k = temp(u, v)
        if kernel == 'matern':
            kr = kernels.Matern(nu=self.gamma)
            k = kr(u,v)
            # temp = kernels.RBF(length_scale=(1/gamma))
            # k = temp(u, v)
        
        return k

    def score(self, X, y, sample_weight=None):
        from scipy.stats import pearsonr
        p, _ = pearsonr(y, self.predict(X))
        return p ** 2
        #return RegressorMixin.score(self, X, y, sample_weight)

    def norm_weights(self):
        n = len(self.supportVectors)

        A = self.alphas.reshape(-1,1) @ self.alphas.reshape(-1,1).T
        # import pdb; pdb.set_trace()
        W = A @ self.K[self.idxs,:]
        return np.sqrt(np.sum(np.diag(W)))





if __name__ == '__main__':
    import numpy as np
    
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from RBFNN import RBFNN

    boston = load_boston()

    X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)

    model = LSSVR(kernel='rbf', C=1000, gamma=0.001,)    
    #model=RBFNN(n_hidden=80, epsilon=1e-5, random_state=0)     
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    print('LSSVR\nMSE', mean_squared_error(y_test, y_hat))
    print('R2 ',model.score(X_test, y_test))
    
    
