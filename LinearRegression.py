import numpy as np
import scipy.optimize as opt
import operator

class OurLinearRegression:

    def __init__(self, num_features, reg=0.1):
        self.reg = reg
        self.num_features = num_features
        self.thetas = np.zeros(self.num_features)

    def cost(self, thetas, X, Y, reg=0):
        m = X.shape[0]
        H = np.dot(X, thetas)
        cost = (1/(2*m)) * np.sum((H-Y.T)**2) + ( reg / (2 * m) ) * np.sum(thetas[1:]**2)
        return cost
    
    def gradient(self, thetas, X, Y, reg=0):
        tt = np.copy(thetas)
        tt[0]=0
        m = X.shape[0]
        H = np.dot(X, thetas)
        gradient = ((1 / m) * np.dot(H-Y.T,X)) + ((reg/m) * tt)
        return gradient

    def linearCostGrad(self, thetas, X, Y, reg=0):
        return (self.cost(thetas,X,Y,reg),self.gradient(thetas,X,Y).flatten())

    def train(self, X, Y):
        initial_thetas = np.zeros(self.num_features)
        self.thetas = opt.minimize(fun=self.linearCostGrad,
                       x0=initial_thetas,
                       args=(X,Y,self.reg),
                       method='TNC',
                       jac=True).x

    def predict(self, X):
        return X @ self.thetas