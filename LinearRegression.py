import numpy as np
import scipy.optimize as opt
import operator

class OurLinearRegression:

    def __init__(self, num_features, reg=0.1):
        self.reg = reg
        self.num_features = num_features
        self.thetas = np.zeros(self.num_features)

    def cost(self, X, Y, reg=0):
        m = X.shape[0]
        H = np.dot(X, self.thetas)
        cost = (1/(2*m)) * np.sum((H-Y.T)**2) + ( reg / (2 * m) ) * np.sum(self.thetas[1:]**2)
        return cost

    def gradient(self, X, Y, reg=0):
        tt = np.copy(self.thetas)
        tt[0]=0
        m = X.shape[0]
        H = np.dot(X, self.thetas)
        gradient = ((1 / m) * np.dot(H-Y.T,X.T)) + ((reg/m) * tt)
        return gradient

    def linearCostGrad(self,X,Y,reg=0):
        return (self.cost(X,Y,reg),self.gradient(X,Y).flatten())

    def train(self, X, Y):
        initial_thetas = np.zeros(X.shape[1])
        return opt.minimize(fun=self.linearCostGrad,
                       x0=initial_thetas,
                       args=(X,Y),
                       method='TNC',
                       jac=True).x