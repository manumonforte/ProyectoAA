import numpy as np
import scipy.optimize as opt
import operator

class OurLogisticRegression:

    def __init__(self,num_features, num_labels, reg=0.1):
        self.num_features= num_features
        self.num_labels = num_labels
        self.reg = reg
        self.thetas= self.create_thetas()
    
    def create_thetas(self):
        return np.zeros((self.num_labels,self.num_features +1))

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def cost(self, thetas, X, Y, Lambda):
        m = X.shape[0]
        sigmoid_X_theta = self.sigmoid(np.matmul(X, thetas))
        
        term_1_1 = np.matmul(np.transpose(np.log(sigmoid_X_theta)), Y)
        term_1_2 = np.matmul(np.transpose(np.log((1 - sigmoid_X_theta))),(1-Y))
        
        term_1 = - (term_1_1 + term_1_2) / np.shape(X)[0]
        term_2 = Lambda/(2*m) * sum(thetas **2)
    
        return term_1 + term_2

    def gradient(self,thetas, X, Y, Lambda):
        m = X.shape[0]
        
        sigmoid_X_theta = self.sigmoid(np.matmul(X,thetas))
        
        term_1 = np.matmul(np.transpose(X),(sigmoid_X_theta - Y)) /  np.shape(X)[0]
        term_2 = (Lambda/m) * thetas

        return term_1 + term_2

    def fit(self,X,Y):
        m = X.shape[0]
        X=np.hstack([np.ones([m,1]), X])
        for label in range(0, self.num_labels):
            filtered_labels = (Y == label) * 1
            thetas = np.zeros(np.shape(X)[1])
            self.thetas[label] = opt.fmin_tnc(func=self.cost, x0= thetas, fprime=self.gradient, args=(X, filtered_labels, self.reg))[0]

    def predict(self,X):
        m = X.shape[0]
        predictions = {}
        Y_pred = []
        X=np.hstack([np.ones([m,1]), X])
        for example in range(np.shape(X)[0]):
            for i in range(self.thetas.shape[0]):
                theta_opt = self.thetas[i]
                label = i
                prediction = round(self.sigmoid(np.matmul(np.transpose(theta_opt), X[example])), 4)
                predictions[label] = prediction
            
            Y_pred.append(max(predictions.items(), key=operator.itemgetter(1))[0])

        return Y_pred

    def score(self, Y, Y_pred):    
        correct = 0
        m = len(Y)
        for i in range(0,m):
            if(Y[i] == Y_pred[i]):
                correct += 1

        return (correct/m)*100 
