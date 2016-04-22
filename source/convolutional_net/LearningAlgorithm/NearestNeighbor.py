'''
Created on Apr 16, 2016

@author: tenma
'''
import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self,X,y):
        self.X = X
        self.y = y
        
    def predict(self,Xtest):
        num_test = Xtest.shape[0]
        Ypred = np.zeros(num_test,dtype = self.y.dtype)
        
        for i in xrange(num_test):
            distance = np.sum(np.abs(self.X-Xtest[i,:]),axis=1)
            #distance = np.sum(np.square(self.X-Xtest[i,:]),axis=1)
            index = np.argmin(distance)
            Ypred[i] = self.y[index]
            
        return Ypred
            