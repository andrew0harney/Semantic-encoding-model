import numpy as np
from GridRegression import model
from sklearn import linear_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__Model__')

"""Concrete class definitions for performing regression with gridRegression.model """
__author__ =  'Andrew O\'Harney'

class OnlineLinearRegression(model):
    """Naive linear regression
    is based on online updates"""
    
    __R__ = None #(X^t)y
    __cov__ = None #(X^t)X
            
    def partial_fit(self,X,y):
        
        #Expand dimensions of vectors
        if X.ndim < 2:
            X = X[np.newaxis,:]
        #Update for these values
        if self.__cov__ is None:
            _,b = X.shape
            self.__cov__ = np.zeros([b,b])
        if self.__R__ is None:
            _,a = X.shape
            _,d = y.shape
            self.__R__ = np.zeros([a,d])
        self.__coefs__ = None
        
        #These dot operations need reviewed for efficiency with Numpy BLAS/ATLAS (is this being done efficiently?)
        self.__cov__ += np.dot(X.T,X)
        self.__R__ += np.dot(X.T,y)
       
    def setCoefs(self,coefs=None,regParam=None):
        
        if coefs is None:
            #Very *naive* method for calculating parameters - use with care! (This is temp and needs to be reviewed)
            self.__cov__ += np.diag(np.random.random(self.__cov__.shape[0])*1e-15) if regParam is None else np.eye(self.__cov__.shape[0])*regParam
            self.__coefs__ = np.dot(np.linalg.inv(self.__cov__),self.__R__)
            
        else:
            self.__coefs__ = coefs
        
    def predict(self,X):
        """Generate prediction for event(s)
        
        Keyword arguments:
        X -- Design for some event(s)"""
    
        if self.__coefs__ is None:
            self.setCoefs()
        return np.dot(X,self.__coefs__)
    

class OnlineSGDRegression(model):
    """Performs Stochastic gradient descent for linear regression.
    Wraps around scikits-learn"""
    
    __model__ = None #List of models (note scikits only allows one target per model)
    __ntargets__ = None #Number of targets
    
    def __init__(self):
        self.__coefs__ = None
    
    def partial_fit(self,X,y):
        thisNTargets = len(y.columns)
        
        #Sanity check on the number of targets
        if self.__ntargets__ is None:
            self.__ntargets__ = thisNTargets
        else:
            if self.__ntargets__ != thisNTargets:
                logger.warn('Number of targets does not match!')
                return
        self.__coefs__ = None
        
        #Create models if we don't have them so far
        if self.__model__ is None:
            self.__model__ = [linear_model.SGDRegressor() for i in range(self.__ntargets__)] #Model for each target variable        
        
        #Update each target model based on the design
        for i in range(self.__ntargets__):
            self.__model__[i].partial_fit(X,y.values[:,i])
        
    def setCoefs(self,coefs=None):
        #Set the coefficients to be used for prediction
        
        if coefs is None:
            self.__coefs__ = np.array([reglin.coef_ for reglin in self.__model__])
        else:
            self.__coefs__ = coefs
             
    def predict(self,X):
        #Calculate the coefficients if needed
        if self.__coefs__ is None:
            self.setCoefs()
        x,_ = X.shape
        prediction = np.zeros([x,self.__ntargets__])
        
        #Iterate each model and get prediction for each target
        for i in range(self.__ntargets__):
            model = self.__model__[i]
            prediction[:,i] = model.predict(X)
        return prediction


class regLearner(model):
    """Learner for performing regularisation with one of the base models"""
    __trainEvents__ = None
    __testEvents__ = None
    __model__ = None
    __regs__ = None
    __trainNum__ = None
    __trainCount__ = None
    __data__ = None

    def __init__(self,events,data,regs,model,trainPrcnt=75):
        #Events - Full training events
        #Regs - regularisation parameters to use
        #Data - training data
        #Model - Any model object to use as base
        #TrainPrcnt - Percentage of training set to use for training each reg parameter (default 75%)
        self.__trainNum__ = len(events)*trainPrcnt/100.0
        self.__trainEvents__ = events[:self.__trainNum__]
        self.__testEvents__ = events[self.__trainNum__:]
        self.__regs__ = regs
        self.__model__ = model
        self.__trainPrcnt__ = trainPrcnt
        self.__trainCount__ = 0
    
    def coefs(self):
        #Return the coefficients currently in use
        return self.__model__.coefs() 

    
    def setCoefs(self,coefs):
        #Set the coefficients to use for prediction
        self.__model__.setCoefs(coefs)
    
    
    def partial_fit(self,X,y):
        #Add training example(s) to model
        self.__model__.partial_fit(X, y)
        
    def train(self):
        pass
    
    def __regularise__(self):
        pass
    
    def predict(self,X):
        #Predict output for X
        self.__model__.predict(X)
        