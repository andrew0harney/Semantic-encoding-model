import numpy as np
import pandas as pd
from numpy.polynomial import Legendre
from gridUtils import *
import logging
from abc import ABCMeta, abstractmethod
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__GridRegression__')
  
"""Functions and classes for performing online regression over data"""
__author__ =  'Andrew O\'Harney'
  

"""Abstract Classes"""
class Encoder():
    """Abstraction for providing encoding for an event."""
    
    __metaclass__ =  ABCMeta
    @abstractmethod
    def __getitem__(self,event,eps=0):
        pass
    @abstractmethod
    def numClasses(self):
        "Return the number of classes(topics) in data"
        pass

class model():
    """Base model class for all online Regression methods"""
    __metaclass__ = ABCMeta
    __coefs__ = None
    
    def coefs(self):
        """Return coefficients in use by the model"""
        if self.__coefs__ is None:
            self.setCoefs()
        return self.__coefs__    

    #Set the coefficients to use for prediction
    @abstractmethod
    def setCoefs(self,coefs):
        """Set the model parameters
        
        Keyword arguments:
        coefs -- Trained model parameters"""
        pass
    
    @abstractmethod
    def partial_fit(self,X,y):
        """Add training example(s) to model
        
        Keyword arguments:
        X -- Design matrix of event(s)
        y -- Data for event(s)"""
        pass
    
    @abstractmethod
    def predict(self,X):
        """Online output for event(s)
        
        Keyword arguments:
        X -- Design matrix of event(s)"""
        pass

    def train(self,X,y,normFunc=None):
        """Train model
        
        Keyword arguments:
        X -- Design matrix of event(s). Must be iterable
        y -- Data to train on
        normaliseFunc -- normalisation function"""
        
        for (Xi,times) in X:
            X_block = normFunc(Xi) if normFunc else Xi #Normalise if needed
            X_block[np.isnan(X_block)] = 0 #Replace any nans from normalisation with 0 (e.g var=0 causes this)
            self.partial_fit(X_block,y.ix[times]) #Fit this event to the model
   
"Concrete Classes"
class GridRegression:
    """Main class for encapsulated regression over data"""
    
    #Private
    __noise__ = None
    __nlags__ = None
    __grid__ = None
    __noiseOrders__ = None
    __coefs__ = None
    __events__ = None
    __longestEvent__ = None
    __encoding__ = None
    __lambda__ = None
    __nClasses__ = None
    __nPoints__ = None
    __normFunc__ = None
    __model__ = None           
    
    def __init__(self,model,signalManager,nlags=1,noiseOrders=None,encoding=None):
        """Keyword Arguments:
        model -- ML model to use for regression. Must be of type GridRegression.model
        signalManager -- Grid object containing data
        nlags -- Number of lags to use for each encoding category
        noiseOrders=None -- Polynomial orders of noise
        encoding=None -- Encoder for given event. Must be of type GridRegression.Encoder"""
        
        self.setNumLags(nlags)
        self.setGrid(signalManager)
        self.setNoiseOrders(noiseOrders)
        self.setEncoding(encoding)
        self.setnClasses(encoding.numClasses())
        self.setnPoints(self.nlags()*self.nClasses()+self.nClasses()+len(self.noise().columns)) #Number of columns in design matrix
        self.setModel(model)
    
    """Getters"""
    def grid(self):
        """Return grid in use"""
        return self.__grid__
    def noise(self):
        """Return noise matrix"""
        return self.__noise__
    def nClasses(self):
        """Return number of classes in data"""
        return self.__nClasses__
    def nPoints(self):
        """Return the number of encoding points in the design matrix (not including noise)"""
        return self.__nPoints__
    def noiseOrders(self):
        """Polynomial order of noise"""
        return self.__noiseOrders__
    def nlags(self):
        "Number of lags for each encoding category"
        return self.__nlags__
    def coefs(self):
        """Inferred Coefficients"""
        #Inferred parameters
        return self.__model__.coefs()
    def events(self):
        """Events used for training"""
        #Current events in use
        return self.__events__
    def longestEvent(self):
        """Longest event in the training set"""
        #Longest event in samples
        return self.__longestEvent__
    def encoding(self):
        """Encoding object (GridRegression.Encoder)"""
        return self.__encoding__
    def model(self):
        """Model in use"""
        return self.__model__
    def alphaH(self):
        """Inferred parameters for encoding (design) columns"""
        return self.coefs()[:-self.noise().shape[1],:]
    def betaH(self):
        """Inferred parameters for noise columns"""
        return self.coefs()[self.noise().shape[1]:,:]
    
    """Setters"""
    def setEvents(self,events):
        """Set events to be used for training
        Keyword Arguments:
        events -- Events to be used for training"""
        logger.info( 'Setting events')
        self.__events__ = events
    def setNoise(self,noise):
        """Set noise matrix
        noise -- Noise matrix to use"""
        logger.info( 'Setting noise matrix')
        #Noise should be a matrix of length=grid.times() describing noise in the signal over time
        self.__noise__ = noise
    def setnClasses(self,nclasses):
        """Set number of classes in data
        Keyword Arguments:
        nclasses -- Number of classes (n.b this does not include lag columns)"""
        self.__nClasses__ = nclasses
    def setnPoints(self,npoints):
        """Set the number of points used for encoding columns
        Keyword Arguments:
        npoints -- Number of points for encoding columns (not including noise columns)"""
        self.__nPoints__ = npoints
    def setCoefs(self,coefs):
        """Set the coefficients to be used by the model
        Keyword Arguments:
        coefs -- Weighting parameters for each row of the design matrix"""
        logger.info( 'Setting coefficients')
        self.__model__.setCoefs(coefs)
    def setGrid(self,grid):
        """Sets the grid in use
        Keyword Arguments:
        grid -- Grid to take data from"""
        logger.info( 'Setting grid')
        #Change grid and update noise matrix if necessary
        self.__grid__ = grid
    def setNumLags(self,numLags):
        """Set the number of lag columns
        Keyword Arguments:
        numLags: Number of lags to be used for each event class"""
        logger.info( 'Setting number of lags')
        self.__nlags__ = numLags
    def setModel(self,model):
        """Set the model to be used for regression
        Keyword Arguments:
        model -- Any model inheriting GridRegression.model"""
        self.__model__ = model
    def setNoiseOrders(self,noiseOrder):
        """Set the polynomial orders for noise
        Keyword Arguments:
        noiseOrders -- Maximum polynomial order to use for noise"""
        logger.info( 'Setting polynomial orders for noise matrix')
        #Set maximum order of noise and update the matrix
        self.__noiseOrders__= noiseOrder
        self.setNoise(self.genNoise(self.grid(), self.noiseOrders()))

    def setLongestEvent(self,longestEvent):
        """Set the longest event to consider in training and testing
        Keyword Arguments:
        longestEvent: Longest event period to consider in seconds"""
        logger.info( 'setting longest event time')
        self.__longestEvent__ = int(longestEvent*self.grid().fs())
        
    def setEncoding(self,encoding):
        """Set the encoder to be used for event encodings
        Keyword Arguments:
        encoding -- GridRegression.Encoder object"""
        logger.info( 'Setting encoding to be used in design matrix')
        self.__encoding__ = encoding
    
    """Machine learning functions"""
    def genNoise(self,grid,maxNoiseOrder):
        """Noise is a matrix of Legendre polynomials of 0<order<maxNoiseOrder
        Additionally 60Hz sine and cosine waves are added to account for the DC component of EEG
        grid -- Grid to be used for timing information
        maxNoiseOrder--Maximum order of noise to be considered"""

        if self.grid() is not None and self.noiseOrders() is not None:
            logger.info( 'Generating noise matrix')
            legpoly = np.array([Legendre.basis(i)(np.arange(len(grid.times()))) for i in self.noiseOrders()]).T #Polynomials
            sw = np.sin(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Sine for AC component
            cw = np.cos(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Cosine for AC component
            legpoly = np.column_stack((legpoly, sw, cw))
            return pd.DataFrame(legpoly,index=self.grid().times())


    def event_design(self,event,encoding=None,prevCode=None,longest=None):
        """Generates the design matrix for an event
        Keyword Arguments:
        encoding=None - Encoder Class (GridRegression.Encoder) that returns the encoding of a given event as a series {topicNumber:value}
        prevCode=None - If design is dependent on a previous state then pass it here
        longest=None - longest event in seconds (Defualt longest event in training)"""
        
        if encoding is None:
            encoding = self.encoding()
        
        if longest is not None:
            self.setLongestEvent(longest)
        
        longest=self.longestEvent()
        
        nClasses = encoding.numClasses()
        logger.debug('nClasses : %d'%(nClasses))
        
        npoints = self.nlags()*nClasses+nClasses #Number of columns in design matrix
        logger.debug('npoints : %d'%(npoints))
        
        if prevCode is None:
            prevCode = np.zeros([npoints])
        classCols = np.arange(nClasses)*(self.nlags()+1)
        lagCols= np.array([ix for ix in np.arange(npoints) if ix%(self.nlags()+1)])
        grid = self.grid()

        times = grid.event_times(event)[:longest] #Get the event times (limited to the maximum event length)
        logger.debug('Event size : %d'%(len(times)))
        noise = self.noise().ix[times] #Get the corresponding noise values for the times of this event
        design = np.zeros([len(times),npoints+noise.shape[1]])
        eventEncoding = encoding[event].values #Specifies the encoding for this class (without lags)
        
        
        for i,time in enumerate(times.values):
            design[i,:] = np.hstack((np.zeros([npoints]),noise.ix[time]))
            design[i,classCols] = eventEncoding #Put the encoding in the t=0 lag columns
            design[i,lagCols] = prevCode[lagCols-1] #Set the lag columns
            design = design
            prevCode = design[i,:]
        return pd.DataFrame(design,index=times)

    def genX(self,events):
        """Generates the event matrix for events (in memory). Use with caution!
        Keyword Arguments:
        events -- Events to generate design matrix for"""
        from tempfile import mkdtemp
        import os.path as path
                       
        #Set longest event in samples
        longest = self.longestEvent()
        allNumPoints = self.grid().eventsTimes(events,limit=longest)
        self.setEvents(events)
        npoints = self.nPoints()
        logger.info('Preparing design matrix')
        filename = path.join(mkdtemp(), 'tempX.dat')
        X = np.memmap(filename, dtype='float32', mode='w+', shape=(len(allNumPoints),npoints)) 
        X.times = allNumPoints
        
        #Build X
        prevCode = np.zeros([npoints])
        pos = 0
        for i in range(len(events)):
            logger.info('Generating event %d/%d design'%(i+1,len(events)))
            design = self.event_design(events.iloc[i],prevCode=prevCode) #Get design matrix for event
            t = design.index #times for the event
            design = design.values
            X[pos:pos+len(t),:] = design
            #
            pos+=len(t)
            prevCode = design[-1,:]
        
        return X
    
    def __iterX__(self,events):      
        """Generator for event matrix blocks
        Returns an iterator for event designs (better than in memory construction of the whole design)
        Keyword Arguments:
        events -- Events to generate design matrix for"""
        
        encoding = self.encoding()             
        longest=self.longestEvent()      
        
        #Find useful indices
        nClasses = encoding.numClasses()       
        npoints = self.nlags()*nClasses+nClasses #Number of columns in design matrix
        _,nNoisePoints = self.noise().shape
        
        prevCode = np.zeros([npoints])
        classCols = np.arange(nClasses)*(self.nlags()+1)
        lagCols= np.array([ix for ix in np.arange(npoints) if ix%(self.nlags()+1)])
        grid = self.grid()
        #design = np.zeros(npoints+nNoisePoints) #Allocate memory for the design
        
        for i in range(len(events)):
            logger.info('Generating event %d design'%(i+1))
            event = events.iloc[i]
            times = grid.event_times(event)[:longest] #Get the event times (limited to the maximum event length)
            logger.debug('Event size : %d'%(len(times)))
            noise = self.noise().ix[times] #Get the noise corresponding to this event
            design = np.zeros([len(times),npoints+nNoisePoints]) #Allocate memory for the design
            eventEncoding = encoding[event].values #Get the event encoding
            for i,time in enumerate(times.values):
                design[i,:] = np.hstack((np.zeros([npoints]),noise.ix[time])) 
                design[i,classCols] = eventEncoding #Set the t=0 lag columns
                design[i,lagCols] = prevCode[lagCols-1]  #Set the lags
                prevCode = design[i,:]
                #design[:] = 0
                #design[npoints:]=noise.ix[time] 
                #design[classCols] = eventEncoding #Set the t=0 lag columns
                #design[lagCols] = prevCode[lagCols-1]
                #prevCode = design
            yield design,times
                
    def train(self,events,y,encoding=None,longest=None,normalise=None):
        """Online training of regression over events 
        Keyword Arguments:
        events -- events to train from 
        y -- training data
        encoding=None -- Dictionary that has row encodings for a given label(not including lags)
        longest=None -- Limit the maximum event length in seconds
        normalisation=None -- Normalisation coefficients to apply to columns of the design {zscore,l1,l2}
        """
        model = self.model()
        
        if longest is not None:
            self.setLongestEvent(longest)
            longest = self.longestEvent()
        else:
            self.setLongestEvent(longest_event(events))
        if encoding is not None:
            self.setEncoding(encoding)
        if model is None:
            logger.info('Must specify a model type')
            return
        if normalise is not None:
            designMean = meanDesign(self.nPoints(),self.__iterX__(events),longest=longest)
            if normalise == 'zscore':
                designVar = varDesign(self.nPoints(),__iterX__(events),designMean,longest=longest)
                designSTD = designVar**0.5
                self.__normFunc__ = lambda x: (x-designMean)/designSTD
            elif normalise == 'l2':
                l2Norm = l2Norm(self.nPoints(),__iterX__(events),longest=longest)
                self.__normFunc__ = lambda x: (x-designMean)/l2Norm
            elif normalise == 'l1':
                l1Norm = l1Norm(self.nPoints(),__iterX__(events),longest=longest)
                self.__normFunc__ = lambda x : (x-designMean)/l1Norm
            else:
                normalise = None         
        self.setEvents(events)
        
        #Predict
        logger.info('Training events')
        design = self.__iterX__(events)
        
        #Train model
        model.train(design, y, self.__normFunc__)
        return model.coefs()
    
    def predict(self,events,coefs=None):
        """Perform prediction
        Keyword Arguments:
        events -- Events to predict signal for
        coefs=None -- Design weightings to be used for prediction (defualt is those inferred from training)"""
        if coefs is None:
            coefs= self.coefs()
        self.setCoefs(coefs)
    
        logger.info('Predicting events')
        design = self.__iterX__(events)
        ntargets = len(self.grid().wc())
        grid = self.grid()
        predictTimes = grid.eventsTimes(events,limit=self.longestEvent()/grid.fs())
        prediction = pd.DataFrame(np.zeros([len(predictTimes),ntargets]),index=predictTimes,columns=grid.wc())
        model = self.model()
        normFunc = self.__normFunc__
        
        #Generate design for each event
        for j,(Xi,times) in enumerate(design):
            X_block = normFunc(Xi) if normFunc else Xi #Normalise if needed
            X_block[np.isnan(X_block)] = 0
            prediction.ix[times]= model.predict(X_block)
        return prediction 
