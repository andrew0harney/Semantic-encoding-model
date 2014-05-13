import numpy as np
import pandas as pd
from numpy.polynomial import Legendre
import logging
from abc import ABCMeta, abstractmethod
import pdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__GridRegression__')
  
"""Classes for performing online regression over data"""
__author__ =  'Andrew O\Harney'

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

#
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
        """Predict output for event(s)
        
        Keyword arguments:
        X -- Design matrix of event(s)"""
        pass

    
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
    
    def __init__(self,model,grid,nlags=0,noiseOrders=None,encoding=None):
        self.setNumLags(nlags)
        self.setGrid(grid)
        self.setNoiseOrders(noiseOrders)
        self.setEncoding(encoding)
        self.setnClasses(encoding.numClasses())
        self.setnPoints(self.nlags()*self.nClasses()+self.nClasses()+len(self.noise().columns)) #Number of columns in design matrix
        self.setModel(model)
    #Getters
    def grid(self):
        return self.__grid__
    def noise(self):
        return self.__noise__
    def nClasses(self):
        return self.__nClasses__
    def nPoints(self):
        return self.__nPoints__
    def noiseOrders(self):
        return self.__noiseOrders__
    def nlags(self):
        return self.__nlags__
    def coefs(self):
        #Inferred parameters
        return self.__model__.coefs()
    def events(self):
        #Current events in use
        return self.__events__
    def longestEvent(self):
        #Longest event in samples
        return self.__longestEvent__
    def encoding(self):
        #Encoding matrix for gesign
        return self.__encoding__
    def model(self):
        return self.__model__
    def alphaH(self):
        return self.coefs()[:-self.noise().shape[1],:]
    def betaH(self):
        return self.coefs()[self.noise().shape[1]:,:]
    #Setters
    def setEvents(self,events):
        logger.info( 'Setting events')
        self.__events__ = events
    def setNoise(self,noise):
        logger.info( 'Setting noise matrix')
        #Noise should be a matrix of length=grid.times() describing noise in the signal over time
        self.__noise__ = noise
    def setnClasses(self,nclasses):
        self.__nClasses__ = nclasses
    def setnPoints(self,npoints):
        self.__nPoints__ = npoints
    def setCoefs(self,coefs):
        logger.info( 'Setting coefficients')
        self.__model__.setCoefs(coefs)
    def setGrid(self,grid):
        logger.info( 'Setting grid')
        #Change grid and update noise matrix if necessary
        self.__grid__ = grid
    def setNumLags(self,numLags):
        logger.info( 'Setting number of lags')
        self.__nlags__ = numLags
    def setLambda(self,lmbda):
        self.__lambda__ = lmbda
    def setModel(self,model):
        self.__model__ = model
    def setNoiseOrders(self,noiseOrders):
        logger.info( 'Setting polynomial orders for noise matrix')
        #Set maximum order of noise and update the matrix
        self.__noiseOrders__= noiseOrders
        self.setNoise(self.genNoise(self.grid(), self.noiseOrders()))

    def setLongestEvent(self,longestEvent):
        logger.info( 'setting longest event time')
        self.__longestEvent__ = int(longestEvent*self.grid().fs())
        
    def setEncoding(self,encoding):
        logger.info( 'Setting encoding to be used in design matrix')
        self.__encoding__ = encoding
    
    #ML functions
    def genNoise(self,grid,maxNoiseOrder):
        #Noise is a matrix of Legendre polynomials of 0<order<maxNoiseOrder
        #Additionally 60Hz sine and cosine waves are added to account for the DC component of EEG

        if self.grid() is not None and self.noiseOrders() is not None:
            logger.info( 'Generating noise matrix')
            legpoly = np.array([Legendre.basis(i)(np.arange(len(grid.times()))) for i in self.noiseOrders()]).T #Polynomials
            sw = np.sin(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Sine for AC component
            cw = np.cos(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Cosine for AC component
            legpoly = np.column_stack((legpoly, sw, cw))
            return pd.DataFrame(legpoly,index=self.grid().times())


    def event_design(self,event,encoding=None,prevCode=None,longest=None):
        #Generates the design matrix for an event
        #encoding - class that returns the encoding of a given event as a series {topicNumber:probability}
        #prevCode - is design is dependent on a previous state then pass it here       
        #longest - longest event in seconds
        
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
        #Generates the event matrix for events (in memory). Use with caution!     
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
        #Generator for event matrix blocks
        
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
            
    ################Normalisation functions##################
    def meanDesign(self,events):
        #Calculates the mean on columns of the full events matrix
        logger.info('Calculating design mean')
        designMean = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        N = 0
        for X,times in design:
            designMean += np.sum(X[:self.__longestEvent__,:],axis=0)
            N += len(times)
        return designMean / N
    
    def l1Norm(self,events):
        #Calculates the l1 normalisation parameter on columns of the design matrix
        
        logger.info('Calculating l1Norm')
        l1 = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        for X,_ in design:
            l1 += np.sum(np.abs(X[:self.__longestEvent__,:]),axis=0)    
        return np.sqrt(l1)
    
    def l2Norm(self,events):
        #L2 norm for real valued event matrix
        logger.info('Calculating l2Norm')
        l2 = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        for X,_ in design:
            l2 += np.sum(X[:self.__longestEvent__,:]**2,axis=0)    
        return np.sqrt(l2)
        
    def varDesign(self,events,mean=None):
        #Calculates variance on columns of design matrix
        
        if mean is None:
            mean = self.meanDesign(events)

        logger.info('Calculating design variance')
        designVar = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        N = 0
        for X,times in design:
            designVar += np.sum((X[:self.__longestEvent__]-mean)**2,axis=0)
            N += len(times)
        return designVar / (N-1)
    
    
    ###########Training functions###################
    def train(self,events,y,encoding=None,longest=None,normalise=None):
        """Online training of regression over events 
        Longest -- Maximum event time in seconds (useful for limiting feedback periods ect.)
        Encoding -- Dictionary that has row encodings for a given label(not including lags)
        Normalisation -- Normalisation coefficients to apply to columns of the design {zscore,l1,l2}
        Model -- Type of model to use {linear,SGD}"""
        model = self.model()
        if longest is not None:
            self.setLongestEvent(longest)
        if encoding is not None:
            self.setEncoding(encoding)
        if model is None:
            logger.info('Must specify a model type')
            return
        if normalise is not None:
            designMean = self.meanDesign(events)
            if normalise == 'zscore':
                designVar = self.varDesign(events,self.__designMean__)
                designSTD = designVar**0.5
                self.__normFunc__ = lambda x: (x-designMean)/designSTD
            elif normalise == 'l2':
                l2Norm = self.l2Norm(events)
                self.__normFunc__ = lambda x: (x-designMean)/l2Norm
            elif normalise == 'l1':
                l1Norm = self.l1Norm(events)
                self.__normFunc__ = lambda x : (x-designMean)/l1Norm
            else:
                normalise = None         
        self.setEvents(events)
        
        #Predict
        logger.info('Training events')
        design = self.__iterX__(events)
        
        #Generate the design for each event
        #self.__model__.train(design)
        for j,(X,times) in enumerate(design):
            X_block = self.__normFunc__(X) if normalise else X #Normalise if needed
            X_block[np.isnan(X_block)] = 0 #Replace any nans from normalisation with 0 (e.g var=0 causes this)
            self.__model__.partial_fit(X_block,y.ix[times]) #Fit this event to the model
        return self.__model__.coefs()
    
    def predict(self,events,coefs=None):

        if coefs is None:
            coefs= self.coefs()
        self.setCoefs(coefs)
    
        logger.info('Predicting events')
        design = self.__iterX__(events)
        ntargets = len(self.grid().wc())
        grid = self.grid()
        predictTimes = grid.eventsTimes(events,limit=self.longestEvent()/grid.fs())
        prediction = pd.DataFrame(np.zeros([len(predictTimes),ntargets]),index=predictTimes,columns=grid.wc())
        
        #Generate design for each event
        for j,(X,times) in enumerate(design):
            X_block = self.__normFunc__(X) if self.__normFunc__ else X #Normalise if needed
            X_block[np.isnan(X_block)] = 0
            prediction.ix[times]= self.__model__.predict(X_block)
        return prediction 
