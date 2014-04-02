import numpy as np
import pandas as pd
from andyClasses.SignalManager import SignalManager,longest_event
from numpy.polynomial import Legendre
import logging,warnings
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__GridRegression__')


class GridRegression:
    #Help with regression over grid data
    #Note that this class does not represent the design matrix explicitly but will instead calculate 
    #it additively based on single events. 
    
    #Private
    __noise__ = None
    __nlags__ = None
    __grid__ = None
    __noiseOrders__ = None
    __cov__ = None
    __coefs__ = None
    __events__ = None
    __longestEvent__ = None
    __encoding__ = None
    __lambda__ = None
    
    def __init__(self,grid,nlags=0,noiseOrders=None,encoding=None):
        self.setNumLags(nlags)
        self.setGrid(grid)
        self.setNoiseOrders(noiseOrders)
        self.setEncoding(encoding)

    class regMatrix():
        #Quick class to encapsulate the results from covariance calculations
        matrix = None
        index = None
        def __init__(self,matrix,index):
            self.matrix = matrix
            self.index = index
    
    
    #Getters
    def grid(self):
        return self.__grid__
    def noise(self):
        return self.__noise__
    def noiseOrders(self):
        return self.__noiseOrders__
    def nlags(self):
        return self.__nlags__
    def cov(self):
        #Returns a regMatrix object
        return self.__cov__
    def covMatrix(self):
        #Return the raw covariance matrix
        return self.cov().matrix
    def times(self):
        #Times used for training
        return self.cov().index
    def coefs(self):
        #Infered parameters
        return self.__coefs__
    def events(self):
        #Current events in use
        return self.__events__
    def longestEvent(self):
        #Longest event in samples
        return self.__longestEvent__
    def encoding(self):
        #Encoding matrix for gesign
        return self.__encoding__
    
    #Setters
    def setEvents(self,events):
        logger.info( 'Setting events')
        self.__events__ = events
    def setNoise(self,noise):
        logger.info( 'Setting noise matrix')
        #Noise should be a matrix of length=grid.times() describing noise in the signal over time
        self.__noise__ = noise
        
    def setCov(self,cov,times):
        logger.info( 'Setting covariance matrix')
        self.__cov__ = self.regMatrix(cov,times)
    def setCoefs(self,coefs):
        logger.info( 'Setting coefficients')
        self.__coefs__ = coefs
    def setGrid(self,grid):
        logger.info( 'Setting grid')
        #Change grid and update noise matrix if necessary
        self.__grid__ = grid
        
    def setNumLags(self,numLags):
        logger.info( 'Setting number of lags')
        self.__nlags__ = numLags
    def setLambda(self,lmbda):
        self.__lambda__ = lmbda
    def setNoiseOrders(self,noiseOrders):
        logger.info( 'Setting polynomial orders for noise matrix')
        #Set maximum order of noise and update the matrix
        self.__noiseOrders__= noiseOrders
        self.setNoise(self.genNoise(self.grid(), self.noiseOrders()))

    def setLongestEvent(self,longestEvent):
        logger.info( 'setting longest event time')
        self.__longestEvent__ = float(longestEvent)*self.grid().fs()
        
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
        #encoding - class that returns the encoding of a given event
        #prevCode - is desgin is dependent on a previous state then pass it here       
        #longest - longest event in seconds
        
        if encoding is None:
            encoding = self.encoding()
        
        if longest is not None:
            self.setLongestEvent(longest)
        else:
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
        noise = self.noise().ix[times]
        design = np.zeros([len(times),npoints+noise.shape[1]])
        eventEncoding = encoding[event]
        
        
        for i,time in enumerate(times.values):
            design[i,:] = np.hstack((np.zeros([npoints]),noise.ix[time]))
            design[i,classCols] = eventEncoding
            design[i,lagCols] = prevCode[lagCols-1]
            prevCode = design[i,:]
        return pd.DataFrame(design,index=times)
    
    def train(self,events,y,encoding=None,longest=None,regParams=[0]):
        #Least squares ridge regression (without loading entire design matrix) 
        #Longest - the maximum event time in seconds (useful for limiting feedback periods ect.)
        #Encoding - dictionary that has row encodings for a given label(not including lags)
                
        #Set longest event in samples
        if longest is not None:
            self.setLongestEvent(longest)
        if encoding is None:
            encoding = self.encoding()        
        self.setEvents(events)

        #Parameters
        nClasses = encoding.numClasses()
        npoints = self.nlags()*nClasses+nClasses+len(self.noise().columns) #Number of columns in design matrix
        
        #Covariance matrix of designMatrix
        # NOTE : OBVIOUS ABILITY for parallelisation 
        for lmda in regParams:
            covMatrix = np.zeros([npoints,npoints])
            R = np.zeros([npoints,len(y.columns)])
            times = np.array([])
            prevCode = np.zeros([npoints])
            #Get covariance for each event
            for i in range(len(events)):
                logger.info('Training progress (lambda %f): event %d/%d'%(lmda,i+1,len(events)))
                design = self.event_design(events.iloc[i],prevCode=prevCode) #Get design matrix for event
                t = design.index #times for the event
                design = design.values #Get values
                #
                R += np.dot(design.T,y.ix[t].values) #Regressor 
                covMatrix += np.dot(design.T,design) #Calculate partial covMatrix
                times = np.hstack((times,t.values)) #Add times to index
                prevCode = design[-1,:]
                
            #
            coefs = np.dot(np.linalg.inv(covMatrix+lmda*np.eye(len(covMatrix))),R) #Niave implementation but fine for out purposes
        self.setCov(np.array(covMatrix), times)
        self.setCoefs(coefs)
        self.setLambda(lmda)
        return self.coefs()
        
    def predict(self,events,coefs=None,longest=None):
        #Events to predict from
        #coefs - weighting parameters to use if not the infered ones from the training step
        #Longest event in num samples
        
        logger.info( 'Predicting')
        if coefs is None:
            coefs = self.coefs()
        
        if longest is not None:
            self.setLongestEvent(longest)
        
        longest = self.longestEvent()
        #Preallocate result size
        allNumPoints = np.array([])
        for i in range(len(events)):
            x = self.grid().event_times(event=events.iloc[i])[:longest]
            allNumPoints = np.hstack([allNumPoints,x])     
        
        result = pd.DataFrame(np.zeros([len(allNumPoints),coefs.shape[1]]),index=allNumPoints,columns=self.grid().wc())
        prevCode = np.zeros([len(coefs)])    

        #Calculate the prediction event-wise
        for i in range(len(events)):
            logger.info('Predicted event %d/%d'%(i+1,len(events)))
            
            design =  self.event_design(events.iloc[i],prevCode=prevCode)
            times = design.index
            design = design.values
            #
            result.ix[times] = np.dot(design,coefs) #Add times to index
        
        return result
    
def pull_signal(grid,designMatrix):
    return designMatrix.index


def gen_designMatrix(grid,events,eventCodes,longest=None,encoding=None,nlags=0):
      #Generates a design matrix from events to be used for regression
      #Grid - Grid events come from
      #Events - events that make up the design matrix
      #Event codes - events corresponding to regressors (others result in empty rows)
      #Longest - the maximum event time in seconds (useful for limiting feedback periods ect.)
      #nlags - number of lags in the design matrix
      warnings.warn('This function is deprecated by the now favored GridRegression class', DeprecationWarning, stacklevel=2)
      
      #Set longest event in samples
      if longest is None:
          longest = longest_event(events)
      else:
          longest *= grid.fs()
      
      print 'Generating design for %d'%(len(events))
      encoding  = dict(zip(eventCodes,np.arange(len(eventCodes))))
      
      #Parameters
      nClasses = len(eventCodes)
      npoints = nlags*nClasses+nClasses #Number of columns in design matrix
      
      #Pre-determine design matrix size for efficiency
      allNumPoints = []
      for (on,off) in events[['pulse.on','pulse.off']].values:
          x = grid.num_points(times=[on,off])
          allNumPoints.append(np.min([longest,x]))     
      designMatrix = np.zeros([int(np.sum(allNumPoints)),int(npoints)],dtype = 'bool')
      
      ix = np.array([]) #Stores the index for the design Matrix
      times = grid.times()
  
      #Create sparse binary masked array
      row = 0
      for i,[start,stop,label] in enumerate(events[['pulse.on','pulse.off','event.code']].values):
          start = grid.snap_time(start)
          stop = grid.snap_time(stop)
          
          numtimes = allNumPoints[i] #How many time points the event lasts for
          design = np.zeros((int(numtimes),int(npoints))) #The matrix portion for this event
  
          #Place the binary mask in the correct column (or leave 0 if isi)
          if encoding.get(label,-1) >= 0:
              design[:,int(encoding[label]*(nlags+1)):int(encoding[label]*(nlags+1)+nlags+1)] = np.tile(np.hstack((np.array([1]),np.zeros(nlags))), [numtimes,1]) #Set the appropriate column to 1
          
          #Append to the design matrix and record event times
          designMatrix[int(row):int(row+numtimes),:] = design #Add to event matrix
          startix = grid.time_to_index(start)
          eventix = times[np.arange(startix,startix+numtimes)].values
          ix = np.hstack((ix,eventix))
          row+=numtimes
  
      #Add in lags
      for start in np.arange(0,npoints,nlags+1):
          for i,pos in enumerate(np.arange(start+1,start+nlags+1)):
              master = designMatrix[:,start]
              designMatrix[i+1:,pos] = master[:-(i+1)]
      
      #Set column names
      cols  = []
      for event in eventCodes:
          cols += [event]*(nlags+1)
      
      #Return matrix thing as pandas dataframe
      df = pd.DataFrame(designMatrix,index=ix)
      df.columns = cols
      df.colEncoding = encoding
      return df
    
