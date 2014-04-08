import numpy as np
import pandas as pd
from andyClasses.SignalManager import SignalManager,longest_event
from numpy.polynomial import Legendre
import logging,warnings
import pickle
from sklearn import linear_model
from sklearn import preprocessing


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
    __R__ = None
    __coefs__ = None
    __events__ = None
    __longestEvent__ = None
    __encoding__ = None
    __lambda__ = None
    __nClasses__ = None
    __nPoints__ = None
    __designMean__ = None
    __designVar__ = None
    
    def __init__(self,grid,nlags=0,noiseOrders=None,encoding=None):
        self.setNumLags(nlags)
        self.setGrid(grid)
        self.setNoiseOrders(noiseOrders)
        self.setEncoding(encoding)
        self.setnClasses(encoding.numClasses())
        self.setnPoints(self.nlags()*self.nClasses()+self.nClasses()+len(self.noise().columns)) #Number of columns in design matrix

    class regMatrix():
        #Quick class to encapsulate the results from covariance calculations
        values = None
        times = None
        def __init__(self,matrix,index):
            self.values = matrix
            self.times = index

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
    def cov(self):
        #Returns a regMatrix object
        return self.__cov__
    def covMatrix(self):
        #Return the raw covariance matrix
        return self.cov().values
    def times(self):
        #Times used for training
        return self.cov().times
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
    def alphaH(self):
        return self.coefs()[:,:-self.noise().shape[1]]
    def betaH(self):
        return self.coefs()[:,self.noise().shape[1]:]
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
        #encoding - class that returns the encoding of a given event
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
        noise = self.noise().ix[times]
        design = np.zeros([len(times),npoints+noise.shape[1]])
        eventEncoding = encoding[event]
        
        
        for i,time in enumerate(times.values):
            design[i,:] = np.hstack((np.zeros([npoints]),noise.ix[time]))
            design[i,classCols] = eventEncoding
            design[i,lagCols] = prevCode[lagCols-1]
            design = design
            prevCode = design[i,:]
        return pd.DataFrame(design,index=times)
            
    def genX(self,events):         
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
    
    
    def train(self,events,y,encoding=None,longest=None,regParams=None,normalise=False):
        #Train LS regression. Uses memory map for large X
        if longest is not None:
            self.setLongestEvent(longest)
        if encoding is not None:
            self.setEncoding(encoding)
        
        X = self.__genX__(events)
        #Predict
        logger.info('Inferring parameters')
        reglin = linear_model.SGDRegressor(fit_intercept=True)
        reglin.fit(X, y.ix[X.times])
        coefs = reglin.coef_
        coefs[:,-self.noise().shape[1]] = reglin.intercept_
        self.setCoefs(coefs)
        self.setCov(X, X.times)
        return self.coefs()   
    
    def predict(self,events,coefs=None,longest=None):
        logger.info( 'Predicting')
        if coefs is None:
            coefs = self.coefs()
        
        if longest is not None:
            self.setLongestEvent(longest)
        X = self.__genX__(events)
        self.setCov(X, X.times)
        return pd.DataFrame(np.dot(X,coefs),columns = self.grid().wc(),index=X.times)
    
    #####################
    #UNDER DEVELOPMENT:  ONLINE RL REGRESSION METHODS
    def __iterX__(self,events):      
     
        encoding = self.encoding()             
        longest=self.longestEvent()      
        nClasses = encoding.numClasses()       
        npoints = self.nlags()*nClasses+nClasses #Number of columns in design matrix

        prevCode = np.zeros([npoints])
        classCols = np.arange(nClasses)*(self.nlags()+1)
        lagCols= np.array([ix for ix in np.arange(npoints) if ix%(self.nlags()+1)])
        grid = self.grid()
        
        for i in range(len(events)):
            event = events.iloc[i]
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
            yield design,times
            
    def meanDesign(self,events):
        
        logger.info('Calculating design mean')
        designMean = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        N = 0
        for X,times in design:
            designMean += np.sum(X,axis=0)
            N += len(times)
        return designMean / N
    
    def varDesign(self,events,mean=None):
        if mean is None:
            mean = self.meanDesign(events)

        logger.info('Calculating design variance')
        designVar = np.zeros(self.nPoints())
        design = self.__iterX__(events)
        N = 0
        for X,times in design:
            designVar += np.sum((X-mean)**2,axis=0)
            N += len(times)
        return designVar / N
      
        
    def online_train(self,events,y,encoding=None,longest=None,normalise=True):
        #Least squares ridge regression (without loading entire design matrix) 
        #Longest - the maximum event time in seconds (useful for limiting feedback periods ect.)
        #Encoding - dictionary that has row encodings for a given label(not including lags)    
        if longest is not None:
            self.setLongestEvent(longest)
        if encoding is not None:
            self.setEncoding(encoding)
        if normalise:
            self.__designMean__ = self.meanDesign(events)
            self.__designVar__ = self.varDesign(events,self.__designMean__)
            designSTD = self.__designVar__**0.5
        self.setEvents(events)
        mv = len(self.grid().wc())
        
        #Predict
        logger.info('Inferring parameters')
        design = self.__iterX__(events)
        reglins = [linear_model.SGDRegressor(fit_intercept=True) for i in range(mv)]
        
        for j,(X,times) in enumerate(design):
            logger.info('Processing event %d'%(j))
            X_block = (X-self.__designMean__)/designSTD if normalise else X #Zscore
            X_block[np.isnan(X_block)] = 0
            for i in range(mv):
                reglins[i].partial_fit(X_block,y.ix[times].values[:,i])
        
        coefs = np.array([reglin.coef_ for reglin in reglins])
        coefs[:,-self.noise().shape[1]] = [reglin.intercept_ for reglin in reglins]
        self.setCoefs(coefs)
        return self.coefs()
    
    def online_predict(self,events,coefs=None):
        pass
    
    
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
    
