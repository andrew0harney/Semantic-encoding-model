import numpy as np
import pandas as pd
from SignalManager import SignalManager
from numpy.polynomial import Legendre

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
    def trainTimes(self):
        return self.__cov__.index
    def coefs(self):
        return self.__coefs__
    def events(self):
        return self.__events__
    def longestEvent(self):
        return self.__longestEvent__
    def encoding(self):
        return self.__encoding__
    
    #Setters
    def setEvents(self,events):
        print 'Setting events'
        self.__events__ = events
    def setNoise(self,noise):
        print 'Setting noise matrix'
        #Noise should be a matrix of length=grid.times() describing noise in the signal over time
        self.__noise__ = noise
        
    def setCov(self,cov,times):
        print 'Setting covariance matrix'
        self.__cov__ = self.regMatrix(cov,times)
    def setCoefs(self,coefs):
        print 'Setting coefficients'
        self.__coefs__ = coefs
    def setGrid(self,grid):
        print 'Setting grid'
        #Change grid and update noise matrix if necessary
        self.__grid__ = grid
        
    def setNumLags(self,numLags):
        print 'Setting number of lags'
        self.__nlags__ = numLags
        
    def setNoiseOrders(self,noiseOrders):
        print 'Setting polynomial orders for noise matrix'
        #Set maximum order of noise and update the matrix
        self.__noiseOrders__= noiseOrders
        self.setNoise(self.genNoise(self.grid(), self.noiseOrders()))

    def setLongestEvent(self,longestEvent):
        print 'setting longest event time'
        self.__longestEvent__ = longestEvent
    def setEncoding(self,encoding):
        print 'Setting encoding to be used in design matrix'
        self.__encoding__ = encoding
    
    #ML functions
    def genNoise(self,grid,maxNoiseOrder):
        #Noise is a matrix of Legendre polynomials of 0<order<maxNoiseOrder
        #Additionally 60Hz sine and cosine waves are added to account for the DC component of EEG

        if self.grid() is not None and self.noiseOrders() is not None:
            print 'Generating noise matrix'
            legpoly = np.array([Legendre.basis(i)(np.arange(len(grid.times()))) for i in self.noiseOrders()]).T #Polynomials
            sw = np.sin(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Sine for AC component
            cw = np.cos(60 * np.arange(len(grid.times())) * 2 * np.pi / float(grid.fs())) # Cosine for AC component
            legpoly = np.column_stack((legpoly, sw, cw))
            return pd.DataFrame(legpoly,index=self.grid().times())

    def event_design(self,event,encoding=None,prevCode=None):
        #Generates the design matrix for an event
        #encoding - column encoding (not including lags) for the label in event
        #prevCode - is desgin is dependent on a previous state then pass it here       
        
        if encoding is None:
            encoding = self.encoding()
        
        nClasses = len(encoding)
        npoints = self.nlags()*nClasses+nClasses #Number of columns in design matrix
        
        if prevCode is None:
            prevCode = np.zeros([npoints])
        default = np.zeros([npoints])
        classCols = np.arange(nClasses)*(self.nlags()+1)
        lagCols= np.array([ix for ix in np.arange(npoints) if ix%(self.nlags()+1)])
        grid = self.grid()
        
        design = []
        [start,stop,label] = event[['pulse.on','pulse.off','label']].values
        times = grid.times()[grid.time_to_index(start):grid.time_to_index(stop)]
        noise = self.noise().ix[times]
        for time in times.values:
            code = np.hstack((np.zeros([npoints]),noise.ix[time]))
            code[classCols] = encoding.get(label,default)
            code[lagCols] = prevCode[lagCols-1]
            design.append(code)
            prevCode = code
        return pd.DataFrame(design,index=times)
    
    def train(self,events,y,encoding=None,longest=None):
        #Least squares parameter estimation (without loading entire design matrix)
        #Longest - the maximum event time in seconds (useful for limiting feedback periods ect.)
        #Encoding - dictionary that has row encodings for a given label(not including lags)
                
        #Set longest event in samples
        if longest is None:
            longest = np.max([longest_event(self.grid(),events),self.longestEvent()])
        else:
            longest = int(longest*self.grid().fs())
        
        if encoding is None:
            encoding = self.encoding()
        
        self.setEvents(events)
        self.setLongestEvent(longest)
        
        #Parameters
        nClasses = len(encoding)
        npoints = self.nlags()*nClasses+nClasses+len(self.noise().columns) #Number of columns in design matrix
        
        #Covariance matrix of designMatrix
        covMatrix = np.zeros([npoints,npoints])
        R = np.zeros([npoints,len(y)])
        times = np.array([])
        prevCode = np.zeros([npoints])
        #Get covariance for each event
        for i in range(len(events)):
            design =  self.event_design(events.iloc[i],prevCode=prevCode)[:longest] #Get design matrix for event
            t = design.index #times that 
            design = np.matrix(design.values)
            #
            R+=design.T*y.ix[t] #Regressor 
            covMatrix += design.T*design #Calculate partial covMatrix
            times = np.hstack((times,t.values)) #Add times to index
            prevCode = design[-1]
        #
        self.setCov(np.array(covMatrix), times)
        self.setCoefs(np.invert(covMatrix)*R)
        return self.cov()
    
    def predict(self,eventsPredict,coefs=None,longest=None):
        print 'Predicting'
        if coefs is None:
            coefs = self.coefs()
        if longest is None:
            longest = self.longestEvent()
       
        times = self.trainTimes()
        result = pd.DataFrame(np.zeros([coefs.shape[0],len(times)]),index=times)
        prevCode = np.zeros([len(coefs)])
        
        #Calculate the prediction event-wise
        for i in range(len(eventsPredict)):
            design =  self.event_design(eventsPredict.iloc[i],prevCode=prevCode)[:longest]
            t = design.index #times that 
            design = np.matrix(design.values) #Assign the correct part of the signal
            result[t] = design
        return result
