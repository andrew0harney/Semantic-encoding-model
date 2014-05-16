import numpy as np
import pandas as pd
from SignalManager import SignalManager
from GridRegression import GridRegression,Encoder
from models import OnlineLinearRegression
import logging
from scipy.stats import pearsonr
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn import linear_model
from ldaUtils import createLabeledCorpDict
import cProfile


#This script searches for the best number of lags to use for signal prediction in the case of known topic cateogies. 
#It is useful as a test to decouple with the LDA step. 

#Create an encoder that can encode a given filename with the know categories represented in that file
class CategoriesEncoder(Encoder):
        #
        __docs__ = None    
        __words__ = None
        #
        def __init__(self,docs):
            #
            self.__docs__ = docs
            self.__categories__ = set([word for doc in docs.values() for word in doc])
            #
        def numClasses(self):
            return len(self.__categories__)
        #
        def categories(self):
            return self.__categories__
        def __getitem__(self,event,eps=0):
            #Get stim fname
            stimName = event['label']
            #If it is a stimulus period
            encoding = pd.Series(np.zeros([self.numClasses()]),index=self.categories())
            if stimName > 0 and stimName in self.__docs__.keys():
                stimWords = self.__docs__[stimName] #Get the labels for the given stimulus
                stimWords = [word for word in stimWords if word in self.categories()]
                encoding[stimWords] = 1.0
                #Series with {topicCategory:prob} structure
            #
            return encoding

#Calculates the predictive ability of each LDA model while varying the number of lags
#This script does a lot and should probably be better structured/split up a bit
#def findBestLags():
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__findBestLags__')


#Load grid and data
data_path = #
log_path =  #
all_stims = #

grid = SignalManager(data_path, log_file=log_path)
baseChannels = grid.channels()[:-2]
chans = [chan for chan in grid.channels()]
grid.set_wd(channels=chans) 


#############################################################################################
#
# Type of events to use as data points
em = grid.event_matrix()
image = em[em['event.code'] == 2]
isi = em[em['event.code'] == 3]
#
#Sort chronologically
events = pd.concat((image,isi))
events = events.sort(columns='pulse.on')
#Get Training data
#
#Get train/test signal
logger.info('Getting data')
ys = grid.wd(channels=chans)
trainset = len(events) * 3 / 4
#
#Get training and test event sent
logger.info('Creating training and test sets')
eventsTrain = events[:trainset][['pulse.on', 'pulse.off', 'trial.id']]
eventsTest = events[trainset:][['pulse.on', 'pulse.off', 'trial.id']]

#Use the tiral ID as the label for encoding
eventsTrain.columns = [['pulse.on', 'pulse.off', 'label']]
eventsTest.columns = [['pulse.on', 'pulse.off', 'label']]
#
###################################################################################################

# Some parameters for  model
#Regression model
noiseOrds = [0,1,2,3] #Orders for nuisance
maxEventLength = 5 #Maximum length (in seconds) to take into account for any event
#
#Get object labels for all images (not a nice solution but hey-ho.)
logger.info('Finding image labels')
labeledImageDictionaryName = #
imgaeSourceReg = #
docs = createLabeledCorpDict(labeledImageDictionaryName,imgaeSourceReg,output=True)
#
#########################################################################################################
#Time to do some regression....
rootName = #
lags = np.arange(100,1100,100) #list of lags to explore
#Load the correct model and associated vocabulary
regressName = rootName
try:
    os.mkdir(regressName)       
except:
    pass
#
#Storing correlations across lags
pcorrs = np.zeros([len(lags),len(chans)])
pvals = np.zeros([len(lags),len(chans)])
encoder = CategoriesEncoder(docs)
#
#Explore the number of lags
for lagNum,nlags in enumerate(lags):
    #lagNum,nlags = 0,10
    lagName = regressName+'lags_'+str(nlags)
    os.mkdir(lagName)
    logging.basicConfig(level=logging.INFO,filname=lagName+'/outLog.txt')
    resultStore = pd.HDFStore(lagName+'/model.h5')
    resultStore['nlags'] = pd.Series([nlags])
    #Train
    logger.info('-'*10+str(nlags)+'-'*10)
    model = OnlineLinearRegression()
    gridRegression = GridRegression(model,grid,nlags,noiseOrds,encoding=encoder)
    #
    logger.info('Training')
    times = grid.eventsTimes(eventsTrain,limit=maxEventLength) 
    ytrain = ys.ix[times] #Get the training data
    resultStore['ytrain'] = ytrain
    coefs = gridRegression.train(eventsTrain,ytrain,longest=maxEventLength) #Train model on the train set
    prediction = gridRegression.predict(eventsTest)
    alphaH = gridRegression.alphaH() #Best estimate of parameters on training
    #
    ###################################
    #Train model on the test set
    logger.info('Testing')
    times = grid.eventsTimes(eventsTest,limit=maxEventLength)
    ytest = ys.ix[times] #Get the training data
    interNuisance = ytest.values
    noise = gridRegression.noise().ix[times].values
    #
    #Infer noise only for test data
    #Regression training data against noise matrix only to infer betaH on test (without effect of the rest of the design)
    reglin = linear_model.RidgeCV(fit_intercept=True,normalize=True)
    reglin.fit(noise, ytest)
    betaH = reglin.coef_
    betaH[:,0] = reglin.intercept_
    #
    ###############################################
    # Correlate the prediction and signal without drift
    predCoefs=np.hstack([alphaH.T,betaH]).T #Stack train predictive parameters and inferred test noise parameters
    prediction = gridRegression.predict(eventsTest,coefs=predCoefs) #Best prediction using alphaH[Train] and betaH[test]
    #
    pp = PdfPages(lagName+'/predictions.pdf')
    plt.hold(True)
    for i,chan in enumerate(chans):
    #      
        pcor = pearsonr(ytest[chan], prediction[chan].values)
        pcorrs[lagNum,i] = pcor[0]
        pvals[lagNum,i] = pcor[1]
        #
        plt.title('%s correlation:%f'%(chan,pcor[0]))
        plt.plot(ytest[chan].ix[:3000],label='Ytest detrended',color='#555555',linewidth=2.0)
        plt.plot(prediction[chan].ix[:3000],label='Prediction',color='#FF6600')
        plt.xlabel('Sample Number (s)')
        plt.ylabel('Signal Amplitude A(s)')
        plt.legend()
        pp.savefig()
        plt.clf()
    #
    #The last page contains histograms of the pearsons correlation and pvalues for each channel
    plt.subplot(211)
    plt.title('Pearsons')
    hist, bins = np.histogram(pcorrs[lagNum,:], bins=50)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2  #
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('Coefficient')
    #
    plt.subplot(212)
    plt.title('P-values')
    hist, bins = np.histogram(pvals[lagNum,:], bins=50)
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2  #
    plt.bar(center, hist, align='center', width=width)
    plt.xlabel('p-value')
    plt.tight_layout()
    pp.savefig()
    plt.close()
    pp.close()
    #
    resultStore['coefs'] = pd.DataFrame(predCoefs)
    resultStore['pcorrs'] = pd.DataFrame(pcorrs)
    resultStore['pvals'] = pd.DataFrame(pvals)
    resultStore.flush()
    resultStore.close()
    #
    coefs = coefs[:,:-1] #Remove the photodiode
    #bestLdaPredict[topicNum,wordNum] = np.mean(np.mean(pcorrs,axis=1))
    np.save(open(regressName+'/optimiseLags.np','w'), pcorrs)
#Plot the average channel correlation v the number of lags
pp = PdfPages(regressName+'/correlationVlags_mean.pdf')
plt.hold(True)
plt.xlabel('Number of lags')
plt.ylabel('Correlation')
plt.gca().grid(True,which="both")
plt.title('Optimising Lags')
plt.plot(lags,pcorrs.mean(axis=1))
plt.legend()
pp.savefig()
plt.clf()
pp.close()
