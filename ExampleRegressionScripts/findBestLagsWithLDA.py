import numpy as np
import pandas as pd
from andyClasses.SignalManager import SignalManager
from andyClasses.GridRegression import GridRegression
from gensim import models
import pickle
import logging
from scipy.stats import pearsonr
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn import linear_model
from andyClasses.ldaUtils import LdaEncoder
from andyClasses.ldaUtils import createLabeledCorpDict

#This script calculates the predictive ability on signals using regression methods. Encoding of the design
#comes from from a series of LDA models. LDA models are those generated by training over a range of 
#vocabulary and topic sizes (see the generateLDAModels example script). The script also varies the number of lags for regression.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__findBestLags__')

#Load grid
data_path = #
log_path =  #
modelDir = #
all_stims = #

#Load the signals and set working data
grid = SignalManager(data_path, log_file=log_path)
baseChannels = grid.channels()[:-2]
chans = [chan for chan in grid.channels()]
grid.set_wd(channels=chans,meanCalcChans=baseChannels) 

#############################################################################################

# Type of events to use as data points
em = grid.event_matrix()
image = em[em['event.code'] == 2]
isi = em[em['event.code'] == 3]
#
#Sort chronologically
events = pd.concat((image,isi))
events = events.sort(columns='pulse.on')

#Get train/test signal
logger.info('Getting data')
ys = grid.wd(channels=chans)
trainset = len(events) * 3 / 4

#Get training and test event sent
logger.info('Creating training and test sets')
eventsTrain = events[:trainset][['pulse.on', 'pulse.off', 'event.code']]
eventsTest = events[trainset:][['pulse.on', 'pulse.off', 'event.code']]

#Make the event.code the label to be used by the encoder
eventsTrain.columns = [['pulse.on', 'pulse.off', 'label']]
eventsTest.columns = [['pulse.on', 'pulse.off', 'label']]

###################################################################################################

# Some parameters for  model

#Regression model
noiseOrds = [0,1,2,3] #Orders for nuisance
maxEventLength = 5 #Maximum length (in seconds) to take into account for any event
regParams = [10**i for i in np.arange(-2,2)] #Regularisation parameters

#LDA model (note the models must have already been generated)
#nwords = np.arange(200,800,100) #Lda number of words
#ntopics = np.arange(10,32,2) #Lda number of topics
lags = np.arange(100,1100,100) #list of lags to explore
nwords = np.arange(300,900,100)
ntopics = np.arange(10,32,2)

#Get object labels for all images (not a nice solution but hey-ho.)
logger.info('Finding image labels')
labeledImageDictionaryName = #
imgaeSourceReg = #
docs = createLabeledCorpDict(labeledImageDictionaryName,imgaeSourceReg,output=True)


#########################################################################################################

#Time to do some regression....
rootName = #
#Explore the number of topics and words while varying the number of lags (i.e 3D grid 
for topicNum,numTopics in enumerate(ntopics):
    for wordNum,numWords in enumerate(nwords):
        #Load the correct model and associated vocabulary
        logger.info('*****Topics %d Words %d******'%(numTopics,numWords))
        modelName = 'vocab_%d_topics_%d/'%(numWords,numTopics)
        lda = models.LdaModel.load(modelDir+modelName+'model')
        ldaDict = pickle.load(open(modelDir+modelName+'dictionary','r'))
        ldaEncoder = ldaEncodingClass(ldaDict,docs,lda)
        regressName = rootName+'vocab_'+str(numWords)+'_topics_'+str(numTopics)
        os.mkdir(regressName)       
        #
        #Storing correlations across lags
        pcorrs = np.zeros([len(lags),len(chans)])
        pvals = np.zeros([len(lags),len(chans)])
        #Explore the number of lags
        for lagNum,nlags in enumerate(lags):
            lagName = regressName+'/lags_'+str(nlags)
            os.mkdir(lagName)
            resultStore = pd.HDFStore(lagName+'/model.h5')
            resultStore['nlags'] = pd.Series([nlags])
            #Train
            logger.info('-'*10+str(nlags)+'-'*10)
            gridRegression = GridRegression(grid,nlags,noiseOrds,encoding=ldaEncoder)
            #
            logger.info('Training')
            coefs = gridRegression.online_train(eventsTrain,ys,longest=maxEventLength,normalise='l2') #Train model on the train set
            times = grid.eventsTimes(eventsTest,limit=maxEventLength)
            resultStore['ytrain'] = ys.ix[times]
            alphaH = gridRegression.alphaH() #Best estimate of parameters on training
            #
            ###################################
            #Test
            logger.info('Testing')
            times = grid.eventsTimes(eventsTest,limit=maxEventLength) #Train model on the test set
            
            ytest = ys.ix[times] #Get the training data
            interNuisance = ytest.values
            noise = gridRegression.noise().ix[times].values
            #
            #Infer noise only for test data
            #Regression training data against noise matrix only to infer betaH on test (without effect of the rest of the design)
            reglin = linear_model.RidgeCV(fit_intercept=True,normalize=True)
            reglin.fit(noise, ys.ix[times])
            betaH = reglin.coef_
            betaH[:,0] = reglin.intercept_
            
            ###############################################
            # Correlate the prediction and signal without drift
            predCoefs=np.hstack([alphaH,betaH]).T #Stack train predictive parameters and inferred test noise parameters
            prediction = gridRegression.online_predict(eventsTest,coefs=predCoefs) #Best prediction using alphaH[Train] and betaH[test]
            #
            pp = PdfPages(lagName+'/predictions.pdf')
            plt.hold(True)
            for i,chan in enumerate(chans):
                #      
                pcor = pearsonr(ytest.values[i,:], prediction.values[i,:])
                pcorrs[lagNum,i] = pcor[0]
                pvals[lagNum,i] = pcor[1]
                #
                plt.title('%s correlation:%f'%(chan,pcor[0]))
                plt.plot(ytest[i,:],label='Ytest detrended')
                #plt.plot(b.values[i,:],label='Prediction')
                plt.xlabel('Sample Number (s)')
                plt.ylabel('Signal Amplitude A(s)')
                plt.legend()
                pp.savefig()
                plt.clf()
            #
            pp.close()
            #
            resultStore['pcorrs'] = pd.DataFrame(pcorrs)
            resultStore['pvals'] = pd.DataFrame(pvals)
            resultStore.flush()
            resultStore.close()
        #
        coefs = coefs[:,:-1] #Remove the photodiode
        np.save(open(regressName+'/optimiseLags.np','w'), pcorrs)
        #Plot the average correlation v number of lags
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

