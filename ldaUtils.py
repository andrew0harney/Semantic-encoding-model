import glob
import re
import pickle
import numpy as np
import pandas as pd
import logging
import ntpath
import heapq
from gensim import models
from GridRegression import Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__ldaUtils__')


#Utility class to store the output of encodings
class LdaEncoding:
    name = None
    values = None
    topicN = None
    def __init__(self,name,values,topicN=0):
        self.name = name
        self.values = values
        self.topicN = topicN 
    def __cmp__(self,y,topicN=None):
        if topicN is None:
            topicN = self.topicN 
        return np.sign(self.values[topicN]-y.values[topicN])
    def __getitem__(self,topicN):
        return self.values[topicN]
    def __str__(self,topicN=None):
        return self.name if topicN is None else self.name + ' ' + str(self.values[topicN]) 
    def setTopicN(self,topicN):
        self.topicN = topicN

#Creates labeled dictionary of corpora for referencing
#Sample running:
#INFO:__ldaUtils__:Processing .../labeled_image_maps/003770.labels.txt
#INFO:__ldaUtils__:Processing .../labeled_image_maps/003771.labels.txt
#INFO:__ldaUtils__:Processing .../labeled_image_maps/003772.labels.txt
#Sample output:
#{3770: ['man', 'bull', 'people', 'stadium', 'dirt'],
#3771: ['grass', 'sky', 'trees', 'village'],
#3772: ['seal', 'rocks']}
def createLabeledCorpDict(labeledImageDictionaryName,sourceReg,output=None):
    #labeledImageDictionaryName - Name for the dictionary
    #sourceReg - regular expression to find labelled image files
    #Output - pickle the dictionary {true,false}
    
    if not glob.glob(labeledImageDictionaryName):
        docs = dict()
        for tFile in glob.glob(sourceReg):
            logger.info('Processing '+str(tFile))
            a =open(tFile).read().splitlines()
            doc=[]
            for line in a:
                line = re.findall(r"[\w']+",line)
                if len(line)>1:
                    for item in line:
                        item = item.lower()
                elif line != []:
                    item = line[0].lower()
                doc.append(item)
            docs[int(re.findall('[0-9]+', tFile)[0])] = list(set(doc))
            #docs[ntpath.basename(tFile)] = list(set(doc))
            
        if output is not None:
            pickle.dump(docs, file(labeledImageDictionaryName,'w'))
        return docs
    else:
        return pickle.load(file(labeledImageDictionaryName,'r'))



    
    
#Utility class to encode an event for a given LDA model
class LdaEncoder(Encoder):
        #
        __ldaDict__ = None
        __ldaModel__= None
        __docs__ = None    
        __modelWordList__  = None
        __numClasses__ = None
        #
        def __init__(self,ldaDict,docs,lda):
            #
            self.__ldaDict__ = ldaDict
            self.__ldaModel__ = lda
            self.__numClasses__ = lda.num_topics
            self.__docs__ = docs
            self.__modelWordList__ = [self.__ldaModel__.id2word[wordid] for wordid in self.__ldaDict__] #Get valid words for this model
            #
        def numClasses(self):
            return self.__numClasses__
        #
        def __getitem__(self,event,eps=0):
            #Get stim fname
            stimName = event['label']
            #If it is a stimulus period
            if stimName >= 0:
                stimWords = self.__docs__[stimName] #Get the labels for the given stimulus
                topicProbs= self.model().__getitem__(self.__ldaDict__.doc2bow([word for word in stimWords if word in self.__modelWordList__]),eps=eps) #Get the topic encoding
                #Series with {topicNum:prob} structure
                return pd.Series([tprob for (_,tprob) in topicProbs],index=[topicNum for (topicNum,_)in topicProbs])
            else: #If it is an isi
                return np.zeros([self.model().num_topics])