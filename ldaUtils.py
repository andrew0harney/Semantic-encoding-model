import glob
import re
import pickle
import numpy as np
import pandas as pd

#Creates labeled dictionary of corpora for referencing
def createLabeledCorpDict(labeledImageDictionaryName,sourceReg,output=None):
    if glob.glob(labeledImageDictionaryName):
        docs = dict()
        for tFile in glob.glob(sourceReg): 
            a =open(tFile).read().splitlines()
            doc=[]
            for line in a:
                line = re.findall(r"[\w']+",line)
                if len(line)>1:
                    for item in line:
                        item = item.lower()
                else:
                    item = line[0].lower()
                doc.append(item)
            docs[int(re.findall('[0-9]+', tFile)[0])] = list(set(doc))
            
        if output is not None:
            pickle.dump(docs, output)
        return docs



#Utility class to encode an event for a given LDA model
class LdaEncoder:
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
        def __getitem__(self,event,n=0):
            #Get stim fname
            stimName = event['label']
            #If it is a stimulus period
            if stimName >= 0:
                stimWords = self.__docs__[stimName] #Get the labels for the given stimulus
                topicProbs= self.model().__getitem__(self.__ldaDict__.doc2bow([word for word in stimWords if word in self.__modelWordList__]),eps=n) #Get the topic encoding
                #Series with {topicNum:prob} structure
                return pd.Series([tprob for (_,tprob) in topicProbs],index=[topicNum for (topicNum,_)in topicProbs])
            else: #If it is an isi
                return np.zeros([self.model().num_topics])
            #
        def get_code(self,event):
            return self.__getitem__(event)
        def model(self):
            return self.__ldaModel__