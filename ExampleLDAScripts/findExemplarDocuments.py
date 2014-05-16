from ldaUtils import LdaEncoder,LdaEncoding,createLabeledCorpDict
import numpy as np
from gensim import models
import pickle
import heapq

#Andrew O'Harney 28/04/14
#This scripts produces nExemplars for each of the topic models 
#(Ordered by probability of belonging to a topic)


nExemplars = 10
labeledDocuments = #
imgaeSourceReg = #

#Load documents to be examined
docs = createLabeledCorpDict(labeledDocuments,imgaeSourceReg,output=True)
fnames = docs.keys() #Get file names

modelDir = #
    
#Load LDA model
modelName = #
lda = models.LdaModel.load(modelDir+modelName+'model')
ldaDict = pickle.load(open(modelDir+modelName+'dictionary','r'))
ldaEncoder = LdaEncoder(ldaDict,docs,lda)

#Probability encoding of each documents
encoding = []

#Encode each of the files
for fname in fnames:
    encoding.append(LdaEncoding(fname,ldaEncoder[{'label':fname}]))

#Output the topic nExemplars for each topic
outf = file(modelDir+modelName+'exemplars','w')
for i in range(ntopics):
    print 'Fininding exempalars for topic '+str(i)
    [e.setTopicN(i) for e in encoding] #Set the topic number to e compared
    exemplars = heapq.nlargest(nExemplars,encoding) #Create limited heap
    outf.write('Topic %d\n%s\n'%(i,'_'*10)) 
    outf.write(str([exemplar.__str__(topicN=i) for exemplar in exemplars])+'\n\n')
outf.close()    
