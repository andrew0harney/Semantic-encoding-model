from ldaUtils import LdaEncoder,LdaEncoding,createLabeledCorpDict
import numpy as np
from gensim import models
import pickle
import heapq

#Andrew O'Harney 28/04/14
#Produce nExemplars for each of the topic models (Ordered by probability of belonging to a topic)

nExemplars = 10
labeledDocuments = '.../allDocuments'
imgaeSourceReg = '.../*.labels.txt'

#Load documents to be examined
docs = createLabeledCorpDict(labeledDocuments,imgaeSourceReg,output=True)
fnames = docs.keys()

modelDir = '...'
for ntopics in np.arange(10,31):
    for sizeVocab in np.arange(200,600,100):
    
        #Load LDA model
        modelName = 'vocab_%d_topics_%d/'%(sizeVocab,ntopics)
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
            [e.setTopicN(i) for e in encoding]
            exemplars = heapq.nlargest(nExemplars,encoding)
            outf.write('Topic %d\n%s\n'%(i,'_'*10))
            outf.write(str([exemplar.__str__(topicN=i) for exemplar in exemplars])+'\n\n')
        outf.close()    
