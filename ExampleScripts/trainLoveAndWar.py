import numpy as np
import
from gensim import corpora, models
import re
import os.path

warWordsPath = 'warWords.txt'
loveWordsPath = 'loveWords.txt'
stopsPath = 'stopWords.txt'

#Output names
vocabName='vocab.dic'
corpName='corp.mm'
modelName='lda.model'

#Generate fale documents
numDocs = 4000
wordsPerDoc = 10

if not os.path.isfile(vocabName) or not os.path.isfile(corpName):
  
    stops = [re.findall(r"[\w']+",word.lower()) for word in open(stopsPath).readlines()] 
    warWords = [re.findall(r"[\w']+",word.lower()) for word in open(warWordsPath).readlines() if word not in stops] 
    loveWords = [re.findall(r"[\w']+",word.lower()) for word in open(loveWordsPath).readlines()if word not in stops]
    
    warWords = list(set([word[0] for word in warWords])-set([word[0] for word in loveWords])) #Make word lists mutually exclusive
    print 'War words:%d\nLove words:%d'%(len(warWords),len(loveWords))
    
    warWords = np.array(warWords) #Whole vocabulary of words describing war
    loveWords = np.array(list(set([label for word in loveWords for label in word]))) #Whole vocabulary of words describing love
    
    
    #Create fake documents sets
    warDocs =[]
    loveDocs = []
    
    print "Generating %d fake documents"%numDocs
    for i in np.arange(numDocs):
        warIx=np.random.randint(0,len(warWords),[1,wordsPerDoc])
        loveIx = np.random.randint(0,len(loveWords),[1,wordsPerDoc])
        warDocs.append(warWords[[warIx]].tolist()[0])
        loveDocs.append(loveWords[[loveIx]].tolist()[0])
    
    print "Processing data"
    documents = warDocs + loveDocs
    vocab = corpora.Dictionary(documents)
    print "Unique words:%d"%len(vocab.token2id)
    vocab.save(fname=vocabName)
    
    documents = [' '.join(document) for document in documents]
    corp = [vocab.doc2bow(doc.split()) for doc in documents]
    corpora.MmCorpus.serialize(corpName, corp)
    
    
#Train LDA
print 'Loading vocab and corpus'
dictionary = corpora.Dictionary.load(vocabName)
corp = corpora.MmCorpus(corpName)

print 'LDA time...'
lda = models.ldamodel.LdaModel(corpus=corp, id2word=dictionary, num_topics=2, update_every=1, chunksize=len(corp), passes=100)
print lda.print_topics(2)
lda.save(fname=modelName)
