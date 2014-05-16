import numpy as np
import glob
from gensim import corpora, models
import pickle
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

#Andrew O'Harney 17/03/14
#Script to perform grid search of vocab size and number of topics on corpus
#Will save each generated gensim topic model and perplexity matrix
#Metric used for model performance is 90/10 train/test split of perplexity
#Full credit for the difficult part of this (LDA analysis) goes to Radim Řehůřek for 
#the excellent gensim toolkit http://radimrehurek.com/gensim/


#Turn on gensim logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionaryName = #Save path for dictionary (gensim encoded)
vocabName = #Save name for complete vocab (plain text of vocab)
corpName = #Save path for corpus (gensim encoded)
documentsName= #Save path for corpus (plain text)
stopsPath = #Path to list of stop words
#######################################################


#Useless stop words to be removed (contain no topic information)
stops = [re.findall(r"[\w']+",word.lower()) for word in open(stopsPath).readlines()]
stemmer = WordNetLemmatizer() 
fnames = []

#Train LDA algorithm from vocab of SUN dataset
all_obj = ''
documents = [re.findall(r"[\w']+",word) for word in open(all_obj).readlines()] #Documents/images

#Stem words
documents=[list(set([stemmer.lemmatize(word,'n') for word in document if [word] not in stops])) for document in documents]

 
#Get frequency counts of each word
freqCount = {}
for document in documents:
    for word in document:
        freqCount[word] = freqCount.get(word,0)+1

random.shuffle(documents)

#Store full corpus and dictionaries
print 'Creating Vocab'
dictionary = corpora.Dictionary(documents)
print 'Number of unique objects %d'%len(dictionary)
dictionary.save(fname=dictionaryName)
#
print 'Creating training corpus'
print 'Total number of images: %d'%len(documents)
pickle.dump(documents,open(documentsName,'w'))
corp = [dictionary.doc2bow(doc) for doc in documents]
corpora.MmCorpus.serialize(corpName, corp)

######################################
#Train LDA models
corp = list(corp)

perplexityName = #Save path for perplexity matrix
nwords = np.arange(200,900,100) #Range of vocab lengths to perform grid search
ntopics = np.arange(10,32,2)  #Range of topic lengths to perform grid search
perplexity = np.zeros([len(ntopics),len(nwords)])
trainSize = 0.9 #fraction of total documents to train on

resultsRoot = #Output folder paths for each of the lda classifiers
for i,num_words in enumerate(nwords):
    #
    #Use only num_words number of words in documents
    print 'Creating training corpus'
    t_dict = copy.copy(dictionary)
    t_dict.filter_extremes(no_below=0.001*len(documents),no_above=1,keep_n=num_words) #Removes infrequently occurring words (less than num_words and occurring in less than 0.1% of documents)
    t_docs = filter(None,[[word for word in doc if word in t_dict.values()] for doc in documents]) #Remove any words no longer accounted for and empty documents
    t_corp = [t_dict.doc2bow(doc) for doc in t_docs] #Create training corpus
    #
    #Training and test sets
    s = int(trainSize*len(t_corp))
    corpTrain = t_corp[:s]
    corpTest = t_corp[s:]
    #
    for j,num_topics in enumerate(ntopics):
        name = resultsRoot+'vocab_'+str(len(t_dict))+'_topics_'+str(num_topics)
        os.mkdir(name)
        print '# Words: %d \t # Topics: %d'%(len(t_dict),num_topics)
        #Train model
        #LDA training with asymetric priors on topic distribution and 50 iterations for VB
        lda = models.ldamodel.LdaModel(corpus=t_corp, id2word=t_dict, num_topics=num_topics, update_every=0, chunksize=len(t_corp), passes=50,alpha='auto')
        #Save data for this model
        lda.save(fname=name+'/model')
        t_dict.save(fname=name+'/dictionary')
        pickle.dump(t_docs,open(name+'/documents','w'))
        corpora.MmCorpus.serialize(name+'/corp', t_corp)
        perplexity[j,i]= np.exp2(-lda.bound(corpTest) / sum(cnt for document in corpTest for _, cnt in document)) #Normalized test perplexity
        #
	#Show word distributions for top 100 words for each topic
        pp = PdfPages(name+'/wordDistribution.pdf')
        for tn in np.arange(num_topics):
            plt.figure()
            plt.title('Topic %d'%(tn+1))
            #
            ldaOut=[s.split('*') for s in lda.print_topic(tn,100).split('+')]
            word_probs = [float(a) for (a,b) in ldaOut]
            words = [b for (a,b) in ldaOut]
            #
            plt.plot(np.arange(len(ldaOut)),word_probs)
            plt.xticks(np.arange(len(ldaOut)),words,rotation=45,size=2)
            pp.savefig()
            plt.close()
        pp.close()

np.save(open(perplexityName,'w'), perplexity)

