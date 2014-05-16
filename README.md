The Semantic-Encoding model is split into two main challanges. Firstly, deriving a set of event categories/topics and secondly, using an encoding of these to correlate signals. In this way the object of the framework is to allow for:

  - The prediction of signals based on distributions over topics
  - The predictions of images based on the signal

In pursuit of these aims the code contained here is comprissed of two main functional pieces. The first is the derivation of topic models, which given a set of documents comprised of some vocabulary, will try to produce a topic distribution over the vocabulary. In this fashion, it would be possible to categorise any document based on it's vocabulary. This is done using the excellent Gensim framework. To see how this can be done take a look at exampleLDAScripts. 

Secondly (and the bulk of the code here) is to predict signal by some supervised training based on topic labels. This is done using the GridRegression class. This class can be insantiated by supplying the following:
  - signalManager - SignalManager object to provide information about the signals
  - n 


---------------------------------
Requirements Notes:
Check the accompanying docs for particular requirments of each module. Especially not the following:

- signal management utility - https://github.com/andrew0harney/Multi-channel-signal-management-utility
- gensim - (great LDA package http://radimrehurek.com/gensim/)
- nltk - If you wish to perform advanced natural language processing for topic clustering http://www.nltk.org/


---------------------------------
Notes:

  - You should ensure that your numpy installation is configured to use ATLAS/BLAS libraries. These are efficient linear algebra libraries that will severely improve performance. 

  - While some effort was made to keep this code general, it was ultimately written with a specific project in mind. Therefore it is not intended as a general release package, however, I hope it may provide some use and it should still be straightforward to adapt.

- Please also refer to to the accompanying license before using any of this code.
