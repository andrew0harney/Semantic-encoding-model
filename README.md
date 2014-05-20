The Semantic-Encoding model is split into two main challenges. Firstly, deriving a set of event categories/topics and secondly, using an encoding of these to correlate signals. In this way the object of the framework is to allow for:

  - The prediction of signals based on distributions over topics
  - The predictions of images based on the signal

In pursuit of this the code contained here is comprised of two main functional pieces. The first is the derivation of topic models, which given a set of documents comprised of some vocabulary, will try to produce a topic distribution over the vocabulary. In this fashion, it would be possible to categorise any document based on it's vocabulary. This is done using the excellent Gensim framework. To see how this can be done take a look at exampleLDAScripts. 

Secondly (and the bulk of the code here) is to predict signal by some supervised training based on topic labels. This is done using the GridRegression class. This class can be instantiated by supplying the following:
  - model - Any object inheriting model from models.py
  - SignalManager - SignalManager object to provide information about the signals
  - nlags - For regression it is useful to consider previous time points when determining the current signal amplitude. The number of previous     points is the number of 'lags'
  - noiseOrders - The Gridregression class also takes care of detrending noise. It will manage a noise as a series of polynomials - the orders     of which are specified here.
  - encoder - Any object inheritying GridRegression.Encoder . This object specifies the column encodings (without lags) to be used for a given     event

The GridRegression class works by training design matrices for each event and works with models in an iterative fashion. That is, the design matrix is never explicitly constructed, but rather models are updated based on a sequence of partial design matrices for each event. This helps to save on memoery. The encoding for a specific event is a vector returned by a GridRegression.Encoder. The GridRegression class will automatically handle the time progression for the lags and portion of the design relating to noise. For encoding based on LDA output the ldaUtils.LDAEncoder is available. 

In addition to the noise orders thare are supplied, GridRegression also adds 60hz sine and cosine waves to deal with mains frequency noise. Note then that the resulting inferred parameters are then split into alpha and beta the design and noise parameters respectively. 


---------------------------------
Requirements Notes:
Check the accompanying docs for particular requirements of each module. Especially not the following:

- signal management utility - https://github.com/andrew0harney/Multi-channel-signal-management-utility
- gensim - (great LDA package http://radimrehurek.com/gensim/)
- nltk - If you wish to perform advanced natural language processing for topic clustering http://www.nltk.org/


---------------------------------
Notes:

  - You should ensure that your numpy installation is configured to use ATLAS/BLAS libraries. These are efficient linear algebra libraries that will severely improve performance. 

  - While some effort was made to keep this code general, it was ultimately written with a specific project in mind. Therefore it is not intended as a general release package, however, I hope it may provide some use and it should still be straightforward to adapt.

- Please also refer to to the accompanying license before using any of this code.
