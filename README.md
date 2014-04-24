Semantic-encoding-model
---------------------------------

Methods for performing decoding/encoding from signals using LDA topics. 
Aims are to:
- Predict signal based on distribution over topics
- Preict (decode) what generated the signal


---------------------------------
Requirements:

- signal management utility - https://github.com/andrew0harney/Multi-channel-signal-management-utility
- numpy
- pandas
- gensim - (great LDA package http://radimrehurek.com/gensim/)
- nltk - If you wish to perform advanced natural language processing for topic clustering http://www.nltk.org/

---------------------------------
Files:
- Grid Regression
- optimiseLDA


---------------------------------
Notes:

While some effort was made to keep this code general, it was ultimately written with a specific project in mind. Therefore it is not intended as a general release package, however, I hope it may provide some use and it should still be straightforward to adapt.


Please also refer to to the accompanying license before using any of this code.
