import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#This script gives a colourmesh of the perplexity of each LDA model  generated in a grid search
#This may be a useful evalutation metric

#Vocab and topic number sizes to use in comparison (suitable LDA models must have been made for each)
nwords = np.arange(200,900,100)
ntopics = np.arange(10,32,2)

perplexities_path = #
perplexity = np.load(open(perplexities_path,'r'))

plt.pcolormesh(nwords,ntopics,perplexity)
plt.xlabel('Vocabulary size')
plt.ylabel('Number of topics')
plt.title('Perplexity Matrix')
plt.colorbar()
plt.show()
