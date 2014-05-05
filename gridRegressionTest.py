import numpy as np
import pandas as pd
from andyClasses.SignalManager import SignalManager
from andyClasses.GridRegression import GridRegression,gen_designMatrix
import matplotlib.pyplot as plt

data_path = '/home/andy/workspace/ecog/Data/cm/cm'
log_path =  '/home/andy/workspace/ecog/Data/cm/pulse.csv'

grid = SignalManager(data_path, log_file=log_path)
chans = ['C127']
grid.set_wd(channels=chans) 

# Some parameters for the model
polyOrds = [0,1,2,3]
nlags = 2

encoding = {2:np.array([1])}

em = grid.event_matrix()
image = em[em['event.code'] == 2]
isi = em[em['event.code'] == 3]

#Get events
events = pd.concat((image,isi))
events = events.sort(columns='pulse.on')

#Get Training data
ys = grid.wd(channels=chans)
trainset = len(events) * 3 / 4
#Get training and test set
eventsTrain = events[:trainset][['pulse.on', 'pulse.off', 'event.code']]
eventsTest = events[trainset:][['pulse.on', 'pulse.off', 'event.code']]
eventsTrain.columns = [['pulse.on', 'pulse.off', 'label']]
eventsTest.columns = [['pulse.on', 'pulse.off', 'label']]

gridRegression = GridRegression(grid,nlags,polyOrds,encoding=encoding)
coefs = gridRegression.train(eventsTrain, ys[['C127']], encoding)
ypredict = gridRegression.predict(eventsTest)

plt.plot(ypredict)
plt.show()


