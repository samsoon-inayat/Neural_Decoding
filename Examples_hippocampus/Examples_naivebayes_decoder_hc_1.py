
#Import standard packages
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import io
from scipy import stats
import sys
import pickle
import h5py


# If you would prefer to load the '.h5' example file rather than the '.pickle' example file. You need the deepdish package
# import deepdish as dd 

#Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

#Import decoder functions
from Neural_Decoding.decoders import NaiveBayesDecoder



folder1='E:/Users/samsoon.inayat/OneDrive - University of Lethbridge/Data/Neural_Decoding/' #ENTER THE FOLDER THAT YOUR DATA IS IN
folder = 'Z:/homes/brendan.mcallister/2P/ProcessedDataMatlab/'

filename = folder + 'NB_decoding.mat'
arrays = {}
fm = h5py.File(filename)
for k, v in fm.items():
    print(type(v))
#     arrays[k] = np.array(v)


# In[113]:


an = 0
# num_points = 20000
aXs_C_train = np.array(fm[fm['aXs_C_train'][an][0]])
aYs_C_train = np.array(fm[fm['aYs_C_train'][an][0]])

aXs_C_test = np.array(fm[fm['aXs_C_test'][an][0]])
aYs_C_test = np.array(fm[fm['aYs_C_test'][an][0]])

bins_before=0 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0 #How many bins of neural data after the output are used for decoding


N=bins_before+bins_current+bins_after 

#Get training data
X_b_train=aXs_C_train
Y_train=aYs_C_train

#Get testing data
X_b_test=aXs_C_test
Y_test=aYs_C_test

# In[114]:
#Make integer format
X_b_train=X_b_train.astype(int)
X_b_test=X_b_test.astype(int)

#Make y's aligned w/ X's
#e.g. we have to remove the first y if we are using 1 bin before, and have to remove the last y if we are using 1 bin after
if bins_before>0 and bins_after>0:
    X_b_train=X_b_train[bins_before:-bins_after]
    X_b_test=X_b_test[bins_before:-bins_after]
    Y_train=Y_train[bins_before:-bins_after]
    Y_test=Y_test[bins_before:-bins_after]
    

#Declare model

#The parameter "encoding_model" can either be linear or quadratic, although additional encoding models could later be added.

#The parameter "res" is the number of bins used (resolution) for decoding predictions
#So if res=100, we create a 100 x 100 grid going from the minimum to maximum of the output variables (x and y positions)
#The prediction the decoder makes will be a value on that grid 

# model_nb=NaiveBayesDecoder(encoding_model='quadratic',res=100)
# In[115]:
model_nb=NaiveBayesDecoder(encoding_model='quadratic',res = 10)

#Fit model
model_nb.fit(X_b_train,Y_train)
# In[116]:
#Get predictions
Y_test_predicted=model_nb.predict(X_b_test,Y_test)


#Get metric of fit
R2_nb=get_R2(Y_test,Y_test_predicted)
print(R2_nb)

fig = plt.figure()
plt.plot(Y_test[:,0],'b')
plt.plot(Y_test_predicted[:,0],'r')

plt.plot(Y_train[:,1])
plt.plot(X_b_train[:,4])
plt.imshow(X_b_train)

