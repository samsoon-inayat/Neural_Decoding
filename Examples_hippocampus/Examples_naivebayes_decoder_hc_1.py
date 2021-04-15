
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

filename = folder + 'NB_decoding_C_Place.mat'
arrays = {}
fm = h5py.File(filename)
filename = folder + 'NB_decoding_A_Place.mat'
fm_A = h5py.File(filename)
for k, v in fm.items():
    print(type(v))
#     arrays[k] = np.array(v)


# In[113]:


an = 2
cn = 0
# num_points = 20000
aXs_C = fm[fm['mean_rasters_T'][cn][an]]
aXs_C1 = np.array(fm[fm['mean_rasters_T'][cn][an]])

# for ii in range(0,aXs_C1.shape[1]):
#     aXs_C1[:,ii] = aXs_C1[:,ii]/4

# aYs_C = 
sz = aXs_C1.shape[0]
aYs_C1p = np.array(fm[fm['xs'][cn][an]])[:sz,0]
aYs_C1 = np.zeros([aYs_C1p.shape[0],2])
aYs_C1[:,0] = aYs_C1p
aYs_C1[:,1] = aYs_C1p
plt.figure(figsize=(8, 4))
plt.plot(aYs_C1p,aXs_C1[:,1])
neural_data = aXs_C1
pos_binned = aYs_C1

plt.imshow(neural_data)

bins_before=0 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=0 #How many bins of neural data after the output are used for decoding



#Remove neurons with too few spikes in HC dataset
nd_sum=np.nansum(neural_data,axis=0) #Total number of spikes of each neuron
rmv_nrn=np.where(nd_sum<100) #Find neurons who have less than 100 spikes total
neural_data=np.delete(neural_data,rmv_nrn,1) #Remove those neurons
X=neural_data


#Set decoding output
y=pos_binned


#Number of bins to sum spikes over
N=bins_before+bins_current+bins_after 

#Remove time bins with no output (y value)
rmv_time=np.where(np.isnan(y[:,0]))# | np.isnan(y[:,1]))
X=np.delete(X,rmv_time,0)
y=np.delete(y,rmv_time,0)


#Set what part of data should be part of the training/testing/validation sets

training_range=[0, 0.5]
valid_range=[0.5,0.65]
testing_range=[0.65, 0.8]


#Number of examples after taking into account bins removed for lag alignment
num_examples=X.shape[0]

#Note that each range has a buffer of"bins_before" bins at the beginning, and "bins_after" bins at the end
#This makes it so that the different sets don't include overlapping neural data
training_set=np.arange(np.int64(np.round(training_range[0]*num_examples))+bins_before,np.int64(np.round(training_range[1]*num_examples))-bins_after)
testing_set=np.arange(np.int64(np.round(testing_range[0]*num_examples))+bins_before,np.int64(np.round(testing_range[1]*num_examples))-bins_after)
valid_set=np.arange(np.int64(np.round(valid_range[0]*num_examples))+bins_before,np.int64(np.round(valid_range[1]*num_examples))-bins_after)

#Get training data
X_train=X[training_set,:]
y_train=y[training_set]

#Get testing data
X_test=X[testing_set,:]
y_test=y[testing_set]

#Get validation data
X_valid=X[valid_set,:]
y_valid=y[valid_set]


#Initialize matrices for neural data in Naive bayes format
num_nrns=X_train.shape[1]
X_b_train=np.empty([X_train.shape[0]-N+1,num_nrns])
X_b_valid=np.empty([X_valid.shape[0]-N+1,num_nrns])
X_b_test=np.empty([X_test.shape[0]-N+1,num_nrns])

#Below assumes that bins_current=1 (otherwise alignment will be off by 1 between the spikes and outputs)

#For all neurons, within all the bins being used, get the total number of spikes (sum across all those bins)
#Do this for the training/validation/testing sets
for i in range(num_nrns):
    X_b_train[:,i]=N*np.convolve(X_train[:,i], np.ones((N,))/N, mode='valid') #Convolving w/ ones is a sum across those N bins
    X_b_valid[:,i]=N*np.convolve(X_valid[:,i], np.ones((N,))/N, mode='valid')
    X_b_test[:,i]=N*np.convolve(X_test[:,i], np.ones((N,))/N, mode='valid')

#Make integer format
X_b_train=X_b_train.astype(int)
X_b_valid=X_b_valid.astype(int)
X_b_test=X_b_test.astype(int)

#Make y's aligned w/ X's
#e.g. we have to remove the first y if we are using 1 bin before, and have to remove the last y if we are using 1 bin after
if bins_before>0 and bins_after>0:
    y_train=y_train[bins_before:-bins_after]
    y_valid=y_valid[bins_before:-bins_after]
    y_test=y_test[bins_before:-bins_after]
    
if bins_before>0 and bins_after==0:
    y_train=y_train[bins_before:]
    y_valid=y_valid[bins_before:]
    y_test=y_test[bins_before:]


#Declare model

#The parameter "encoding_model" can either be linear or quadratic, although additional encoding models could later be added.

#The parameter "res" is the number of bins used (resolution) for decoding predictions
#So if res=100, we create a 100 x 100 grid going from the minimum to maximum of the output variables (x and y positions)
#The prediction the decoder makes will be a value on that grid 

model_nb=NaiveBayesDecoder(encoding_model='quadratic',res=5)

model_nb=NaiveBayesDecoder(encoding_model='quadratic')

#Fit model
model_nb.fit(X_b_train,y_train)


#Get predictions
y_train_predicted=model_nb.predict(X_b_train,y_train)

#Get predictions
y_valid_predicted=model_nb.predict(X_b_valid,y_valid)


#Get metric of fit
R2_nb=get_R2(y_valid,y_valid_predicted)
print(R2_nb)

fig = plt.figure()
plt.plot(y_valid,'b')
plt.plot(y_valid_predicted,'r')





