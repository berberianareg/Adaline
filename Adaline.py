#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Berberian
"""

import numpy as np  
import random as rand
import matplotlib.pyplot as plt

#%% unit step function ========================================================

def stepfunction(act):
    threshold = 0.5                                                             # activation threshold for stepfunction
    act[act >= threshold] = 1                                                   # if the activation is greater than or equal to the threshold, give 1 
    act[act < threshold] = 0                                                    # if the activation is less than the threshold, give 0 
    
    return act

#%% learning function =========================================================

def learning(inputPattern, targetPattern, alpha, minLearnError):
    [nbProto, dim] = np.shape(inputPattern)                                     # find the number of prototypes and their corresponding dimension
    mseList = []                                                                # creation of empty list for storing mean squared errors
    seList = np.ones(nbProto)                                                   # place holder array for squared errors
    w = np.zeros([nbProto, dim])                                                # initialization of weights
    iteration = 0                                                               # initialization of iteration
    
    while np.mean(seList) > minLearnError:                                          
        samples = rand.sample(range(nbProto), nbProto)                          # generation of a list of random samples
        isample = 0                                                             # initialization of sample index
        
        while isample < len(samples):                                               
            inp = np.transpose([inputPattern[samples[isample]]])                # random selection of the input pattern
            target = np.transpose([targetPattern[samples[isample]]])            # random selection of the corresponding target pattern
            obtained = np.dot(w, inp)                                           # computation of the activation
            error = target - obtained                                           # computation of the error
            seList[isample] = np.dot(np.transpose(error),(error))               # computation of the squared error
            w += alpha * error * np.transpose(inp)                              # update of the weights
            isample += 1                                                        # isample increment
        
        mseList.append(np.mean(seList))                                         # computation of the mean of squared error 
        iteration +=1                                                           # iteration increment
    
    return [w, mseList]

#%% recall function ===========================================================

def recall(inputPattern, targetPattern, noise, nbReps, w):
    [nbProto, dim] = np.shape(inputPattern)                                     # find the number of prototypes and their corresponding dimension
    sePerformance = np.zeros(nbProto)                                           # preallocate squared error of performance
    recallPerformance = []                                                      # creation of an empty list for storing recall performance
    
    for inoise in range(len(noise)):
        msePerformance = []                                                     # creation of an empty list for storing the mean squared error
        
        for irep in range(nbReps):
            noiseVect = np.transpose([np.random.uniform(0, noise[inoise],dim)]) # specification of uniform noise level
            samples = rand.sample(range(nbProto), nbProto)                      # generation of a list of random samples
            isample = 0                                                         # initialization of sample index
            
            while isample < len(samples):                                           
                xt = np.transpose([inputPattern[samples[isample]]]) + noiseVect # addition of noise to randomly selected input pattern
                xt = xt/max(xt)                                                 # rescaling of noisy input between 0 and 1
                ot = np.dot(w, xt)                                              # computation of the activation
                ot = stepfunction(ot)                                           # application of the step function
                t = np.transpose([targetPattern[samples[isample]]])             # random selection of the corresponding target pattern
                error = t - ot                                                  # computation of the error
                sePerformance[isample] = np.dot(np.transpose(error), (error))   # computation of the squared error 
                isample += 1                                                    # isample increment
            
            sePerformance = np.double(sePerformance < 1)                        # specification of squared errors less than 1
            msePerformance.append(np.mean(sePerformance))                       # computation of the mean squared error 
        
        recallPerformance.append(np.mean(msePerformance) * 100)                 # computation of recall performance in percentage
    
    return recallPerformance

#%% generation of input and target patterns ===================================

inputPattern = np.array([[1,0,0,1,0,0,1,0,0],[1,1,1,0,0,0,0,0,0],
                         [1,0,0,0,1,0,0,0,1]])                                  # input patterns
targetPattern = np.array([[1,0,0],[0,1,0],[0,0,1]])                             # target patterns

#%% learning phase ============================================================

alpha = 0.01                                                                    # learning rate
minLearnError = 0.001                                                           # minimum learning error (tolerance)

[w, mseList] = learning(inputPattern, targetPattern, alpha, minLearnError)      # call learning function

#%% recall phase ==============================================================

noise = np.arange(0, 5, 0.1)                                                    # specification of the range of noise level
nbReps = 1000                                                                   # nb of repetitions

recallPerformance = recall(inputPattern, targetPattern, noise, nbReps, w)       # call recall function

#%% plotting input and target patterns ========================================

nbProto = np.size(inputPattern, 0)                                              # finding the number of prototypes
fig, ax = plt.subplots(figsize=(8,6))                                           # generate a figure
for isample in range(nbProto):
    plt.subplot(2, nbProto, isample + 1)                                        # introduce a subplot
    inp = np.reshape(-inputPattern[isample,:],(nbProto, nbProto))               # select the input pattern
    plt.imshow(inp, cmap = 'gray')                                              # plot 
    plt.tick_params(axis='both', which='both', 
                    bottom=False, top=False, labelbottom=False, right=False, 
                    left=False, labelleft=False)                                # specify tick parameters
    if isample == 1:
        plt.title('Inputs',fontsize=15)                                         # specify title
    plt.subplot(2, nbProto, isample + nbProto + 1)                              # introduce subplot
    tar = np.transpose([-targetPattern[isample,:]])                             # select the target pattern
    plt.imshow(tar, cmap = 'gray')                                              # plot
    plt.tick_params(axis='both', which='both', 
                    bottom=False, top=False, labelbottom=False, right=False, 
                    left=False, labelleft=False)                                # specify tick parameters 
    if isample == 1:
        plt.title('Targets',fontsize=15)                                        # specify title
        
#%% plotting mse during learning ==============================================

fig, ax = plt.subplots(figsize=(8,5))                                           # generate a figure
plt.plot(mseList, 'k', linewidth = 4)                                           # plot
plt.xlabel("Learning iteration", fontsize=15)                                   # specify xlabel
plt.ylabel("Mean Squared Error",fontsize=15)                                    # specify ylabel
plt.title('Learning',fontsize=15)                                               # specify title
plt.xticks(fontsize = 15)                                                       # specify xticks
plt.yticks(fontsize = 15)                                                       # specify yticks

#%% plotting recall performance as a function of noise level ==================

fig, ax = plt.subplots(figsize=(8,5))                                           # generate a figure
plt.plot(noise, recallPerformance, 'k', linewidth=4)                            # plot
plt.xlabel("Noise level", fontsize=15)                                          # specify xlabel
plt.ylabel("Recall performance (%)",fontsize=15)                                # specify ylabel
plt.title('Recall',fontsize=15)                                                 # specify title
plt.xticks(fontsize = 15)                                                       # specify xticks 
plt.yticks(fontsize = 15)                                                       # specify yticks 
plt.ylim(0, 101)                                                                # specify ylim
 