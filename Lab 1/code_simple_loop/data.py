# Functions for generating fake data and selecting batches
# As in Lab 1, Questions 1.2 - 1.4 (https://gnn.seas.upenn.edu/labs/lab1/

import numpy as np
import torch

def dataLab1(N, M, nSamples):

    # Generate nSamples input vectors x of dimension N (Question 1.3).    
    x = np.random.multivariate_normal(np.zeros(N), 0.5*np.eye(N), nSamples)

    # Generate nSamples noise vectors w of dimension N (Question 1.3).
    w = np.random.multivariate_normal(np.zeros(M), 0.5*np.eye(M), nSamples)

    # Generate matrix A of dimensions M by N with Bernoulli entries
    # (Qeustion 1.2)
    A = np.random.binomial(n=1, p=1/M, size=(N,M))
    A = A.squeeze()

    # Generate output pairs as per model (Question 1.4)
    y = np.sign(np.matmul(x,A)+w)

    # Data is converted to torch tensors, which are different from np matrices.
    # Just a technicality of Pytorch operation
    x = torch.tensor(x)
    y = torch.tensor(y)

    # Function output 
    return x, y

def getBatch(xTrain, yTrain, batchSize):
    
    nTrain = xTrain.shape[0]
    
    # Select a random batch of samples (xTrainBatch, yTrainBatch).
    # This is done with indexes drawn from a random permutation 
    randomPermutation = np.random.permutation(nTrain) # generate permutation 
    idx = [int(i) for i in randomPermutation] # write permutation in index array
    thisBatchIndices = idx[0:batchSize] # select batchSize entries
    xTrainBatch = xTrain[thisBatchIndices,:] # x components of batch
    yTrainBatch = yTrain[thisBatchIndices,:] # y components of batch
    
    return xTrainBatch, yTrainBatch