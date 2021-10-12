# Example Training, Validation and Testing Loop with Random Splits and Epochs.
#
# We train a linear parametrization of the form Hx to approximate outputs y
# according to the squared Euclidean error \| y - Hx \|^2. As in Lab 1,
# Question 3.1 (https://gnn.seas.upenn.edu/labs/lab1/

################################################################################
# Import Standard Libraries 

import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim
import copy

################################################################################
# Import Libraries created for this code

import data as data
import architectures as archit

################################################################################
# Data Generation.

# Parameters of Lab 1, Question 3.1 (https://gnn.seas.upenn.edu/labs/lab1/
N = 100 # input dimension
M = 100 # output dimension
nSamples = 1200 # total number of samples

# Call function data.dataLab1 to generate fake data pairs (x_q, y_q). The data
# x, y are Pytorch tensors
x, y = data.dataLab1(N, M, nSamples) # input and output samples

nTrain = 1000 # number of training samples
nValid = 100 # number of validation samples
nTest = nSamples - nTrain - nValid # number of test samples

# Definition of the training, validation, and test random splits
permutation = np.random.permutation(nSamples) # permutation of the samples
# Train split
xTrain = x[permutation[0:nTrain],:]
yTrain = y[permutation[0:nTrain],:]
# Validation split
xValid = x[permutation[nTrain:nTrain+nValid],:]
yValid = y[permutation[nTrain:nTrain+nValid],:]
# Test split
xTest = x[permutation[-nTest:],:]
yTest = y[permutation[-nTest:],:]

################################################################################
# Specification of learning parametrization.

# Specification of the class where the learning parametriation is defined.
# This is asymbolic call. Not a computation. 
estimator = archit.Parametrization(N, M)

################################################################################
# Training loop initialization

# Parameters used in training loop
epsilon = 0.0001 # optimization step
nEpochs = 250 # total number of epochs (full passes over the training set)
batchSize = 50 # size of each batch
validationInterval = 50 # after how many steps to print training loss

# Specify the optimizer that is used to perform descent updates
# This is asymbolic call. Not a computation. 
optimizer = optim.Adam(estimator.parameters(), lr=epsilon)
#optimizer = optim.SGD(parametrization.parameters(), lr=learningRate)

################################################################################
# Training and validation loop 

costValid = [] # initialize list to store the validation losses and keep track 
               # of the best model

# Determine the number of batches per epoch
if nTrain < batchSize:
    nBatches = 1
    batchSize = [nTrain]
elif nTrain % batchSize != 0:
    nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
    batchSize = [batchSize] * nBatches
    while sum(batchSize) != nTrain:
        batchSize[-1] -= 1
else:
    nBatches = np.int(nTrain/batchSize)
    batchSize = [batchSize] * nBatches
    
# Determine first index of each batch
batchIndex = np.cumsum(batchSize).tolist()
batchIndex = [0] + batchIndex

epoch = 0 # initialize epoch counter    

while epoch < nEpochs:
    
    # Permute training samples
    randomPermutation = np.random.permutation(nTrain) 
    idxEpoch = [int(i) for i in randomPermutation]

    batch = 0 # initialize batch counter
    
    while batch < nBatches:
        
        # Determine batch indices gicen batch count
        thisBatchIndices = idxEpoch[batchIndex[batch]:batchIndex[batch+1]]
        x = xTrain[thisBatchIndices,:] # x components of batch
        y = yTrain[thisBatchIndices,:] # y components of batch
            
        # Reset gradient computation 
        estimator.zero_grad()
    
        # Specify the parametrization that produces estimates
        yHat = estimator.forward(x)
    
        # Specify the loss function
        loss = torch.mean((yHat-y)**2)
    
        # Compute gradients
        loss.backward()
    
        # Update weights with an optimizer step
        optimizer.step()
        
        # Validation
        if  (epoch * nBatches + batch) % validationInterval == 0:
            
            # Obtain the output of the GNN for the validation set
            # without computing gradients
            with torch.no_grad():
                yHatValid = estimator.forward(xValid)
    
            # Specify the loss function
            lossValueValid = torch.mean((yHatValid-yValid)**2)
    
            # Compute training and validation losses and save them
            costValueTrain = loss.item()
            costValueValid = lossValueValid.item()
            costValid += [costValueValid]
            
            # Print training and validation loss  
            if (epoch * nBatches + batch) % validationInterval == 0:
                print("")
                print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
                print("")
            
                print("\t Loss: %6.4f [T]" % (costValueTrain) + " %6.4f [V]" % (
                                costValueValid))
            
            # Save the best model so far 
            if len(costValid) > 1:
                if costValueValid <= min(costValid):
                    bestModel =  copy.deepcopy(estimator)
            else:
                bestModel =  copy.deepcopy(estimator)
        
        # Next batch
        batch+=1
    
    # Next epoch
    epoch+=1
    
print("")

################################################################################
# Testing the trained model

print("Final evaluation results")

# Testing last model
with torch.no_grad():
    yHatTest = estimator.forward(xTest)
lossTestLast = torch.mean((yHatTest-yTest)**2)
costTestLast = lossTestLast.item()

# Testing best model (according to validation)
with torch.no_grad():
    yHatTest = bestModel.forward(xTest)
lossTestBest = torch.mean((yHatTest-yTest)**2)
costTestBest = lossTestBest.item()

# Print test results
print(" Test loss: %6.4f [Best]" % (
                    costTestBest) + " %6.4f [Last]" % (
                    costTestLast))
 

