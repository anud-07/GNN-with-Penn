# Example Training Loop.
#
# We train a linear parametrization of the form Hx to approximate outputs y
# according to the squared Euclidean error \| y - Hx \|^2. As in Lab 1,
# Question 3.1 (https://gnn.seas.upenn.edu/labs/lab1/

################################################################################
# Import Standard Libraries 

import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim

################################################################################
# Import Libraries created for this code

import data as data
import architectures as archit
import plot_results as pltr

################################################################################
# Data Generation.

# Parameters of Lab 1, Question 3.1 (https://gnn.seas.upenn.edu/labs/lab1/
N = 100 # input dimension
M = 100 # output dimension
nTrain = 50 # number of samples (Q)
nTest = 100 # number of samples of test set
H = 100 # hidden units

# Call function data.dataLab1 to generate fake data pairs (x_q, y_q). The data
# xTrain, yTrain are two Pytorch tensors
xTrain, yTrain = data.dataLab1(N, M, nTrain) 
xTest, yTest = data.dataLab1(N, M, nTrain) 

################################################################################
# Specification of learning parametrization.

# Instantiation of the estimator object, which belongs to the
# Parametrization class
estimator = archit.TwoLayerNN(N, M, H)

################################################################################
# Training loop initialization

# Parameters used in training loop
epsilon = 0.001 # optimization step
nTrainingSteps = 3000 # total number of training steps
batchSize = 50 # size of each batch
printInterval = 50 # after how many steps to print training loss

# Specify the optimizer that is used to perform descent updates. This is an
# instantiation of an optimizer object from the optim.Adam class.
# Alternatively, we can use the optim.SGD class
optimizer = optim.Adam(estimator.parameters(), lr=epsilon)
#optimizer = optim.SGD(parametrization.parameters(), lr=learningRate)

################################################################################
# Training loop
error_train=[] # list where error in train set will be stored
error_test=[] # list where error in test set will be stored
step = 0 # initialize step counter

while step < nTrainingSteps:
    
    # Get input and output samples in this batch
    x, y = data.getBatch(xTrain, yTrain, batchSize)
        
    # Reset gradient computation 
    estimator.zero_grad()

    # Specify the parametrization that produces estimates
    yHat = estimator.forward(x)

    # Specify the loss function 
    loss = torch.mean(( yHat - y )**2)

    # Compute gradients
    loss.backward()

    # Update weights with an optimizer step
    optimizer.step()
    
    # Print loss value every so often to evaluate progress  
    if step % printInterval == 0:     

        # Compute loss value
        cost = loss.item()
        
        # Evaluate the function in the Test set
        yHat_test = estimator.forward(xTest)

        # Compute the loss function for the Test set
        loss_test = torch.mean((yHat_test - yTest) ** 2)
        
        # Print
        print("")
        print("    (Step: %3d)" % (step+1),end = ' ')
        print("")        
        print("\t Loss: %6.4f [Train]" % (cost))
        print("\t Loss: %6.4f [Test]" % (loss_test))

        # Add values of error to the list
        error_train = error_train + [cost]
        error_test = error_test + [loss_test]
    # Next Step
    step+=1
    
print("")
 
pltr.plot_MSE(error_train,error_test,nTrain,printInterval,save=True)
