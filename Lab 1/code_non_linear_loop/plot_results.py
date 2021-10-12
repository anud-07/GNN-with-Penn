# Example Ploting Results
#
# We train a linear parametrization of the form Hx to approximate outputs y
# according to the squared Euclidean error \| y - Hx \|^2. As in Lab 1,
# Question 3.1 (https://gnn.seas.upenn.edu/labs/lab1/

################################################################################
# Import Standard Libraries 

import torch; torch.set_default_dtype(torch.float64)
import torch.optim as optim
import matplotlib.pyplot as plt

################################################################################
# Plot Mean Square Error

def plot_MSE(error_train,error_test,nTrain,printInterval,save):
    plt.figure() # Create New Figure
    plt.xlabel('Iterations') # Write Iterations on the X label
    plt.ylabel('MSE') # Write MSE on the Y label
    plt.title('Mean Square Error with respect to iterations for Q=%d'%(nTrain)) # Write the title with the amount of samples used
    plt.semilogy([i*printInterval for i in range(len(error_train))], error_train, '*r') # Plot the Train error 
    plt.semilogy([i*printInterval for i in range(len(error_test))], error_test, '*b') # Plot the Test error
    plt.legend(['Train Set','Test Set']) # Add Legends 
    plt.grid() # Add Grid 
    if save: # Save Figure if wanted
        plt.savefig('MSE_loss_Q_'+str(nTrain)+'.pdf', bbox_inches='tight', dpi=150)
    plt.show()
