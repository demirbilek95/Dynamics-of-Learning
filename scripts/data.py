import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


class MNISTParity:
    def __init__(self, dataset, k=1, batch_size=128):
        # torch.manual_seed(42) # to make data deterministic
        self.k = k
        # indexes to sample from MNIST data
        left = torch.randperm(dataset.data.shape[0])
        right = torch.randperm(dataset.data.shape[0])
        middle = torch.randperm(dataset.data.shape[0])

        # TODO: These parts are added for t-SNE, may need to remove later 
        self.original_target = dataset.targets
        self.left_target = dataset.targets[left]
        self.middle_target = dataset.targets[middle]
        self.right_target = dataset.targets[right]
        
        # concatenate them to have horizontally stacked images
        if k == 2:
            self.data = torch.cat((dataset.data[left],  dataset.data[right]), dim=2)
            self.data = self.__normalize(self.data)
            self.targets = ((dataset.targets[left] + dataset.targets[right]) % 2)
            
        elif k == 3:
            self.data = torch.cat((dataset.data[left], dataset.data[middle],  dataset.data[right]), dim=2)
            self.data = self.__normalize(self.data)
            self.targets = ((dataset.targets[left] + dataset.targets[middle] + dataset.targets[right]) % 2)

        # k = 1    
        else:
            self.data = self.__normalize(dataset.data)
            self.targets = dataset.targets % 2

        self.loader = torch.utils.data.DataLoader(TensorDataset(self.data, self.targets), batch_size=batch_size,
                                                  shuffle=False)

    def __normalize(self, data):
        return (data - 33.3285) / 78.5655

    def plotRandomData(self):
        """Plot random data from trainset with label as title"""
        randomIdx = torch.randint(len(self.data), (1,)).item()
        plt.axis("off")
        plt.imshow(self.data[randomIdx].numpy(), cmap="gray")
        plt.title("Label {}".format(self.targets[randomIdx]))


# Random Data
def cluster_center(p ,k):
    p1 = torch.remainder(p-1, k) + 1
    p2 = torch.div(p-1, k, rounding_mode="trunc") + 1
    Delta = 1/(3*k-1)
    x1 = Delta * (1 + 3*(p1-1)) - 1/2
    x2 = Delta * (1 + 3*(p2-1)) - 1/2
    return x1, x2

def syntheticData(k, n, n_test, sd):
    Delta = 1/(3*k-1) # interclass distance
    A = torch.ones(k**2) # cluster affectation
    A[torch.randperm(k**2)[0:torch.div(k**2,2, rounding_mode="trunc")]] = 0  

    # sample from it
    P = torch.randint(1,k**2+1,(1,n)).reshape(-1) # cluster label
    T = 2*np.pi * torch.rand(n) # shift angle
    R = Delta * torch.rand(n) # shift magnitude
    X = torch.hstack((torch.stack((torch.ones(n), cluster_center(P, k)[0] + R * torch.cos(T), cluster_center(P,k)[1] + R * torch.sin(T)),dim=1), torch.rand(n,sd) -1/2))
    y = A[P-1]

    P_test = torch.randint(1,k**2+1,(1,n_test)).reshape(-1) # cluster label
    T_test = 2*np.pi * torch.rand(n_test) # shift angle
    R_test = Delta * torch.rand(n_test) # shift magnitude
    X_test = torch.hstack((torch.stack((torch.ones(n_test), cluster_center(P_test, k)[0] + R_test * torch.cos(T_test), cluster_center(P_test,k)[1] + R_test * torch.sin(T_test)),dim=1), torch.rand(n_test  ,sd) -1/2))
    y_test = A[P_test-1]
    return TensorDataset(X,y), TensorDataset(X_test,y_test)
