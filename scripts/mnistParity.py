import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets

class MNISTParityVertical(Dataset):
    
    def __init__(self, k, data_root = "data", transforms = None):
        
        # download and import the original MNIST data
        self.trainset = datasets.MNIST(data_root, train=True, transform=transforms, download=True)
        self.testset = datasets.MNIST(data_root, train=False, transform=transforms, download=True)
        
        # reshape the datasets by given paramter k
        numTrainSet = len(self.trainset.data)
        numTestSet = len(self.testset.data)
        remainderTrain = numTrainSet % k
        remainderTest = numTestSet % k

        self.trainset.data = self.trainset.data[:numTrainSet-remainderTrain].reshape(numTrainSet // k, k * 28, 28)
        self.testset.data = self.testset.data[:numTestSet-remainderTest].reshape(numTestSet // k, k * 28, 28)

        self.trainset.targets = self.trainset.targets[:numTrainSet-remainderTrain].reshape(numTrainSet // k, k).sum(axis=1) % 2
        #self.trainset.targets[self.trainset.targets == 0] = -1
        self.testset.targets = self.testset.targets[:numTestSet-remainderTest].reshape(numTestSet // k, k).sum(axis=1) % 2
        #self.testset.targets[self.testset.targets == 0] = -1 
        
        
    def __len__(self):
        return len(self.trainset.data)
    
    def __getitem__(self,idx):
        return self.trainset.data[idx]
    
    def plotRandomData(self):
        """Plot random data from trainset with label as title"""
        randomIdx = torch.randint(len(self.testset.data), (1,)).item()
        plt.axis("off")
        plt.imshow(self.testset.data[randomIdx].numpy(), cmap = "gray")
        plt.title("Label {}".format(self.testset.targets[randomIdx]))
        
        
class MNISTParityHorizontal(Dataset):
 
	def __init__(self, k, data_root = "data", transforms = None):
    
		# download and import the original MNIST data
		self.trainset = datasets.MNIST(data_root, train=True, transform=transforms, download=True)
		self.testset = datasets.MNIST(data_root, train=False, transform=transforms, download=True)

		self.trainset.data = self.__stackData(self.trainset.data, k)
		self.testset.data = self.__stackData(self.testset.data, k)

		self.trainset.targets = self.__getLabels(self.trainset.targets, k)
		self.testset.targets = self.__getLabels(self.testset.targets, k)
    
	def __stackData(self, dataset, k):
        
		tensorListtoStack = []
		tensorListtoHStack = []

		numSet = len(dataset)
		remainder = numSet % k

		for idx,x in enumerate(dataset[:numSet-remainder]):
		    tensorListtoHStack.append(x)
		    if idx % k == k-1:
		        y = torch.hstack(tensorListtoHStack)
		        tensorListtoStack.append(y)
		        tensorListtoHStack = []
		data = torch.stack(tensorListtoStack)
		return data
    
	def __getLabels(self, targets, k):
		numSet = len(targets)
		remainder = numSet % k
		labels = targets[:numSet-remainder].reshape(numSet // k, k).sum(axis=1) % 2
		#labels[labels == 0] = -1
		return labels
    
	def plotRandomData(self):
		"""Plot random data from trainset with label as title"""
		randomIdx = torch.randint(len(self.testset.data), (1,)).item()
		plt.axis("off")
		plt.imshow(self.testset.data[randomIdx].numpy(), cmap = "gray")
		plt.title("Label {}".format(self.testset.targets[randomIdx]))
		
# This is the best version, other methods reduce the number fo datapoints, by using only existing ones
# whereas this class sample from dataset and it's re-created for each epoch
class MNISTParity:
    def __init__(self, dataset, k = 1, batch_size = 128):

        self.k = k
        # indexes to sample from MNIST data
        left = np.random.permutation(dataset.data.shape[0])
        right = np.random.permutation(dataset.data.shape[0])
        middle = np.random.permutation(dataset.data.shape[0])
        
        # concatenate them to have horizontaly stacked images
        if k == 2:
            self.data =torch.Tensor( np.concatenate(( dataset.data[left],  dataset.data[right]), axis=2)).float()
            self.targets = ((dataset.targets[left] + dataset.targets[right]) %2)
            
        elif k == 3:
            self.data =torch.Tensor( np.concatenate(( dataset.data[left], dataset.data[middle],  dataset.data[right]), axis=2)).float()                   
            self.targets = ((dataset.targets[left] + dataset.targets[middle] + dataset.targets[right]) %2)

        # k = 1    
        else:
            self.data = dataset.data.float()
            self.targets = dataset.targets % 2
            
            
        self.loader = torch.utils.data.DataLoader(TensorDataset(self.data, self.targets), batch_size=batch_size,
                                          shuffle=True)
        
        
    def plotRandomData(self):
        """Plot random data from trainset with label as title"""
        randomIdx = torch.randint(len(self.data), (1,)).item()
        plt.axis("off")
        plt.imshow(self.data[randomIdx].numpy(), cmap = "gray")
        plt.title("Label {}".format(self.targets[randomIdx]))
