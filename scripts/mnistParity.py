import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
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
        self.testset.targets = self.testset.targets[:numTestSet-remainderTest].reshape(numTestSet // k, k).sum(axis=1) % 2
        
    def __len__(self):
        return len(self.trainset.data)
    
    def __getitem__(self,idx):
        return self.trainset.data[idx]
    
    def plotRandomData(self):
        """Plot random data from trainset with label as title"""
        randomIdx = torch.randint(len(self.trainset.data), (1,)).item()
        plt.axis("off")
        plt.imshow(self.trainset.data[randomIdx].numpy(), cmap = "gray")
        plt.title("Label {}".format(self.trainset.targets[randomIdx]))
        
        
class MNISTParityHorizontal(Dataset):
 
	def __init__(self, k, data_root = "datasets", transforms = None):
    
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
		return labels
    
	def plotRandomData(self):
		"""Plot random data from trainset with label as title"""
		randomIdx = torch.randint(len(self.trainset.data), (1,)).item()
		plt.axis("off")
		plt.imshow(self.trainset.data[randomIdx].numpy(), cmap = "gray")
		plt.title("Label {}".format(self.trainset.targets[randomIdx]))
