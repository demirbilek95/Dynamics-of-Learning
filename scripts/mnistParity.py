import matplotlib.pyplot as plt
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
