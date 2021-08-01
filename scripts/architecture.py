import torch

class MLP(torch.nn.Module):
    def __init__(self, inputShape0, inputShape1):
        super().__init__()
        self.flat = torch.nn.Flatten() # X comes in as a n x 1 x 56 x 28 -> we need n 1568-size vectors (or, a n x 1568 matrix). flatten does this
        self.layer1 = torch.nn.Linear(inputShape0 * inputShape1, 512)
        self.layer2 = torch.nn.Linear(512, 2) 

    def forward(self, X): 
        out = self.flat(X)
        out = self.layer1(out)
        out = torch.nn.functional.relu(out)
        out = self.layer2(out)
        # out = torch.sigmoid(out)
        return out
