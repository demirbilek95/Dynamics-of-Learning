import math

import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self, k, activation):
        super().__init__()
        self.k = k
        self.activation = activation
        self.flat = torch.nn.Flatten()  # X comes in as a n x 1 x 56 x 28 -> we need n 1568-size vectors
        # (or, a n x 1568 matrix). flatten does this

        if "NTK" in activation:
            self.layerEx = torch.nn.Linear(784*k, 512)

        self.layer1 = torch.nn.Linear(784 * k, 512)
        self.layer2 = torch.nn.Linear(512, 1) 

        if "Gaussian" in activation:
            # TODO: Use torch.sqrt instead of numpy
            self.layer1.weight.data.normal_(0, np.sqrt(6 / float(784*k + 512)))

    def forward(self, X): 
        out = self.flat(X)
        if "linear" in self.activation:
            out = self.layer1(out)
        elif "NTK" in self.activation:
            out = torch.nn.functional.glu(torch.cat((self.layer1(out), self.layerEx(out)), 1))
        else:
            out = self.layer1(out)
            out = torch.nn.functional.relu(out)

        out = self.layer2(out)
        return out

# TODO: Try to switch to DFA


class MLPManual(torch.nn.Module):
    def __init__(self, param_k, device_to_run, size_batch, losstype="Cross Entropy"):
        super().__init__()

        self.batch_size = size_batch
        self.input_dim = 28 * 28 * param_k
        self.hidden_dim = 512
        self.losstype = losstype
        if losstype == "Cross Entropy":
            self.output_dim = 2
        else:                           # BCE case
            self.output_dim = 1
        self.learning_rate = 0.001
        self.flat = torch.nn.Flatten()  # when input comes as 28x28, this'll convert to 784
        # WEIGHTS
        # initialize the weights as pytorch does by default --> IT DIVERGES and perform worse (90%) for k=1
        # e.g. 784 x 512
        self.w1 = torch.empty(self.input_dim, self.hidden_dim).to(device_to_run)
        stdv1 = 1. / math.sqrt(self.w1.size(1))
        self.w1.uniform_(-stdv1, +stdv1)
        self.w1_grads = torch.empty_like(self.w1)
        #  e.g. 512 x 1
        self.w2 = torch.empty(self.hidden_dim, self.output_dim).to(device_to_run)
        stdv2 = 1. / math.sqrt(self.w2.size(1))
        self.w2.uniform_(-stdv2, +stdv2)
        self.w2_grads = torch.empty_like(self.w2)

        # BIASES
        self.b1 = torch.empty(size_batch, self.hidden_dim).to(device_to_run)
        self.b1.uniform_(-stdv1, stdv1)
        self.b1_grads = torch.empty_like(self.b1)
        self.b2 = torch.empty(size_batch, self.output_dim).to(device_to_run)
        self.b2.uniform_(-stdv1, stdv1)
        self.b2_grads = torch.empty_like(self.b2)

    @staticmethod
    def softmax(x):
        maxes = torch.max(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-maxes)
        x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
        return x_exp/x_exp_sum

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + torch.exp(-s))

    @staticmethod
    def reLU(s):
        s[s < 0] = 0
        return s.float()

    @staticmethod
    def reLUPrime(s):
        s[s < 0] = 0
        s[s > 0] = 1
        return s.float()

    # Forward propagation
    def forward(self, X):
        X = self.flat(X)
        # batch_size changes at the end of the epoch from 128 to 96, this spawned a problem in calculations
        dynamic_batch_size = X.shape[0]
        # a_k = W_k @ h_{k-1} + b_k, h_k = f(a_k) where h_0 = X and f is the non linearity, a_2 = y^
        a1 = torch.matmul(X, self.w1) + self.b1[:dynamic_batch_size, :]  # e.g. k=1 --> 128x784 @ 784x512 + 128x512
        # where 128 is batch_size (X.shape[1])
        h1 = self.reLU(a1)       # f is the reLU
        # need to make these variables class attribute to access from `backward` method
        self.a1 = a1
        self.h1 = h1
        a2 = torch.matmul(h1, self.w2) + self.b2[:dynamic_batch_size, :]

        if self.losstype == "Cross Entropy":
            y_hat = torch.nn.functional.softmax(a2, dim=1)
        else:
            y_hat = self.sigmoid(a2)

        return y_hat  # some loss functions handle output layer non-linearity

    # Backward propagation
    def backward(self, X, y, y_hat):
        X = self.flat(X)
        dynamic_batch_size = X.shape[0]
        # gradients of W2 --> dBCE/dW2 = dE/dy^.dy^/da2. da2/dW2 = (y^ - y) h1
        if self.losstype == "Cross Entropy":
            e = y_hat - torch.nn.functional.one_hot(y)  # e - 128x2, h1.t - 512,128 for k=1
        else:
            e = y_hat - y.reshape(len(y), 1)  # e - 128x1, h1.t - 512,128 for k=1

        self.w2_grads = torch.matmul(self.h1.t(), e)
        # gradients of W1 --> dBCE/dW1 = dE/dh1 . dh1/da1 . da1/dW1
        # where dE/dh1 = dE/dy^ . dy^/da2 . da2/dh1
        dBCE_da1 = torch.matmul(e, self.w2.t()) * self.reLUPrime(self.a1)  # e - 128x1, w2.t - 1,512 , a1 - 128,512
        self.w1_grads = torch.matmul(X.t(), dBCE_da1)  # x.t - 784,128, dBCE_da1 128,512
        # gradients of b2 --> dBCE/db2 = dBCE/dy^. dy^/da2. da2/db2 = (y^-y)*1
        self.b2_grads = e[:dynamic_batch_size, :]
        # gradients of b1 --> dBCE/db1 = dBCE/dh1. dh1/da1. da1/db1
        # where dBCE/dh1 = dBCE/dy^ . dy^/da2 . da2/dh1
        self.b1_grads = dBCE_da1[:dynamic_batch_size, :]

        # Implement SGD here
        self.w1 -= self.learning_rate * self.w1_grads
        self.w2 -= self.learning_rate * self.w2_grads
        self.b1[:dynamic_batch_size, :] -= self.learning_rate * self.b1_grads
        self.b2[:dynamic_batch_size, :] -= self.learning_rate * self.b2_grads

    def train_manually(self, X, y):
        # Forward propagation
        y_hat = self.forward(X)
        # Backward propagation and gradient descent
        self.backward(X, y, y_hat)
