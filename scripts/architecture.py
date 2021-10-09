import math
import torch


class MLP(torch.nn.Module):
    def __init__(self, param_k, activation, losstype):
        super().__init__()

        self.losstype = losstype
        self.output_dim = 2 if self.losstype == "Cross Entropy" else 1

        self.activation = activation
        self.flat = torch.nn.Flatten()  # X comes in as a n x 1 x 56 x 28 -> we need n 1568-size vectors
        # (or, a n x 1568 matrix). flatten does this

        if "NTK" in activation:
            self.layerEx = torch.nn.Linear(784 * param_k, 512, bias=False)

        self.layer1 = torch.nn.Linear(28 * 28 * param_k, 512, bias=False)
        self.layer2 = torch.nn.Linear(512, self.output_dim, bias=False)

        if "Gaussian" in activation:
            self.layer1.weight.data.normal_(0, torch.sqrt(torch.tensor(6 / (784*param_k + 512))))

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
        out = torch.nn.functional.softmax(out, dim=1) if self.losstype == "Cross Entropy" else torch.sigmoid(out)
        return out


# TODO: Try to switch to DFA
class MLPManual(torch.nn.Module):
    def __init__(self, param_k, lr, losstype, getWeights=False):
        super().__init__()
        self.input_dim = 28 * 28 * param_k
        self.hidden_dim = 512
        self.flat = torch.nn.Flatten()
        self.losstype = losstype
        self.device_to_run = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.output_dim = 2 if losstype == "Cross Entropy" else 1
        self.learning_rate = lr
        # initialize weights
        if getWeights:
            self.w1, self.w2 = getWeights
            self.w1_grads = torch.empty_like(self.w1)
            self.w2_grads = torch.empty_like(self.w2)
        else:
            self.w1, self.w2 = self.initializeWeights()

    def initializeWeights(self):
        # initialize the weights as pytorch does by default
        # e.g. 784 x 512 (pytorch initializes in this way out x in)
        w1 = torch.empty(self.hidden_dim, self.input_dim).to(self.device_to_run)
        stdv1 = 1. / math.sqrt(w1.size(1))
        w1.uniform_(-stdv1, +stdv1)
        #  e.g. 512 x 1
        w2 = torch.empty(self.output_dim, self.hidden_dim).to(self.device_to_run)
        stdv2 = 1. / math.sqrt(w2.size(1))
        w2.uniform_(-stdv2, +stdv2)
        return w1, w2

    @staticmethod
    def reLUPrime(s):
        s = (s > 0.0) * 1.0
        return s

    # Forward propagation
    def forward(self, X):
        X = self.flat(X)
        # batch_size changes at the end of the epoch from 128 to 96, this spawned a problem in calculations
        # a_k = W_k @ h_{k-1} + b_k, h_k = f(a_k) where h_0 = X and f is the non linearity, a_2 = y^
        a1 = torch.matmul(X, self.w1.t())  # e.g. k=1 --> 128x784 @ 784x512
        # where 128 is batch_size (X.shape[0])
        h1 = torch.nn.functional.relu(a1)       # f is the reLU
        # need to make these variables class attribute to access from `backward` method
        self.a1 = a1
        self.h1 = h1
        a2 = torch.matmul(h1, self.w2.t())
        y_hat = torch.nn.functional.softmax(a2, dim=1) if self.losstype == "Cross Entropy" else torch.sigmoid(a2)
        return y_hat

    # Backward propagation
    def backward(self, X, y, y_hat):
        X = self.flat(X)
        e = y_hat - torch.nn.functional.one_hot(y) if self.losstype == "Cross Entropy" else y_hat - y.reshape(len(y), 1)  # e - 128x1, h1.t - 512,128 for k=1
        # gradients of W2 --> dBCE/dW2 = dBCE/dy^.dy^/da2. da2/dW2 = (y^ - y) h1
        # e - 128x2, h1.t - 512,128 for k=1
        self.w2_grads = torch.matmul(self.h1.t(), e)
        # gradients of W1 --> dBCE/dW1 = dBCE/dh1 . dh1/da1 . da1/dW1
        # where dBCE/dh1 = dBCE/dy^ . dy^/da2 . da2/dh1
        dBCE_da1 = torch.matmul(e, self.w2) * self.reLUPrime(self.a1)  # e - 128x1, w2 - 1,512 , a1 - 128,512
        self.w1_grads = torch.matmul(X.t(), dBCE_da1)  # x.t - 784,128, dBCE_da1 128,512
        # Implement SGD here
        self.w1 -= (self.learning_rate * self.w1_grads.t()) / X.shape[0]
        self.w2 -= (self.learning_rate * self.w2_grads.t()) / X.shape[0]

    def train_manually(self, X, y):
        # Forward propagation
        y_hat = self.forward(X)
        # Backward propagation and gradient descent
        self.backward(X, y, y_hat)
