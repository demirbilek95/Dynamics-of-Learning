import math
import torch
from .optimizer import Optimizer


class MLP(torch.nn.Module):
    def __init__(self, param_k, activation, losstype):
        super().__init__()

        self.output_dim = 2 if losstype == "Cross Entropy" else 1
        self.losstype = losstype
        self.activation = activation
        self.flat = torch.nn.Flatten()  # X comes in as a n x 1 x 28 x 28 -> we need n 784-size vectors
        # (or, a n x 784 matrix). flatten does this

        if "NTK" in activation:
            self.layerEx = torch.nn.Linear(28 * 28 * param_k, 512, bias=False)

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


class MLPManual(torch.nn.Module):
    def __init__(self, input_dim, lr, losstype, train_method, B_initialization, optim, device, measure_alignment, update_both=True, getWeights=False):
        super().__init__()
        self.device_to_run = device
        self.input_dim = input_dim
        self.hidden_dim = 512
        self.output_dim = 2 if losstype == "Cross Entropy" else 1
        self.flat = torch.nn.Flatten()
        self.losstype = losstype
        self.train_method = train_method
        self.B_initialization = B_initialization
        self.measure_alignment = measure_alignment
        self.optim = optim
        self.learning_rate = lr
        self.update_both = update_both

        # initialize weights and gradients
        if getWeights:
            self.w1, self.w2 = getWeights
        else:
            self.w1, self.w2 = self.initializeWeights()

        self.w1_grads = torch.empty_like(self.w1)
        self.w2_grads = torch.empty_like(self.w2)

        if self.measure_alignment:
            self.w1_BP , self.w2_BP = self.w1.clone(), self.w2.clone()
            self.w1_grads_BP, self.w2_grads_BP =  self.w1_grads.clone(), self.w2_grads.clone()
        
        self.B = self.initializeB(self.B_initialization) if train_method == "DFA" else None
        self.optimizer = Optimizer(self.optim, self.learning_rate, self.w1.size(), self.w2.size(), self.update_both, self.device_to_run)

    def initializeWeights(self):
        # initialize the weights as pytorch does by default
        # e.g. 784 x 512 (pytorch initializes in this way out x in)
        w1 = torch.empty(self.input_dim, self.hidden_dim).to(self.device_to_run)
        stdv1 = 1. / math.sqrt(w1.size(0))
        w1.uniform_(-stdv1, +stdv1)
        #  e.g. 512 x 1
        w2 = torch.empty(self.hidden_dim, self.output_dim).to(self.device_to_run)
        stdv2 = 1. / math.sqrt(w2.size(0))
        w2.uniform_(-stdv2, +stdv2)
        return w1, w2

    def initializeB(self, initializaiton_method):
        #torch.manual_seed(42)
        stdv2 = 1. / math.sqrt(self.w2.size(0))
        if initializaiton_method == "standard uniform":
            B = torch.empty_like(self.w2.t())
            B.uniform_()
        elif initializaiton_method == "uniform":
            # Initialize the random matrix similar to w2
            B = torch.empty_like(self.w2.t())
            B.uniform_(-stdv2, stdv2)
        elif initializaiton_method == "standard gaussian":
            # N(0,1)
            B = torch.randn(self.w2.t().size()).to(self.device_to_run)
        elif initializaiton_method == "gaussian":
            # N(mean = stdv2, sigma = stdv2)
            B = torch.normal(stdv2, stdv2, size=self.w2.t().shape).to(self.device_to_run)
        else:
            raise NotImplementedError
        return B

    def setWeightsGradientsDefault(self):
        self.w1, self.w2 = self.initializeWeights()
        self.w1_grads = torch.empty_like(self.w1)
        self.w2_grads = torch.empty_like(self.w2)

    @staticmethod
    def reLUPrime(s):
        s = (s > 0.0) * 1.0
        return s

    # Forward propagation
    def forward(self, X, w1, w2):
        X = self.flat(X)
        # batch_size changes at the end of the epoch from 128 to 96, this spawned a problem in calculations
        # a_k = W_k @ h_{k-1} + b_k, h_k = f(a_k) where h_0 = X and f is the non linearity, a_2 = y^
        a1 = torch.matmul(X, w1)  # e.g. k=1 --> 128x784 @ 784x512
        # where 128 is batch_size (X.shape[0])
        h1 = torch.nn.functional.relu(a1)       # f is the reLU
        a2 = torch.matmul(h1, w2)
        y_hat = torch.nn.functional.softmax(a2, dim=1) if self.losstype == "Cross Entropy" else torch.sigmoid(a2)
        return y_hat, a1, h1

    # Backward propagation
    def backward(self, X, y, y_hat, h1, a1, w2, B, train_method):
        X = self.flat(X)
        self.e = y_hat - torch.nn.functional.one_hot(y) if self.losstype == "Cross Entropy" else y_hat - y.reshape(len(y), 1)  # e - 128x1, h1.t - 512,128 for k=1
        # gradients of W2 --> dBCE/dW2 = dBCE/dy^.dy^/da2. da2/dW2 = (y^ - y) h1
        # e - 128x2, h1.t - 512,128 for k=1
        w2_grads = torch.matmul(h1.t(), self.e)
        w2_grads /= X.shape[0]
        # gradients of W1 --> dBCE/dW1 = dBCE/dh1 . dh1/da1 . da1/dW1
        # where dBCE/dh1 = dBCE/dy^ . dy^/da2 . da2/dh1

        if train_method == "DFA": 
            self.dBCE_da1 = torch.matmul(self.e, B) * self.reLUPrime(a1)    
        else:  # BP
            self.dBCE_da1 = torch.matmul(self.e, w2.t()) * self.reLUPrime(a1)  # e - 128x1, w2 - 512,1 , a1 - 128,512

        w1_grads = torch.matmul(X.t(), self.dBCE_da1)  # x.t - 784,128, dBCE_da1 128,512   

        # w1_grads = torch.matmul(X.t(), dBCE_da1)  # x.t - 784,128, dBCE_da1 128,512
        w1_grads /= X.shape[0]
        return w1_grads, w2_grads

    def train_manually(self, X, y, t, momentum, nesterov_momentum, weight_decay, train_method, measure_alignment):
        # Forward propagation
        y_hat, a1, h1 = self.forward(X, self.w1, self.w2)
        # Backward propagation and gradient descent
        self.w1_grads, self.w2_grads =  self.backward(X, y, y_hat, h1, a1, self.w2, self.B, train_method)
        self.w1, self.w2 = self.optimizer.updateParameters(t, self.w1, self.w2, self.w1_grads, self.w2_grads, momentum, nesterov_momentum, weight_decay)

        if measure_alignment:
            self.w1_grads_BP, self.w2_grads_BP = self.backward(X, y, y_hat, h1, a1, self.w2, None, "BP")

    def measureSimilarity(self):
        cos = torch.nn.CosineSimilarity()
        similarity_w2B = cos(self.B, self.w2.t()).mean().item()
        similarity_w1_grads = cos(self.w1_grads, self.w1_grads_BP ).mean().item()
        return similarity_w2B, similarity_w1_grads
