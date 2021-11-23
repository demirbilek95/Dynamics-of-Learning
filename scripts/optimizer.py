import torch


class Optimizer:
    def __init__(self, method, lr, w1_size, w2_size, update_both, device_to_run):
        self.method = method
        self.lr = lr
        self.update_both = update_both
        # optim matrices
        self.A1 = torch.zeros(w1_size, device=device_to_run) # 784x512
        self.A2 = torch.zeros(w2_size, device=device_to_run) # 512x1
        self.B1 = torch.zeros(w1_size, device=device_to_run)
        self.B2 = torch.zeros(w2_size, device=device_to_run)
        self.eps = 1e-8

    def __updateWeights(self, deltaw1, deltaw2, w1, w2):
        if self.update_both:
            w1 -= deltaw1
        w2 -= deltaw2
        return w1, w2

    def updateParameters(self, t, w1, w2, w1_grads, w2_grads, momentum, nesterov_momentum, weight_decay):
        if self.method == "SGD":
            w1, w2 = self.__SGD(w1, w2, w1_grads, w2_grads, momentum, nesterov_momentum, weight_decay)
        elif self.method == "Adagrad":
            w1, w2 = self.__Adagrad(w1, w2, w1_grads, w2_grads, weight_decay)
        elif self.method == "Adadelta":
            w1, w2 = self.__Adadelta(w1, w2, w1_grads, w2_grads, weight_decay)
        elif self.method == "RMSProp":
            w1, w2 = self.__RMSProp(w1, w2, w1_grads, w2_grads, weight_decay)
        elif self.method == "Adam":
            w1, w2 = self.__Adam(t, w1, w2, w1_grads, w2_grads, weight_decay)
        else:
            raise NotImplementedError

        return w1, w2

    def __SGD(self, w1, w2, w1_grads, w2_grads, momentum=0.9, nesterov_momentum=0.9, weight_decay = None):
        if weight_decay:
            w1_grads += weight_decay * w1
            w2_grads += weight_decay * w2
        if momentum:
            self.A1 = momentum * self.A1 + self.lr * w1_grads
            self.A2 = momentum * self.A2 + self.lr * w2_grads
            w1, w2 = self.__updateWeights(self.A1, self.A2, w1, w2)
        elif nesterov_momentum:
            self.A1 = nesterov_momentum * self.A1 + self.lr * (w1_grads - nesterov_momentum * self.A1)
            self.A2 = nesterov_momentum * self.A2 + self.lr * (w2_grads - nesterov_momentum * self.A2)
            w1, w2 = self.__updateWeights(self.A1, self.A2, w1, w2)
        else:
            deltaw1 = self.lr * w1_grads
            deltaw2 = self.lr * w2_grads
            w1, w2 = self.__updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def __Adagrad(self, w1, w2, w1_grads, w2_grads, weight_decay = None):
        if weight_decay:
            w1_grads += weight_decay * w1
            w2_grads += weight_decay * w2
        self.A1 += torch.square(w1_grads)
        self.A2 += torch.square(w2_grads)
        deltaw1 = self.lr * w1_grads / torch.sqrt(self.A1 + self.eps)
        deltaw2 = self.lr * w2_grads / torch.sqrt(self.A2 + self.eps)
        w1, w2 = self.__updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def __Adadelta(self, w1, w2, w1_grads, w2_grads, weight_decay = None):
        gamma=0.9
        if weight_decay:
            w1_grads += weight_decay * w1
            w2_grads += weight_decay * w2
        self.A1 = gamma * self.A1 + (1-gamma) * torch.square(w1_grads)
        self.A2 = gamma * self.A2 + (1-gamma) * torch.square(w2_grads)

        deltaw1 = torch.sqrt(self.B1 + self.eps) * w1_grads / torch.sqrt(self.A1 + self.eps)
        deltaw2 = torch.sqrt(self.B2 + self.eps) * w2_grads / torch.sqrt(self.A2 + self.eps)
        deltaw1 *= self.lr
        deltaw2 *= self.lr
        w1, w2 = self.__updateWeights(deltaw1, deltaw2, w1, w2)

        self.B1 = gamma * self.B1 + (1-gamma) * torch.square(deltaw1)
        self.B2 = gamma * self.B2 + (1-gamma) * torch.square(deltaw2)
        return w1, w2

    def __RMSProp(self, w1, w2, w1_grads, w2_grads, weight_decay = None):
        gamma=0.9
        if weight_decay:
            w1_grads += weight_decay * w1
            w2_grads += weight_decay * w2
        self.A1 = gamma * self.A1 + (1-gamma) * torch.square(w1_grads)
        self.A2 = gamma * self.A2 + (1-gamma) * torch.square(w2_grads)

        deltaw1 = self.lr * w1_grads / torch.sqrt(self.A1 + self.eps)
        deltaw2 = self.lr * w2_grads / torch.sqrt(self.A2 + self.eps)
        w1, w2 = self.__updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def __Adam(self, t, w1, w2, w1_grads, w2_grads, weight_decay = None):
        beta1, beta2 = 0.9,0.999
        if weight_decay:
            w1_grads += weight_decay * w1
            w2_grads += weight_decay * w2
        self.A1 = beta1 * self.A1 + (1-beta1) * w1_grads
        self.A2 = beta1 * self.A2 + (1-beta1) * w2_grads

        self.B1 = beta2 * self.B1 + (1-beta2) * torch.square(w1_grads)
        self.B2 = beta2 * self.B2 + (1-beta2) * torch.square(w2_grads)
        # bias corrected form
        mt1hat = self.A1 / (1-(beta1 ** (t+1)))
        mt2hat = self.A2 / (1-(beta1 ** (t+1)))

        vt1hat = self.B1 / (1-(beta2 ** (t+1)))
        vt2hat = self.B2 / (1-(beta2 ** (t+1)))

        deltaw1 = self.lr * mt1hat / (torch.sqrt(vt1hat) + self.eps)
        deltaw2 = self.lr * mt2hat / (torch.sqrt(vt2hat) + self.eps)
        w1, w2 = self.__updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2
