import torch


class Optimizer:
    def __init__(self, method, lr, w1_size, w2_size):
        self.device_to_run = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.method = method
        self.lr = lr
        # optim matrices
        self.A1 = torch.zeros(w1_size, device=self.device_to_run) # 784x512
        self.A2 = torch.zeros(w2_size, device=self.device_to_run) # 512x1
        self.B1 = torch.zeros(w1_size, device=self.device_to_run)
        self.B2 = torch.zeros(w2_size, device=self.device_to_run)

        print("Training with {}".format(method))

    def updateWeights(self, deltaw1, deltaw2, w1, w2):
        w1 -= deltaw1
        w2 -= deltaw2
        return w1, w2

    def updateParameters(self, t, momentum, nesterov_momentum, w1, w2, w1_grads, w2_grads):
        if self.method == "SGD":
            w1, w2 = self.SGD(momentum, nesterov_momentum, w1, w2, w1_grads, w2_grads)
        elif self.method == "Adagrad":
            w1, w2 = self.Adagrad(w1, w2, w1_grads, w2_grads)
        elif self.method == "Adadelta":
            w1, w2 = self.Adadelta(w1, w2, w1_grads, w2_grads)
        elif self.method == "RMSProp":
            w1, w2 = self.RMSProp(w1, w2, w1_grads, w2_grads)
        elif self.method == "Adam":
            w1, w2 = self.Adam(t, w1, w2, w1_grads, w2_grads)
        else:
            raise NotImplementedError

        return w1, w2

    def SGD(self, momentum, nesterov_momentum, w1, w2, w1_grads, w2_grads):
        if momentum:
            self.A1 = 0.9 * self.A1 + self.lr * w1_grads
            self.A2 = 0.9 * self.A2 + self.lr * w2_grads
            w1, w2 = self.updateWeights(self.A1, self.A2, w1, w2)
        elif nesterov_momentum:
            self.A1 = 0.9 * self.A1 + self.lr * (w1_grads - 0.9 * self.A1)
            self.A2 = 0.9 * self.A2 + self.lr * (w2_grads - 0.9 * self.A2)
            w1, w2 = self.updateWeights(self.A1, self.A2, w1, w2)
        else:
            deltaw1 = self.lr * w1_grads
            deltaw2 = self.lr * w2_grads
            w1, w2 = self.updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def Adagrad(self, w1, w2, w1_grads, w2_grads):
        eps = 1e-8
        self.A1 += torch.square(w1_grads)
        self.A2 += torch.square(w2_grads)
        deltaw1 = self.lr * w1_grads / torch.sqrt(self.A1 + eps)
        deltaw2 = self.lr * w2_grads / torch.sqrt(self.A2 + eps)
        w1, w2 = self.updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def Adadelta(self, w1, w2, w1_grads, w2_grads):
        eps = 1e-8
        self.A1 = 0.9 * self.A1 + 0.1 * torch.square(w1_grads)
        self.A2 = 0.9 * self.A2 + 0.1 * torch.square(w2_grads)

        deltaw1 = torch.sqrt(self.B1 + eps) * w1_grads / torch.sqrt(self.A1 + eps)
        deltaw2 = torch.sqrt(self.B2 + eps) * w2_grads / torch.sqrt(self.A2 + eps)
        w1, w2 = self.updateWeights(deltaw1, deltaw2, w1, w2)

        self.B1 = 0.9 * self.B1 + 0.1 * torch.square(deltaw1)
        self.B2 = 0.9 * self.B2 + 0.1 * torch.square(deltaw2)
        return w1, w2

    def RMSProp(self, w1, w2, w1_grads, w2_grads):
        eps = 1e-8
        self.A1 = 0.9 * self.A1 + 0.1 * torch.square(w1_grads)
        self.A2 = 0.9 * self.A2 + 0.1 * torch.square(w2_grads)

        deltaw1 = self.lr * w1_grads / torch.sqrt(self.A1 + eps)
        deltaw2 = self.lr * w2_grads / torch.sqrt(self.A2 + eps)
        w1, w2 = self.updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2

    def Adam(self, t, w1, w2, w1_grads, w2_grads):
        eps = 1e-8
        self.A1 = 0.9 * self.A1 + 0.1 * w1_grads
        self.A2 = 0.9 * self.A2 + 0.1 * w2_grads

        self.B1 = 0.999 * self.B1 + 0.001 * torch.square(w1_grads)
        self.B2 = 0.999 * self.B2 + 0.001 * torch.square(w2_grads)
        # bias corrected form
        mt1hat = self.A1 / (1-(0.9 ** (t+1)))
        mt2hat = self.A2 / (1-(0.9 ** (t+1)))

        vt1hat = self.B1 / (1-(0.999 ** (t+1)))
        vt2hat = self.B2 / (1-(0.999 ** (t+1)))

        deltaw1 = self.lr * mt1hat / (torch.sqrt(vt1hat) + eps)
        deltaw2 = self.lr * mt2hat / (torch.sqrt(vt2hat) + eps)
        w1, w2 = self.updateWeights(deltaw1, deltaw2, w1, w2)
        return w1, w2
