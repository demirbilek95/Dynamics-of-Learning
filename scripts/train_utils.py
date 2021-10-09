import torch
from torch import Tensor

class AverageMeter(object):
    """
    a generic class to keep track of performance metrics during training or testing of models
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def predict_single_out(nn_output: torch.Tensor):
    """
    Make prediction when model output is only one node
    if probability is higher than 0.5, prediction is 1, else 0
    :param nn_output:
    :return: prediction of the model
    """
    nn_output[nn_output > 0.5] = 1
    nn_output[nn_output < 0.5] = 0
    return nn_output.reshape(len(nn_output)).int()


def predict_multiple_out(nn_output: torch.Tensor):
    return torch.argmax(nn_output, dim=1)


def accuracy(nn_output: torch.Tensor, ground_truth: torch.Tensor, loss_type="Cross Entropy"):
    if loss_type == "Cross Entropy":
        nn_out_classes = predict_multiple_out(nn_output)
    else:
        nn_out_classes = predict_single_out(nn_output)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc
