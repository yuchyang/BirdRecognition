import torch


class SilenceLayer(torch.autograd.Function):
    def __init__(self):
        pass

    def forward(self, input):
        return input * 1.0

    def backward(self, gradOutput):
        return 0 * gradOutput