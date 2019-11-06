import torch
from torch.autograd import Variable


class RMMLayer(torch.autograd.Function):
    def __init__(self, input_dim_list=list(), output_dim=1024):
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [Variable(torch.randn(input_dim_list[i], output_dim)) for i in range(self.input_num)]
        for val in self.random_matrix:
            val.requires_grad = False

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_list[0] = return_list[0] / float(self.output_dim)
        return return_list

    def cuda(self):
        self.random_matrix = [val.cuda() for val in self.random_matrix]
