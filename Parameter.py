""" Base class for module parameters. Attributes:
        data (Tensor): parameter tensor
        grad (Tensor): tensor initialized with zero, for accumulating gradients """
import torch

class Parameter(object):
    def __init__(self, tensor=None, grad=None):
        assert tensor is None or isinstance(tensor, torch.Tensor), "Not a tensor"
        self.data = tensor
        self.grad = torch.zeros(tensor.size())
    
    def set_data(self, tensor):
        assert tensor is None or isinstance(tensor, torch.Tensor), "Not a tensor"
        self.data = tensor  
    
    def set_grad_zero(self):
        self.grad = torch.zeros(self.grad.size())  

        