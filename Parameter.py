""" Base class for module parameters. Attributes:
        data (Tensor): parameter tensor
        grad (Tensor): tensor initialized with zero, for accumulating gradients
        requires_grad (bool): if the parameter requires gradient
        """

class Parameter(object):
    def __init__(self, tensor=None, grad=None, requires_grad=True):
        assert tensor is None or isinstance(tensor, torch.Tensor), "Not a tensor"
        self.data = tensor
        self.grad = torch.empty(tensor.size())
        self.requires_grad = requires_grad
    
    def set_data(self, tensor):
        assert tensor is None or isinstance(tensor, torch.Tensor), "Not a tensor"
        self.data = tensor  
    
    def set_grad_zero(self):
        self.grad = torch.zeros(self.grad.size())  

        