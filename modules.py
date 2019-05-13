import Module
from Module import *
import math

class Sequential(Module):
    """ Sequential module, Modules are added to it in the order tehey are passed in the constructor. """
    def __init__(self, *args):
        super(Sequential, self).__init__('seq_nn')
        for key_mod, module in enumerate(args):
            self.add_children(module)
            
    def forward(self, input):
        self.save_for_backward = input
        for module in self._children.values():
            input = module(input)
        return input
    
    def backward(self, *grad_output):
        for module in reversed(self._children.values()):
            if isinstance(grad_output, tuple): grad_output = grad_output[0]
            grad_output = module.backward(grad_output)
        return grad_output      
    

class Linear(Module):
    """ Implements a R^C -> R^D fully-connected layer:
            name (string): name of the module. Has to be unique for a Sequential module
            in_features: size of each input sample
            out_features: size of each output sample
            bias (bool): if set to 'False', the layer will not learn an additive bias; default: 'True'
         """
    def __init__(self, name, in_features, out_features, bias=True):
        assert name is not None, "Module that have parameters must have a unique name"
        super(Linear, self).__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = None
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.add_parameter('weight', self.weight)
        if bias:
            self.add_parameter('bias', self.bias)
              
    def forward(self, input):
        if self.training == True:
            self.save_for_backward = input
        if input.dim() == 1:
            output = input * self.weight.data
        else:
            output = torch.matmul(input, self.weight.data)
        if self.bias: 
            output += self.bias.data
        return output
              
    def backward(self, grad_output):
        assert(hasattr(self, 'save_for_backward')), "backward() should only be called after a forward pass."
        input = self.save_for_backward
        if input.dim() == 1:
            grad_input = grad_output * self.weight.data
            grad_weight = input * grad_output
        else: 
            grad_input = torch.matmul(grad_output, self.weight.data.t())
            grad_weight = torch.matmul(input.t(), grad_output)
        self.weight.grad += grad_weight
        if self.bias: 
            grad_bias = grad_output.sum(0).squeeze(0)
            self.bias.grad += grad_bias
        return grad_input 
    
    def reset_parameters(self):
        gain = calculate_gain('linear')
        stdv = gain / math.sqrt(self.in_features)
        bound = math.sqrt(3.0) * stdv
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.uniform_(-bound, bound)
         
        
class ReLU(Module):
    def __init__(self, name=None):
        """ Applies the rectified linear unit function element-wise """
        if name is None: name = 'relu'
        super(ReLU, self).__init__(name)
    
    def forward(self, input):
        if self.training == True:
            self.save_for_backward = input
        return input.clamp(min=0)
    
    def backward(self, grad_output):
        assert(hasattr(self, 'save_for_backward')), "backward() should only be called after a forward pass."
        input = self.save_for_backward
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
    
class Tanh(Module):
    def __init__(self, name=None):
        if name is None: name = 'tanh'
        super(Tanh, self).__init__(name)
        
    def forward(self, input):
        if self.training == True:
            self.save_for_backward = input
        return torch.tanh(input)
    
    def backward(self, grad_output):
        assert(hasattr(self, 'save_for_backward')), "backward() should only be called after a forward pass."
        input = self.save_for_backward
        grad_input = 1 - torch.tanh(input)**2
        return grad_input * grad_output

    
class Sigmoid(Module):
    def __init__(self, name=None):
        if name is None: name = 'sigmoid'
        super(Sigmoid, self).__init__(name)
            
    def forward(self, input):
        if self.training == True:
            self.save_for_backward = input
        return 1 / (1 + torch.exp(-input))
    
    def backward(self, grad_output):
        assert(hasattr(self, 'save_for_backward')), "backward() should only be called after a forward pass."
        eps = 1e-12
        input = self.save_for_backward
        sigmoid = 1 / (1 + torch.exp(-input))
        #return sigmoid * ((1 - sigmoid).clamp(min=eps)) * grad_output
        return sigmoid * (1 - sigmoid) * grad_output
               
    
class MSELoss(Module):
    def __init__(self, name=None):
        if name is None: name = 'mse'
        super(MSELoss, self).__init__(name)
    
    def forward(self, input, target):
        assert(input.size() == target.size()), "Input size different to target size."
        if self.training == True:
            self.save_for_backward_input = input
            self.save_for_backward_target = target
        se = (input - target)**2
        return torch.mean(se)

    def backward(self, grad_output=None):
        assert(hasattr(self, 'save_for_backward_input')), "backward() should only be called after a forward pass."
        input = self.save_for_backward_input
        target = self.save_for_backward_target
        grad_se = 2*(input - target) / len(input)
        return grad_se
    

class BCELoss(Module):
    def __init__(self, name=None):
        if name is None: name = 'bce'
        super(BCELoss, self).__init__(name)
        
    def forward(self, input, target):
        assert(input.size() == target.size()), "Input size different to target size."
        assert(input.dim() == 1), "Input and target must be 1d."
        a = torch.Tensor([1])
        b = torch.Tensor([0])
        assert(torch.where(input < 0, a, b).sum() == 0. and torch.where(input > 1, a, b).sum() == 0.), "Input values must be between 0 and 1."
        if self.training == True:
            self.save_for_backward_input = input
            self.save_for_backward_target = target
        eps = 1e-12
        #return - (target * torch.log(input.clamp(min=eps)) + (1 - target) * torch.log((1 - input).clamp(min=eps))).mean()
        return (- target * torch.log(input + eps) - (1 - target) * torch.log(1 - input + eps)).mean()
    
    def backward(self, grad_output=None):
        assert(hasattr(self, 'save_for_backward_input')), "backward() should only be called after a forward pass."
        eps = 1e-12
        if self.training == True:
            input = self.save_for_backward_input
            target = self.save_for_backward_target   
        # We multiply by a constant factor (0.143) to get the same results as the BCE loss implemented in PyTorch
        # Since it's a constant factor, it doesn't actually influence the solution of the optimization
        #return (input - target) / ((input * (1 - input)).clamp(min=eps)) * 0.143
        return (input - target) / ((input + eps) * (1 - input + eps))
    
    
class BCEWithLogitsLoss(Module):
    def __init__(self, name=None):
        if name is None: name = 'bceLogits'
        super(BCEWithLogitsLoss, self).__init__(name)         
            
    def forward(self, input, target):
        assert(input.size() == target.size()), "Input size different to target size."
        assert(input.dim() == 1), "Input and target must be 1d."
        if self.training == True:
            self.save_for_backward_input = input
            self.save_for_backward_target = target
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()
    
    def backward(self, grad_output=None):
        assert(hasattr(self, 'save_for_backward_input')), "backward() should only be called after a forward pass."
        eps = 1e-12
        input = self.save_for_backward_input
        target = self.save_for_backward_target        
        sigmoid = (1 / (1 + torch.exp(-input)))
        return - target * (1 - sigmoid) + (1 - target) * sigmoid
            
def calculate_gain(nonlinearity='relu'):
    linear_fns = ['linear', 'conv1d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    else:
        raise ValueError("Specified non-linearity is not implemented")