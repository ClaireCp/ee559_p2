class Sequential(Module):
    """ Sequential module, Modules are added to it in the order tehey are passed in the constructor. """
    def __init__(self, *args):
        super(Sequential, self).__init__('seq_nn')
        for module in args:
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
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()
        self.add_parameter('weight', self.weight)
        self.add_parameter('bias', self.bias)
              
    def forward(self, input):
        self.save_for_backward = input
        output = torch.matmul(input, self.weight.data)
        if self.bias: 
            output += self.bias.data
        return output
              
    def backward(self, grad_output):
        input = self.save_for_backward
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
    def __init__(self):
        """ Applies the rectified linear unit function element-wise """
        super(ReLU, self).__init__('relu')
    
    def forward(self, input):
        self.save_for_backward = input
        return input.clamp(min=0)
    
    def backward(self, grad_output):
        input = self.save_for_backward
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
    
    
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__('tanh')
        
    def forward(self, input):
        self.save_for_backward = input
        return torch.tanh(input)
    
    def backward(self, grad_output):
        input = self.save_for_backward
        grad_input = 1 - torch.tanh(input)**2
        return grad_input 
    
    
class MSELoss(Module):
    def __init__(self, name=None):
        if name is None: name = 'mse'
        super(MSELoss, self).__init__(name)
    
    def forward(self, input, target):
        assert(input.size() == target.size()), "Input size different to target size."
        self.save_for_backward_input = input
        self.save_for_backward_target = target
        se = (input - target)**2
        return torch.mean(se)

    def backward(self, grad_output=None):
        input = self.save_for_backward_input
        target = self.save_for_backward_target
        grad_se = 2*(input - target) / len(input)
        return grad_se
 