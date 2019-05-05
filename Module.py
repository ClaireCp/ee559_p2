class Module(object):
    """ Base class """
    def __init__(self, name):
        self.name = name
        self._parameters = OrderedDict()
        self._children = OrderedDict()
        self.training = True
        
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
        
    def forward(self, *input):
        raise NotImplementedError
        
    def backward(self, *grad_output):
        """ backward receives as input a pointer to a tensor or a tuple of tensors containing the gradient of the loss (or the function of interest) wrt the module's output, accumulates the gradient wrt the parameters, and returns a tensor or a tuple of tensors containing the gradient of the loss wrt the module's input (Application of the chain rule). """
        raise NotImplementedError
        
    def add_children(self, module):
        print("adding child = ", module)
        assert isinstance(module, Module) and module is not None, "Not a Module."
        assert module.name not in self._children, "Module {} already exists".format(module.name)
        self._children[module.name] = module
        
    def add_parameter(self, name, param):
        assert isinstance(param, Parameter), "Not a Parameter."
        assert name not in self._parameters, "Parameter {} already exists".format(name)
        self._parameters[name] = param
        
    def param(self, recurse=True):
        """ param returns a list of Parameters, each composed of a parameter tensor, and a gradient tensor of same size. This list is empty for parameterless modules. """
        if recurse == False or self._children is not None:
            print("Arrived in leaf module")
            return self.param_per_module()
        else:
            for key_mod, module in self._children.items():
                print("Looping over children, module = ", module)
                for key_param, parameter in module._parameters:
                    return param_per_module()
                    
    
    def param_per_module(self):
        if self._parameters:
            yield self._parameters
        else:
            yield None