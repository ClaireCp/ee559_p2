""" Base class for all modules. Attributes:
        name (string): name of the module. Not necessary for parameterless modules. For modules with parameters however, a unique name is necessary to identify its parameters.
        _parameters (OrderedDict): parameters of the current module, with key=name and value=parameter
        _children (OrderedDict): children of the current module, with key=name, and value=module """
from collections import OrderedDict
from Parameter import *
import torch

class Module(object):
    """ Base class for all Modules"""
    def __init__(self, name):
        self.name = name
        self._parameters = OrderedDict()
        self._children = OrderedDict()
        self.training = True
        
    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)
        
    def forward(self, *input):
        """ defines the computation performed at every call """
        raise NotImplementedError
        
    def backward(self, *grad_output):
        """ backward receives as input a pointer to a tensor or a tuple of tensors containing the gradient of the loss (or the function of interest) wrt the module's output, accumulates the gradient wrt the parameters, and returns a tensor or a tuple of tensors containing the gradient of the loss wrt the module's input (Application of the chain rule) """
        raise NotImplementedError
        
    def add_children(self, module):
        """ adds a child to the current module; the module can be accessed as an attribute using the given name. """
        assert isinstance(module, Module) and module is not None, "{} is not a Module.".format(torch.typename(module))
        #assert module.name not in self._children, "module {} already in submodules".format(module.name)
        self._children[module.name] = module
        
    def add_parameter(self, name, param):
        """ adds a parameter to the module; the module can be accessed as an attribute using the given name. """
        assert isinstance(param, Parameter), "Not a Parameter."
        assert name not in self._parameters, "Parameter {} already exists".format(name)
        self._parameters[name] = param
        
    def param(self, recurse=True):
        """ param returns an iterator over Parameters, each composed of a parameter tensor, and a gradient tensor of same size. The OrderedDict is empty for parameterless modules. If recurse=True, then yields the parameters of this module and all its submodules. """      
        if recurse == False or isEmpty(self._children):
            yield self._parameters
        else:          
            for key_mod, module in self._children.items():
                yield module.param(recurse)             
            
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            if name in self._parameters: return self._parameters[name]
        if '_children' in self.__dict__:
            if name in self._children: return self._children[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(self.name, name))
                                 
def isEmpty(dict):
    if dict: return False
    else: return True