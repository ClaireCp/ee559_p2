""" This source file contains some helper functions to load the initial train and test sets, and to compare our deep-learning framework with PyTorch """
from modules import *
import types
import random
import matplotlib.pyplot as plt
import torch

def load_data_p2():
    train_input = torch.rand((1000, 2))
    test_input = torch.rand((1000, 2))
    a = torch.Tensor([[0], [1]]) # We use One-Hot-Encoding for the targets
    b = torch.Tensor([[1], [0]])
    bound = 1 / torch.sqrt(torch.Tensor([2]))
    train_radius = torch.sqrt(train_input[:,0]**2 + train_input[:,1]**2)
    test_radius = torch.sqrt(test_input[:,0]**2 + test_input[:,1]**2)
    train_target = torch.where(train_radius < bound, a, b).t()
    test_target = torch.where(test_radius < bound, a, b).t()
    return train_input, train_target, test_input, test_target

##########################################################################################

def print_parameters_as_torch(model):
    for param_dict in model.param():
        if isinstance(param_dict, types.GeneratorType): param_dict = next(param_dict)
        for key, param in param_dict.items():
            print("Parameter containing:")
            if key == 'weight': print(param.data.t())
            if key == 'bias': print(param.data)
                
def print_torch_parameters(model_torch):
    for param in model_torch.parameters():
        print(param)
    
def set_initial_parameters(model, model_torch):
    param_list = []
    for param_dict in model.param():
        if isinstance(param_dict, types.GeneratorType): param_dict = next(param_dict)
        if param_dict is not None:
            for key, param in param_dict.items():
                if key == 'weight': param_list.append(param.data.t().clone())
                if key == 'bias': param_list.append(param.data.clone())
    for param, param_torch in zip(param_list, model_torch.parameters()):
        param_torch.data = param.clone()

def hook(module, grad_input, grad_output):
    for grad in grad_output:
        print("grad_output = ", grad_output[0][:5].t())
        break
                                                