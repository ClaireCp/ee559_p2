""" This source file contains some helper functions to load the initial train and test sets, as well as tests and plots methods, and other functions needed to compare our deep-learning framework with PyTorch """
from modules import *
import types
import random
import matplotlib.pyplot as plt
import torch

##########################################################################################
""" Functions needed for the main problem of project2: 
        1) Generating the training and test sets 
        2) Training the network with MSE 
        3) Computing the train and test errors """
def load_data_p2():
    """ To generate the train and test sets """
    train_input = torch.rand((1000, 2))
    test_input = torch.rand((1000, 2))
    a = torch.Tensor([[1], [0]]) # We use One-Hot-Encoding for the targets
    b = torch.Tensor([[0], [1]])
    bound = 1 / torch.sqrt(torch.Tensor([2]))
    train_radius = torch.sqrt(train_input[:,0]**2 + train_input[:,1]**2)
    test_radius = torch.sqrt(test_input[:,0]**2 + test_input[:,1]**2)
    train_target = torch.where(train_radius < bound, a, b).t()
    test_target = torch.where(test_radius < bound, a, b).t()
    return train_input, train_target, test_input, test_target

def test_model_mse(model, test_input, test_target_index):
    """ We define the testing function for our model. We first need to set the model in eval mode (training = False) such that the forward pass doesn't override the self.save_for_backward of each module which is necessary for the backward pass. Arguments:
        test_input: input for our model
        test_target_index: the target as index (ie 0 for target [1, 0]; 1 for target [0, 1] """
    model.eval()
    test_output = model(test_input)
    output_to_prediction = torch.max(test_output, 1)[1]
    test_accuracy = torch.where(output_to_prediction == test_target_index,torch.Tensor([1]), torch.Tensor([0])).sum() / len(test_input)
    model.train()
    return test_accuracy

def visualize_mse(model, train_input, test_input, title):
    """ For visualizing the model predictions on both the training and test sets. To predict, we consider that the max value of our two output units corresponds to the +1 label, the other to the 0 label. """
    train_output = model(train_input)
    train_prediction = torch.max(train_output, 1)[1]
    test_output = model(test_input)
    test_prediction = torch.max(test_output, 1)[1]
    plt.scatter(train_input[:,0], train_input[:,1], c=train_prediction, s=10)
    plt.scatter(test_input[:,0], test_input[:,1], c=test_prediction, s=10)
    plt.suptitle(title, y=0.95)
    plt.show()
    return

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
                                                