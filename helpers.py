from modules import *
import types

def print_parameters_as_torch(model):
    for param_dict in model.param():
        if isinstance(param_dict, types.GeneratorType): param_dict = next(param_dict)
        for key, param in param_dict.items():
            print("Parameter containing:")
            if key == 'weight': print(param.data.t())
            if key == 'bias': print(param.data)
    
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

                                                