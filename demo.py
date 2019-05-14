""" This source file provides useful demo that were tried during development to check the behavior of our deep-learning framework. To ensure that our modules do indeed behave correctly, we compare our results using PyTorch as reference (when possible). This source file is also useful to understand how to use the custom deep-learning framework. """
import Module, modules, optimizers, helpers
from importlib import reload
reload(Module)
reload(modules)
reload(helpers)
from modules import *
from helpers import *
from optimizers import *
import torch
torch.set_grad_enabled(False)
torch.set_default_tensor_type(torch.FloatTensor)

def load_mock_test_data():
    input, target, test_input, test_target = load_data_p2()
    # Our results are inconsistent with PyTorch for multi-output and MSELoss so we consider a one dimensional output and target
    target = target[:,0].view(-1,1)
    test_target = test_target[:,0].view(-1,1)
    return input, target, test_input, test_target


def train_custom_and_torch(model, criterion, model_torch, criterion_torch,
                           nb_epochs, lr):
    input, target, _, _ = load_mock_test_data()
    
    print("\nWorking with custom DL")
    optimizer = SGD(model, lr=lr)
    for e in range(nb_epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        grad_output = criterion.backward()
        if grad_output.dim() == 1: grad_output = grad_output.view(-1,1)
        if e % 100 == 0: print("e = {}, loss = {}".format(e, round(loss.item(), 5)))
        model.backward(grad_output)
        optimizer.step(loss)
 
    from torch import nn
    torch.set_grad_enabled(True)
    print("\nWorking with PyTorch")
    optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=lr)
    for e in range(nb_epochs):
        optimizer_torch.zero_grad()
        output_torch = model_torch(input)
        loss_torch = criterion_torch(output_torch, target)
        if e % 100 == 0: print("e = {}, loss = {}".format(e, round(loss_torch.item(), 5)))
        loss_torch.backward()
        optimizer_torch.step()
    del nn
    torch.set_grad_enabled(False)
    return


def test_model(model, test_input, test_target):
    model.eval()
    test_output = model(test_input)
    output_to_prediction = torch.ge(test_output, 0.5).flatten()
    test_accuracy = torch.where(output_to_prediction == test_target.flatten().type(torch.ByteTensor), torch.Tensor([1]), torch.Tensor([0])).sum() / len(test_input)
    model.train()
    return test_accuracy

def visualize(model, train_input, test_input, title):
    train_output = model(train_input)
    train_prediction = torch.ge(train_output, 0.5).flatten()
    test_output = model(test_input)
    test_prediction = torch.ge(test_output, 0.5).flatten()  
    plt.scatter(train_input[:,0], train_input[:,1], c=train_prediction, s=10)
    plt.scatter(test_input[:,0], test_input[:,1], c=test_prediction, s=10)
    plt.suptitle(title, y=0.95)
    plt.show()
    return

def plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title):
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.ylabel('Training loss')
    plt.subplot(2, 1, 2)
    plt.plot(test_acc_history, label="Test accuracy")
    plt.plot(train_acc_history, label="Train accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epochs')
    plt.suptitle(title, y=0.95)
    plt.legend()
    plt.show()
    return
    

def train_and_test_model(model, criterion, data, nb_epochs, lr):
    torch.set_grad_enabled(False)
    (input, target, test_input, test_target) = data
    loss_history = []
    test_acc_history = []
    train_acc_history = []
    
    print("\nWorking with custom DL")
    optimizer = SGD(model, lr=lr)
    for e in range(nb_epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss_history.append(loss)
        test_acc = test_model(model, test_input, test_target)
        test_acc_history.append(test_acc)
        train_acc = test_model(model, input, target)
        train_acc_history.append(train_acc)
        grad_output = criterion.backward()
        if grad_output.dim() == 1: grad_output = grad_output.view(-1,1)
        if e % 100 == 0: print("e = {}, loss = {}".format(e, round(loss.item(), 5)))
        model.backward(grad_output)
        optimizer.step(loss)       
    return loss_history, train_acc_history, test_acc_history

def train_and_test_model_torch(model, criterion, data, nb_epochs, lr):
    (input, target, test_input, test_target) = data
    loss_history = []
    test_acc_history = []
    train_acc_history = []
    
    from torch import nn
    torch.set_grad_enabled(True)
    print("\nWorking with PyTorch")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for e in range(nb_epochs):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output.flatten(), target)
        loss_history.append(loss)
        test_acc = test_model(model, test_input, test_target)
        test_acc_history.append(test_acc)
        train_acc = test_model(model, input, target)
        train_acc_history.append(train_acc)
        if e % 100 == 0: print("e = {}, loss = {}".format(e, round(loss.item(), 5)))
        loss.backward()
        optimizer.step() 
    del nn
    torch.set_grad_enabled(False)
    return loss_history, train_acc_history, test_acc_history


def test_standalone_linear_module():
    print("\n---------------------------------------------------------------------------")
    print("Testing standalone linear module.")
    print("My model: ")
    model = Linear('fc1', 2, 1)
    print_parameters_as_torch(model)
    criterion = MSELoss()
    
    from torch import nn
    print("\n Torch-model: ")
    model_torch = nn.Linear(2, 1)
    set_initial_parameters(model, model_torch)
    print_torch_parameters(model_torch)
    criterion_torch = nn.MSELoss()
    del nn
    
    train_custom_and_torch(model, criterion, model_torch, criterion_torch, nb_epochs=301, lr=0.01)
    return


def test_sequential_module_with_relu_tanh_sigmoid():
    print("\n---------------------------------------------------------------------------")
    print("Testing sequential module with every non-linearities.")
    print("My model: ")
    model = Sequential(
        Linear('fc1', 2, 6), ReLU(),
        Linear('fc2', 6, 2), Tanh(),
        Linear('fc3', 2, 1), Sigmoid())
    print_parameters_as_torch(model)
    criterion = MSELoss()
    
    from torch import nn
    print("\nTorch-model: ")
    model_torch = nn.Sequential(
        nn.Linear(2, 6), nn.ReLU(),
        nn.Linear(6, 2), nn.Tanh(),
        nn.Linear(2, 1), nn.Sigmoid())
    set_initial_parameters(model, model_torch)
    print_torch_parameters(model_torch)
    criterion_torch = nn.MSELoss()
    del nn
    
    train_custom_and_torch(model, criterion, model_torch, criterion_torch, nb_epochs=301, lr=0.01)
    return
  
    
def test_assertion_error_with_multiple_unnamed_sigmoid():
    print("\n---------------------------------------------------------------------------")
    print("Testing assertion error with multiple unnamed parameterless modules.")
    try:
        model = Sequential(
            Linear('fc1', 2, 6), ReLU(),
            Linear('fc2', 6, 2), Sigmoid(),
            Linear('fc3', 2, 1), Sigmoid())
        print("Uncorrect behavior, error should have been thrown.")
    except AssertionError as error:
        print(error)
        
        
def test_not_corrupting_forward_variables_in_eval_mode():
    print("\n---------------------------------------------------------------------------")
    print("Testing that save_for_backward attributes are not changed in eval mode")
    model = Sequential(
        Linear('fc1', 2, 6), ReLU(),
        Linear('fc2', 6, 4), Tanh(),
        Linear('fc3', 4, 1))
    criterion = MSELoss()
    input, target, test_input, test_target = load_mock_test_data()
    
    output = model(input) # save_for_backward variables are computed during the forward pass
    loss = criterion(output, target) # for the loss as well
    for module in model._children.values():
        print("After the first forward pass, for {}, save_for_backward = {}".format(module.name, module.save_for_backward[:3]))
    test_acc = test_model(model, test_input, test_target)             
    for module in model._children.values():
        print("After the second forward pass in eval mode, for {}, save_for_backward = {}".format(module.name, module.save_for_backward[:3]))
    return

def test_assertion_error_without_sigmoid_before_BCELoss():
    print("\n---------------------------------------------------------------------------")
    print("Testing assertion error when using BCELoss without final sigmoid layer.")
    model = Sequential(
        Linear('fc1', 2, 6), ReLU(),
        Linear('fc2', 6, 4), Tanh(),
        Linear('fc3', 4, 1))
    criterion = BCELoss()  
    data = load_mock_test_data()
    try:
        train_and_test_model(model, criterion, data, nb_epochs=301, lr=0.001)
        print("Uncorrect behavior, error should have been thrown.")
    except AssertionError as error:
        print(error)

              
def test_training_with_BCELoss():
    print("\n---------------------------------------------------------------------------")
    print("Testing BCELoss with sequential module.")
    model = Sequential(
        Linear('fc1', 2, 6), ReLU(),
        Linear('fc2', 6, 4), Tanh(),
        Linear('fc3', 4, 1), Sigmoid())
    criterion = BCELoss()    
    
    train_input, train_target, test_input, test_target = load_mock_test_data()
    data = (train_input, train_target, test_input, test_target)
    title = 'Initial untrained model predictions on train and test'
    visualize(model, train_input, test_input, title)
    
    loss_history, train_acc_history, test_acc_history = train_and_test_model(model, criterion, data, nb_epochs=301, lr=0.001)
    title = 'Plots for custom DL framework, training with BCELoss'
    plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title)
    title = 'Model predictions on train and test after training'
    visualize(model, train_input, test_input, title) 
    test_accuracy = test_model(model, test_input, test_target)
    print("Final test accuracy = ", test_accuracy)
    return
        
        
def test_training_with_BCEWithLogitsLoss():
    print("\n---------------------------------------------------------------------------")
    print("Testing BCEWithLogitsLoss with sequential module.")
    model = Sequential(
        Linear('fc1', 2, 6), ReLU(),
        Linear('fc2', 6, 4), Tanh(),
        Linear('fc3', 4, 1))
    criterion = BCEWithLogitsLoss()    
    
    train_input, train_target, test_input, test_target = load_mock_test_data()
    data = (train_input, train_target, test_input, test_target)
    title = 'Initial untrained model predictions on train and test'
    visualize(model, train_input, test_input, title)
    
    loss_history, train_acc_history, test_acc_history = train_and_test_model(model, criterion, data, nb_epochs=301, lr=0.001)   
    title = 'Plots for custom DL framework, training with BCEWithLogitsLoss'
    plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title)
    title = 'Model predictions on train and test after training'
    visualize(model, train_input, test_input, title) 
    test_accuracy = test_model(model, test_input, test_target)
    print("Final test accuracy = ", test_accuracy)
    return
    
import dlc_practical_prologue as prologue 
from dlc_practical_prologue import *
def test_on_mnist_data():
    print("\n---------------------------------------------------------------------------")
    print("Testing framework on (reduced) MNIST dataset.")
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
       load_random_datasets()
    train_input = train_input.view(len(train_input), -1)
    train_target = train_target.type(torch.FloatTensor)
    test_input = test_input.view(len(train_input), -1)
    test_target = test_target.type(torch.FloatTensor)
    data = (train_input, train_target, test_input, test_target)
    input_size = train_input.shape[1]
    nb_hidden1 = 10
    nb_hidden2 = 6
    
    model = Sequential(
    Linear('fc1', input_size, nb_hidden1), ReLU(),
    Linear('fc2', nb_hidden1, nb_hidden2), Tanh(),
    Linear('fc3', nb_hidden2, 1), Tanh('tanh2'))
    criterion = BCEWithLogitsLoss()

    from torch import nn
    model_torch = nn.Sequential(
        nn.Linear(input_size, nb_hidden1), nn.ReLU(),
        nn.Linear(nb_hidden1, nb_hidden2), nn.Tanh(),
        nn.Linear(nb_hidden2, 1), nn.Tanh())
    set_initial_parameters(model, model_torch)
    criterion_torch = nn.BCEWithLogitsLoss()
    del nn
    
    nb_epochs = 600
    title = 'Plots for custom DL framework (lr=0.001)'
    loss_history, train_acc_history, test_acc_history = train_and_test_model(model, criterion, data, nb_epochs, lr=0.001)
    plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title)
    test_accuracy = test_model(model, test_input, test_target)
    print("Final test accuracy = ", test_accuracy)
    
    title = 'Plots for PyTorch (lr=0.01)'
    loss_history_torch, train_acc_history_torch, test_acc_history_torch = train_and_test_model_torch(model_torch, criterion_torch, data, nb_epochs, lr=0.01)
    plot_loss_and_acc(loss_history_torch, train_acc_history_torch, test_acc_history_torch, title)
    test_accuracy_torch = test_model(model_torch, test_input, test_target)
    print("Final test accuracy with torch = ", test_accuracy_torch)
    
    

def test_on_mnist_data_numerically_stable_training():
    print("\n---------------------------------------------------------------------------")
    print("Testing framework on (reduced) MNIST dataset., stable training conditions.")
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
       load_random_datasets()
    train_input = train_input.view(len(train_input), -1)
    train_target = train_target.type(torch.FloatTensor)
    test_input = test_input.view(len(train_input), -1)
    test_target = test_target.type(torch.FloatTensor)
    data = (train_input, train_target, test_input, test_target)
    input_size = train_input.shape[1]
    nb_hidden1 = 100
    nb_hidden2 = 200
    
    model = Sequential(
    Linear('fc1', input_size, nb_hidden1), ReLU(),
    Linear('fc2', nb_hidden1, nb_hidden2), Tanh(),
    Linear('fc3', nb_hidden2, 1), Tanh('tanh2'))
    criterion = BCEWithLogitsLoss()

    from torch import nn
    model_torch = nn.Sequential(
        nn.Linear(input_size, nb_hidden1), nn.ReLU(),
        nn.Linear(nb_hidden1, nb_hidden2), nn.Tanh(),
        nn.Linear(nb_hidden2, 1), nn.Tanh())
    set_initial_parameters(model, model_torch)
    criterion_torch = nn.BCEWithLogitsLoss()
    del nn
    
    nb_epochs = 600
    title = 'Plots for custom DL framework (lr=0.0001)'
    loss_history, train_acc_history, test_acc_history = train_and_test_model(model, criterion, data, nb_epochs, lr=0.00001)
    plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title)
    test_accuracy = test_model(model, test_input, test_target)
    print("Final test accuracy = ", test_accuracy)
    
    title = 'Plots for PyTorch (lr=0.1)'
    loss_history_torch, train_acc_history_torch, test_acc_history_torch = train_and_test_model_torch(model_torch, criterion_torch, data, nb_epochs, lr=0.01)
    plot_loss_and_acc(loss_history_torch, train_acc_history_torch, test_acc_history_torch, title)
    test_accuracy_torch = test_model(model_torch, test_input, test_target)
    print("Final test accuracy with torch = ", test_accuracy_torch)
    
    
    
  