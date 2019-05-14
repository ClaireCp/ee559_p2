from importlib import reload
import Module, modules, optimizers, helpers
reload(Module)
reload(modules)
reload(helpers)
from modules import *
from helpers import *
from optimizers import *
torch.set_grad_enabled(False)

""" First we generate the data """
import random
import matplotlib.pyplot as plt
import torch

train_input = torch.rand((1000, 2))
test_input = torch.rand((1000, 2))
a = torch.Tensor([[0], [1]]) # We use One-Hot-Encoding for the targets
b = torch.Tensor([[1], [0]])
bound = 1 / torch.sqrt(torch.Tensor([2]))
train_radius = torch.sqrt(train_input[:,0]**2 + train_input[:,1]**2)
test_radius = torch.sqrt(test_input[:,0]**2 + test_input[:,1]**2)
train_target = torch.where(train_radius < bound, a, b).t()
test_target = torch.where(test_radius < bound, a, b).t()

"For visualization """
plt.scatter(train_input[:,0], train_input[:,1], c=train_target[:,0], s=10)
plt.scatter(test_input[:,0], test_input[:,1], c=test_target[:,0], s=10)

nb_0 = sum(train_target[:,0])
print("Are the classes balanced? nb_0 = ", nb_0)

""" We define the testing function for our model. We first need to set both the model in eval mode (training = False)
such that the forward pass doesn't override the self.save_for_backward of each module which is necessary for the 
backward pass. """
test_target_index = torch.max(test_target, 1)[1]
def test_model(model, test_input, test_target_index):
    model.eval()
    test_output = model(test_input)
    output_to_prediction = torch.max(test_output, 1)[1]
    test_accuracy = torch.where(output_to_prediction == test_target_index, 
                                torch.Tensor([1]), torch.Tensor([0])).sum() / len(test_input)
    model.train()
    return test_accuracy

""" Next we create our neural net and train it with MSELoss """
nb_input = 2
nb_hidden = 25
nb_output = 2
neural_net = Sequential(
    Linear('fc1', nb_input, nb_hidden), ReLU(),
    Linear('fc2', nb_hidden, nb_hidden), Tanh(),
    Linear('fc3', nb_hidden, nb_output), Tanh('tanh2')
    )

"For visualization """
output = neural_net(test_input)
output_to_prediction = torch.max(output, 1)[1]
plt.scatter(test_input[:,0], test_input[:,1], c=output_to_prediction, s=10)

test_acc = test_model(neural_net, test_input, test_target_index)
print("test_acc = ", test_acc)

criterion = MSELoss()
optimizer = SGD(neural_net, lr=0.01)
nb_epochs = 500

loss_history = []
test_acc_history = []

for e in range(nb_epochs):
    optimizer.zero_grad()
    output = neural_net(train_input)
    loss = criterion(output, train_target)
    loss_history.append(loss)
    test_accuracy = test_model(neural_net, test_input, test_target_index)
    test_acc_history.append(test_accuracy)
    if e % 40 == 0: print("e = {}, loss = {}, test accuracy = {}".format(e, loss, test_accuracy))
    grad_output = criterion.backward()
    neural_net.backward(grad_output)
    optimizer.step(loss)
    
    
import matplotlib.pyplot as plt
plt.plot(loss_history, label='Custom DL framework')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
plt.plot(test_acc_history, label='Custom DL framework')
plt.legend()
plt.show()


"For visualization """
output = neural_net(test_input)
output_to_prediction = torch.max(output, 1)[1]
plt.scatter(test_input[:,0], test_input[:,1], c=output_to_prediction, s=10)

test_acc = test_model(neural_net, test_input, test_target_index)
print("test_acc = ", test_acc)

train_input = torch.cat((train_input, train_input ** train_input), 1)
test_input = torch.cat((test_input, test_input ** test_input), 1)

neural_net2 = Sequential(
    Linear('fc1', nb_input*2, nb_hidden), ReLU(),
    Linear('fc2', nb_hidden, nb_hidden), ReLU('relu2'),
    Linear('fc3', nb_hidden, nb_output), 
    )

criterion = MSELoss()
optimizer = SGD(neural_net2, lr=0.01)
nb_epochs = 500

loss_history = []
test_acc_history = []

for e in range(nb_epochs):
    optimizer.zero_grad()
    output = neural_net2(train_input)
    loss = criterion(output, train_target)
    loss_history.append(loss)
    test_accuracy = test_model(neural_net2, test_input, test_target_index)
    test_acc_history.append(test_accuracy)
    if e % 40 == 0: print("e = {}, loss = {}, test accuracy = {}".format(e, loss, test_accuracy))
    grad_output = criterion.backward()
    neural_net2.backward(grad_output)
    optimizer.step(loss)

import matplotlib.pyplot as plt
plt.plot(test_acc_history, label='Custom DL framework')
plt.legend()
plt.show()

import demo
reload(demo)
from demo import *
test_standalone_linear_module()
test_sequential_module_with_relu_tanh_sigmoid()
test_assertion_error_with_multiple_unnamed_sigmoid()
test_not_corrupting_forward_variables_in_eval_mode()
test_assertion_error_without_sigmoid_before_BCELoss()

test_training_with_BCELoss()

test_training_with_BCEWithLogitsLoss()
test_on_mnist_data()
test_on_mnist_data_numerically_stable_training()



