from importlib import reload
import Module, modules, optimizers, helpers
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from modules import *
from helpers import *
from optimizers import *
import demo
reload(demo)
from demo import *
torch.set_grad_enabled(False)

""" First we generate the data """
train_input, train_target, test_input, test_target = load_data_p2()
""" We use the test_target_index to test our model; this actually corresponds to the second column of our target (the second column is defined as 'outside', so if point is inside [1,0] and index=0, and if outside [0,1] and index=1. """
test_target_index = torch.max(test_target, 1)[1]
train_target_index = torch.max(train_target, 1)[1]

"For visualization """
plt.scatter(train_input[:,0], train_input[:,1], c=train_target[:,1], s=10)
plt.scatter(test_input[:,0], test_input[:,1], c=test_target[:,1], s=10)
plt.suptitle('Train and test sets with labelling')
plt.show()

nb_0 = sum(train_target[:,0])
print("Are the classes balanced? nb_0 = ", nb_0.item())

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
title = 'Initial untrained model prediction on train and test'
visualize_mse(neural_net, train_input, test_input, title) 
test_acc = test_model_mse(neural_net, test_input, test_target_index)
print("Initial test accuracy for untrained model = ", round(test_acc.item(), 4))

criterion = MSELoss()
optimizer = SGD(neural_net, lr=0.01)
nb_epochs = 500

loss_history = []
test_acc_history = []
train_acc_history = []

for e in range(nb_epochs):
    optimizer.zero_grad()
    output = neural_net(train_input)
    loss = criterion(output, train_target)
    loss_history.append(loss)
    test_accuracy = test_model_mse(neural_net, test_input, test_target_index)
    test_acc_history.append(test_accuracy)
    train_accuracy = test_model_mse(neural_net, train_input, train_target_index)
    train_acc_history.append(train_accuracy)
    if e % 100 == 0: print("e = {}, loss = {}, test accuracy = {}".format(e, round(loss.item(), 4), round(test_accuracy.item(), 4)))
    grad_output = criterion.backward()
    neural_net.backward(grad_output)
    optimizer.step(loss)
    
title = 'Plots for custom DL framework, training with MSELoss'
plot_loss_and_acc(loss_history, train_acc_history, test_acc_history, title)
title = 'Model prediction on train and test after training'
visualize_mse(neural_net, train_input, test_input, title)
test_acc = test_model_mse(neural_net, test_input, test_target_index)
print("Final test accuracy = ", round(test_acc.item(), 4))


from demo import *
print("Starting tests")
input("Press Enter to continue...")
test_standalone_linear_module()
input("Press Enter to continue...")
test_sequential_module_with_relu_tanh_sigmoid()
input("Press Enter to continue...")
test_assertion_error_with_multiple_unnamed_sigmoid()
input("Press Enter to continue...")
test_not_corrupting_forward_variables_in_eval_mode()
input("Press Enter to continue...")
test_assertion_error_without_sigmoid_before_BCELoss()
input("Press Enter to continue...")
test_assertion_error_when_calling_backward_first()
input("Press Enter to continue...")
test_training_with_BCELoss()
input("Press Enter to continue...")
test_training_with_BCEWithLogitsLoss()
input("Press Enter to continue...")
test_on_mnist_data()
input("Press Enter to continue...")
test_on_mnist_data_numerically_stable_training()



