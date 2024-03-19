import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from PINNLoss import PINNLoss


def create_model(n_inputs, layers=None, n_outputs=1):
    if layers is None:
        layers = [10, 10, 10, 10, 10]
    if len(layers) != 5:
        raise 'incompatable layer list'

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.dummy_param = nn.Parameter(torch.empty(0))
            self.layer1 = nn.Linear(n_inputs, layers[0])
            self.layer2 = nn.Linear(layers[0], layers[1])
            self.layer3 = nn.Linear(layers[1], layers[2])
            self.layer4 = nn.Linear(layers[2], layers[3])
            self.layer5 = nn.Linear(layers[3], layers[4])
            self.layer6 = nn.Linear(layers[4], n_outputs)

        def forward(self, x):
            device = self.dummy_param.device
            x = torch.sigmoid(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            x = torch.relu(self.layer3(x))
            x = torch.relu(self.layer4(x))
            x = torch.relu(self.layer5(x))
            x = self.layer6(x)
            return x

    # Load the model
    model = NeuralNetwork().double()
    return model


def create_model_2(n_inputs, layer_sizes, n_outputs, n_hidden_layers, act_fn):
    if type(layer_sizes) != list:
        layer_sizes = n_hidden_layers * [layer_sizes]
    if type(act_fn) != list:
        act_fn = n_hidden_layers * [act_fn]

    # Define a list to hold the layers
    layers = []

    # Add input layer
    layers.append(nn.Linear(n_inputs, layer_sizes[0]))
    layers.append(act_fn[0])

    # Add hidden layers
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(act_fn[i + 1])

    # Add output layer
    layers.append(nn.Linear(layer_sizes[-1], n_outputs))

    # Combine all layers into a sequential model
    model = nn.Sequential(*layers)

    return model


def train_model(model, loss_fn, optimizer, n_epochs, learning_rate, x_train, y_train):
    X = torch.tensor(x_train.iloc[:, :len(x_train.columns)].values)
    X = X.cuda()
    print(X.device)
    X.requires_grad = True

    # Extract the output data from the last column
    y = torch.tensor(y_train.iloc[:, -1].values).view(-1, 1)
    y = y.cuda()
    print(y.device)


    losses = []
    t = time.time()
    n = 10
    model.train()

    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)

        # Compute the loss
        if loss_fn == PINNLoss:
            loss = loss_fn(y_pred, y, X)
        else:
            loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        # Zero the gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # show progress
        progress = int((epoch / n_epochs) * 100)

        if progress - n >= 0:
            n = n + 10
            print('training progress: '+str(progress)+'%')
            print('time remaining: ' + str(((time.time()-t) / progress) * (100 - progress)) + 's')
    print('done in ' + str(time.time()-t))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('epoch = ' + str(n_epochs) + ', lr = ' + str(learning_rate))
    plt.show()
    return model


def test_model(model, loss_fn, scaler, x_test, y_test):
    model.eval()
    X_test = torch.tensor(x_test.iloc[:, :len(x_test.columns)].values)
    X_test = X_test.cuda()

    X_test.requires_grad = True

    Y_test = torch.tensor(y_test.iloc[:, -1].values).view(-1, 1)
    Y_test = Y_test.cuda()
    y_test_pred = model(X_test)
    pred_eval = pd.DataFrame(y_test_pred.cpu().detach().numpy()).set_index(y_test.index)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'pred_scaled'})
    pred_eval = pred_eval.join(y_test)
    pred_eval = pred_eval.rename(columns={pred_eval.columns[-1]: 'real_scaled'})
    scaler = scaler[y_test.columns[0]]
    pred_eval['pred_log'] = pred_eval['pred_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['real_log'] = pred_eval['real_scaled'] * scaler['std'] + scaler['mean']
    pred_eval['pred'] = np.power(10, pred_eval['pred_log'])
    pred_eval['real'] = np.power(10, pred_eval['real_log'])
    print(pred_eval)
    print(np.mean(np.power(pred_eval['pred_log'] - pred_eval['real_log'], 2)))



    plt.scatter(pred_eval['real_log'], pred_eval['pred_log'])
    plt.plot([0, 10], [0, 10], color='red', linestyle='--')
    plt.xlabel('y_test')
    plt.ylabel('predicted')
    plt.legend()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def export_model(model):
    pass
