import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import DPfunctions as dp
print(torch.cuda.is_available())

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.layer1 = nn.Linear(10, 12)
        self.layer2 = nn.Linear(12, 12)
        self.layer3 = nn.Linear(12, 12)
        self.layer4 = nn.Linear(12, 12)
        self.layer5 = nn.Linear(12, 12)
        self.layer6 = nn.Linear(12, 1)

    def forward(self, x):
        device = self.dummy_param.device
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

# Create an instance of the neural network
model = NeuralNetwork()
model = model.double()
model.to('cuda')
print(model.dummy_param.device)

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Loss functions
criterion = nn.MSELoss()

# Load the data from "NNTrainingData.csv"
base = pd.read_csv("Data processing/processed/traindata070324170302.csv")
base = base.set_index('nr')
data = dp.col_filter(base, ['Ncycles'], 'exclude')
target = dp.col_filter(base, ['Ncycles'], 'include')
print(data)
print(target)

# Extract the input data from the first 10 columns
X = torch.tensor(data.iloc[:, :10].values)
X = X.cuda()
print(X.device)

# Extract the output data from the last column
y = torch.tensor(target.iloc[:, -1].values).view(-1, 1)
y = y.cuda()
print(y.device)

#L1
losses = []

#Forward pass
#print(model(X))

#Train the model
for epoch in range(5000):
    #Forward pass
    y_pred = model(X)

    #Compute the loss
    loss = criterion(y_pred, y)
    losses.append(loss.item())

    #Zero the gradients
    optimizer.zero_grad()

    #Backward pass
    loss.backward()

    #Update the weights
    optimizer.step()

#Plot the loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Test the model
'''test_data = pd.read_csv("Data processing/processed/testdata070324170302.csv")
X_test = torch.tensor(test_data.iloc[:, :10].values)
X_test = X_test.cuda()
y_test = torch.tensor(test_data.iloc[:, -1].values).view(-1, 1)
y_test = y_test.cuda()
y_test_pred = model(X_test)
loss = criterion(y_test_pred, y_test)
print(loss.item())
print(y_test_pred-y_test)'''

#Save the model
path = 'NNModelArchive/model'+ dp.timetag() +'.pth'
torch.save(model.state_dict(), path)