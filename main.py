import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
print(torch.cuda.is_available())

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(9, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

# Create an instance of the neural network
model = NeuralNetwork()
model = model.double()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

#Loss functions
criterion = nn.MSELoss()

# Load the data from "NNTrainingData.csv"
data = pd.read_csv("NNTrainingData.csv")

# Extract the input data from the first 9 columns
X = torch.tensor(data.iloc[:, :9].values)
print(X.dtype)

# Extract the output data from the last column
y = torch.tensor(data.iloc[:, -1].values).view(-1, 1)
print(y.dtype)

#Loss
losses = []

#Forward pass
#print(model(X))

#Train the model
for epoch in range(10000):
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

#Save the model
torch.save(model.state_dict(), 'model.pth')