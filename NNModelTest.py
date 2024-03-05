import torch
import pandas as pd
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.layer1 = nn.Linear(9, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 10)
        self.layer4 = nn.Linear(10, 10)
        self.layer5 = nn.Linear(10, 10)
        self.layer6 = nn.Linear(10, 1)

    def forward(self, x):
        device = self.dummy_param.device
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x

#Load the model
model = NeuralNetwork().double()
model.load_state_dict(torch.load("model.pth"))
model.to('cuda')
print(model.dummy_param.device)

#Loss functions
criterion = nn.MSELoss()

#Test the model
test_data = pd.read_csv("NNTestData.csv")
X_test = torch.tensor(test_data.iloc[:, :9].values)
X_test = X_test.cuda()
y_test = torch.tensor(test_data.iloc[:, -1].values).view(-1, 1)
y_test = y_test.cuda()
y_test_pred = model(X_test)
loss = criterion(y_test_pred, y_test)
print(loss.item())
print(y_test_pred)
print(y_test_pred-y_test)