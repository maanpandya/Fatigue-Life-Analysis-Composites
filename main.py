import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import function as f
import DataProcessing.DPfunctions as dp
import PINNLoss
print(torch.cuda.is_available())

# Load the data
base = pd.read_csv("DataProcessing/processed/traindata2.csv")
base = base.set_index('nr')
data = base.drop(columns=['Ncycles'])
target = base[['Ncycles']]
ndata = len(data.columns)
print('n inputs:'+str(ndata))
#print(data)
#print(target)

# Create an instance of the neural network
model = f.create_model(ndata, [20, 20, 20, 20, 20], 1)
model.to('cuda')
print(model.dummy_param.device)

#Optimizer
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Extract the input data from the first 10 columns
X = torch.tensor(data.iloc[:, :ndata].values)
X = X.cuda()
print(X.device)

# Extract the output data from the last column
y = torch.tensor(target.iloc[:, -1].values).view(-1, 1)
y = y.cuda()
print(y.device)

#L1
losses = []

X.requires_grad = True

#Train the model
epochs = 30000
for epoch in range(epochs):
    #Forward pass
    y_pred = model(X)

    #Compute the loss
    loss = PINNLoss(y_pred, y, X)
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
tag = dp.timetag()
plt.title('epoch = '+str(epochs)+', lr = '+str(lr)+', tag = '+tag)
plt.savefig("NNModelArchive/Loss Function Convergence/loss.png")
plt.show()
print(losses[-1])

#Save the model
path = 'NNModelArchive/model'+ tag +'.pth'
torch.save(model.state_dict(), path)