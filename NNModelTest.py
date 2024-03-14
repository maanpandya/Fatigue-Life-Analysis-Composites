import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import function as f
import DataProcessing.DPfunctions as dp
from PINNLoss import PINNLoss

#Get the data to test the model
base = pd.read_csv("DataProcessing/processed/testdata2.csv")
base = base.set_index('nr')
test_data = base.drop(columns=['Ncycles'])
test_target = base[['Ncycles']]
ndata = len(test_data.columns)

#Load the model
model = f.create_model(ndata, [20, 20, 20, 20, 20], 1)
model.load_state_dict(torch.load("NNModelArchive/model140324153816.pth"))
model.to('cuda')
print(model.dummy_param.device)

#Loss functions
criterion = nn.MSELoss()



X_test = torch.tensor(test_data.iloc[:, :ndata].values)
X_test = X_test.cuda()

X_test.requires_grad = True

y_test = torch.tensor(test_target.iloc[:, -1].values).view(-1, 1)
y_test = y_test.cuda()
y_test_pred = model(X_test)
(torch.sum(y_test_pred)).backward()
loss = criterion(y_test_pred, y_test)
print(loss.item())
log_err = y_test_pred-y_test
print(log_err)

'''abs_cycle_error = torch.abs(torch.pow(10, y_test) - torch.pow(10, y_test_pred))
rel_cycle_error = torch.abs(torch.pow(10, y_test_pred) / torch.pow(10, y_test) )
print(float(torch.mean(abs_cycle_error)))
print(float(torch.mean(rel_cycle_error)))'''

#Take the avg gradient of each column
Xgrad = [np.mean(X_test.grad[:,i].cpu().detach().numpy()) for i in range(ndata)]
print(Xgrad)
#print(X_test.grad[:,0])
#print(len(X_test.grad))
#print(len(X_test))
#print(torch.sum(X_test.grad[:,0]))
#print(torch.mean(X_test.grad[:,0]))
#print(torch.min(X_test.grad[:,0]))
#print(torch.max(X_test.grad[:,0]))

#print(loss.item())
#print(y_test_pred)
#print(y_test_pred)
#print(((y_test_pred-y_test)/y_test)*100)
#print(torch.mean(((y_test_pred-y_test)/y_test)*100))

plt.scatter(y_test.cpu().numpy(), y_test_pred.cpu().detach().numpy())
max = np.max(y_test_pred.cpu().detach().numpy())
plt.plot([0, max+1], [0, max+1], color='red', linestyle='--')
plt.xlabel('y_test')
plt.ylabel('predicted')
plt.legend()
plt.xlim(0, max+1)
plt.ylim(0,max+1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

#row number 1 from testdata2.csv
#SN Curve Implementation

testvalue = torch.tensor([[0.4079718252543413,1.6242047812119902,1.4644113964375178,0.16800788310210216,1.4595548966921477,-0.15179418517972942,1.8456687590702017,-1.1422271739143373,-1,3.714078164981856,-0.191945547470551,-1.4151157428500964,-0.02295207457515992]])
testvalue = testvalue.double()
testvalue = testvalue.cuda()

SNcurveinput = torch.empty_like(testvalue)  # Initialize SNcurveinput with the shape of testvalue
SNcurveinput = SNcurveinput.double()
SNcurveinput = SNcurveinput.cuda()

for i in range(200):
    testvalue[0, 8] = -1 + i * 0.01
    SNcurveinput = torch.cat((SNcurveinput, testvalue), dim=0)

    
SNcurveinput = SNcurveinput[:, :-1]

print(SNcurveinput)
print(SNcurveinput.shape)
SNcurveoutput = model(SNcurveinput)
plt.scatter(SNcurveoutput.cpu().detach().numpy(),SNcurveinput[:, 8].cpu().detach().numpy(), s=1)
plt.xlabel('Fatigue life')
plt.ylabel('Stress max')
plt.show()