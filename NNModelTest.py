import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import function as f
import DataProcessing.DPfunctions as dp

#Get the data to test the model
base = pd.read_csv("DataProcessing/processed/testdata2.csv")
base = base.set_index('nr')
test_data = dp.col_filter(base, ['Ncycles'], 'exclude')
test_target = dp.col_filter(base, ['Ncycles'], 'include')
ndata = len(test_data.columns)

#Load the model
model = f.create_model(ndata, [14, 14, 14, 14, 14], 1)
model.load_state_dict(torch.load("NNModelArchive/model120324122635.pth"))
model.to('cuda')
print(model.dummy_param.device)

#Loss functions
criterion = nn.MSELoss()



X_test = torch.tensor(test_data.iloc[:, :ndata].values)
X_test = X_test.cuda()

#X_test.requires_grad = True

y_test = torch.tensor(test_target.iloc[:, -1].values).view(-1, 1)
y_test = y_test.cuda()
y_test_pred = model(X_test)
#(torch.sum(y_test_pred)).backward()
loss = criterion(y_test_pred, y_test)
print(loss.item())
log_err = y_test_pred-y_test

abs_cycle_error = torch.abs(torch.pow(10, y_test) - torch.pow(10, y_test_pred))
rel_cycle_error = torch.abs(torch.pow(10, y_test_pred) / torch.pow(10, y_test) )
print(float(torch.mean(abs_cycle_error)))
print(float(torch.mean(rel_cycle_error)))
print(log_err)

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
plt.plot([0, 8], [0, 8], color='red', linestyle='--')
plt.xlabel('y_test')
plt.ylabel('predicted')
plt.legend()
plt.xlim(0, 8)
plt.ylim(0,8)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.show()

#row number 69 from testdata
#SN Curve Implementation
'''testvalue = torch.tensor([[0.064575973,0.881653434,-1.251636953,-27.80118015,0.287646659,-0.089699437,-0.282228627,-2,-0.967860625,6.52150394]])
testvalue = testvalue.double()
testvalue = testvalue.cuda()

SNcurveinput = torch.empty_like(testvalue)  # Initialize SNcurveinput with the shape of testvalue
SNcurveinput = SNcurveinput.double()
SNcurveinput = SNcurveinput.cuda()

for i in range(400):
    testvalue[0, 7] = -2 + i * 0.01
    SNcurveinput = torch.cat((SNcurveinput, testvalue), dim=0)

    
SNcurveinput = SNcurveinput[:, :-1]

print(SNcurveinput)
print(SNcurveinput.shape)
SNcurveoutput = model(SNcurveinput)
plt.scatter(SNcurveoutput.cpu().detach().numpy(),SNcurveinput[:, 7].cpu().detach().numpy(), s=1)
plt.xlabel('Fatigue life')
plt.ylabel('Stress max')
plt.show()'''