import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        """
        Neural Network class for a simple feedforward network.
        This network consists of 5 linear layers with sigmoid activation function,
        followed by a final linear layer.
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(1, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 5)
        self.layer4 = nn.Linear(5, 5)
        self.layer5 = nn.Linear(5, 2)

    def forward(self, x):
        """
        Forward pass of the neural network.
        Applies sigmoid activation function to each linear layer except the last one.
        """
        x = nn.Sigmoid()(self.layer1(x))
        x = nn.Sigmoid()(self.layer2(x))
        x = nn.Sigmoid()(self.layer3(x))
        x = nn.Sigmoid()(self.layer4(x))
        x = self.layer5(x)
        return x

# Create an instance of the neural network
model = NeuralNetwork()

# Print the model architecture
print(model)

# Loss functions
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Create a random input tensor
input_tensor = torch.rand(1)

# Forward pass
output = model(input_tensor)
print(output)

# Backward pass and iterations
target = torch.tensor([1.0, 5])
epochs = 10000
losses = []
for epoch in range(epochs):
    output = model(input_tensor)
    loss = criterion(output, target) 
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss over epochs
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Test the model
output = model(input_tensor)
print(output)