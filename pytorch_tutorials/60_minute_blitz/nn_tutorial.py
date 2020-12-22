import torch
import torch.nn as nn
import torch.nn.functional as F

# Classify digit images
"""
INPUT -> Conv2d -> MaxPool -> Conv2d -> MaxPool -> FC -> FC -> Output 
"""

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3) # 1 input channel, 6 output channels, 3x3 convolution
        self.conv2 = nn.Conv2d(6, 16, 3)
        # linear transformation, y = xA.T + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6x6 from image dimensino
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pool over 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))   # reshape the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(x)

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions besides the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
#print(net)
params = list(net.parameters())
# print(params[0].size()) # conv1's weight
nn_input = torch.randn(1, 1, 32, 32)
nn_out = net(nn_input)
# print(nn_out)
# net.zero_grad()
# nn_out.backward(torch.randn(1, 10))

target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(nn_out, target)
# print(loss)
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backprop, should zero out existing gradients
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update weights: w = w - learning_rate(gradient)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim # import various optimizers like RMSprop, Adam, etc
# do in training loop
optimizer = optim.Adam(net.parameters(), lr=0.01)
optimizer.zero_grad()
nn_output = net(nn_input)
loss = criterion(nn_output, target)
loss.backward()
optimizer.step() # does the update