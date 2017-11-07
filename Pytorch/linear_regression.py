import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable



X_val = [i for i in range(11)]
X_train = np.array(X_val , dtype=np.float32)
X_train = np.reshape(X_train,(X_train.shape[0], 1))

Y_train = [2*X_val[i] + 1 for i in X_val]
Y_train = np.array(Y_train , dtype = np.float32)
Y_train = Y_train.reshape(-1, 1)

# Now our X and Y labels are ready we can perform simple Linear regression and use gradient descent to minimize the error

class LinearRegression(nn.Module):
    def __init__(self, input_dim , output_dim):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(input_dim , output_dim)

    def forward(self,X):
        out = self.linear(X)
        return out


input_dim = 1
output_dim = 1

model = LinearRegression(input_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()

loss_op = nn.MSELoss()

lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

epoch = 100
for i in range(epoch):
    if torch.cuda.is_available():
        X = Variable(torch.from_numpy(X_train).cuda())
        Y = Variable(torch.from_numpy(Y_train).cuda())
    else:
        X = Variable(torch.from_numpy(X_train))
        Y = Variable(torch.from_numpy(Y_train))

    optimizer.zero_grad()

    output = model(X)

    loss = loss_op(output,Y)

    loss.backward()

    # Updates parameters according to the optimizer we selected
    optimizer.step()

    print("cost is %d " % (loss.data[0]))

    
