import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


matplotlib.style.use('ggplot')

def save_plots(train_acc, valid_acc):
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='real data'
    )
    plt.xlabel('')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()


x = 20 * np.random.rand(20000, 2) - 10
print(np.min(x), np.max(x))
y = np.sin(x[:, 0] + 2 * x[:, 1]) * np.exp(-(2 * x[:, 0] + x[:, 1])**2)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2201)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=2201)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float().reshape(-1, 1)
y_train = torch.from_numpy(y_train).float().reshape(-1, 1)
y_test = torch.from_numpy(y_test).float().reshape(-1, 1)

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))

        x = self.dropout(x)
        x = self.output_layer(x)
        return x

input_size = 2
hidden_size = 40
output_size = 1

model = RegressionModel(input_size, hidden_size, output_size)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    inputs = x_train
    targets = y_train

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/600], Loss: {loss.item()}")

# Calculate the mean squared error on the test set
with torch.no_grad():
    outputs = model(x_test)
    loss = criterion(outputs, y_test)
    mse = loss.item()
    outputs_val = model(x_val)

print(f"MSE on test set: {mse}")
save_plots(y_val, outputs_val)


