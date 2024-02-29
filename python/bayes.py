from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np

# Prior parameters
alpha_prior = 1
beta_prior = 1

# Data: 10 successes out of 30 trials
successes = 10  # number of successes
trials = 30

# Posterior parameters
alpha_post = alpha_prior + successes
beta_post = beta_prior + trials - successes

# Plotting
x = np.linspace(0, 1, 1002)[1:-1]
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
plt.legend()
plt.show()

# Print the mean of the posterior
print(beta.mean(alpha_post, beta_post))

# Print the variance of the posterior
print(beta.var(alpha_post, beta_post))

# Print the 95% credible interval
print(beta.interval(0.95, alpha_post, beta_post))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('/workspaces/codespaces-blank/python/tickers/TSLA/index0002.csv')

# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Create training and test datasets
train_data = data_scaled[:int(len(data_scaled)*0.8)]
test_data = data_scaled[int(len(data_scaled)*0.8):]

# Run training and test data
train_data = np.expand_dims(train_data, axis=1)
test_data = np.expand_dims(test_data, axis=1)

# Print the shapes of the data
# print(train_data.shape)
# print(test_data.shape)

# Import libraries
import torch
import torch.nn as nn

# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1, 50)
        self.l2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

model = Net()

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    inputs = torch.from_numpy(train_data).float()
    targets = torch.from_numpy(train_data).float()

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

# Test model 
model.eval()
with torch.no_grad():
    inputs = torch.from_numpy(test_data).float()
    targets = torch.from_numpy(test_data).float()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    print(f'Test Loss: {loss.item()}')

# Create training and test labels
train_labels = data['Close'][:int(len(data)*0.8)]
test_labels = data['Close'][int(len(data)*0.8):]

# Save the data to file
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)

# Load the data from file
train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

# Print the shapes of the data
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
# print(train_data.shape)
# print(test_data.shape)
# print(train_labels.shape)
# print(test_labels.shape)