# Part 1: Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_extract, count, sum, lit
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import CountVectorizer, Tokenizer , HashingTF, VectorAssembler, StringIndexer  
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



# Initialize Spark session
spark = SparkSession.builder \
    .appName('SentimentAnalysis') \
    .getOrCreate()
    
    # .config('spark.hadoop.fs.defaultFS', 'hdfs://namenode:9000') \
    # .config('spark.executor.memory', '1g') \
    # .config('spark.executor.cores', '1') \
    # .config('spark.yarn.am.memory', '1g') \
    # .getOrCreate()

# Start #
print("Start")
#  Part 2: Data Preparation

# Define transformations for MNIST images (normalization)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Load MNIST datasets (modify paths if needed)
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
print('Train Dataset:' + str(len(train_data)) + ' Images')
print('Test Dataset:' + str(len(test_data)) + ' Images')


image, label = train_data[0]

image = image.squeeze().numpy()

# Plot the image
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.savefig("Sample of the dataset.png", dpi=300)
plt.show()

image2, label2 = train_data[1]

image2 = image2.squeeze().numpy()

# Plot the image
plt.imshow(image2, cmap='gray')
plt.title(f"Label: {label2}")
plt.savefig("Sample 2 of the dataset.png", dpi=300)
plt.show()


# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


#Part 3: Custom BatchNorm class

# Custom BatchNorm class
class CustomBatchNorm(nn.Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
    super(CustomBatchNorm, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))
    if self.affine:
      self.weight = nn.Parameter(torch.ones(num_features))
      self.bias = nn.Parameter(torch.zeros(num_features))

  def forward(self, x):
    if self.training:
      self.batch_mean = torch.mean(x, dim=0)
      self.batch_var = torch.var(x, dim=0, unbiased=False)
      # Update running statistics with momentum
      self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
      self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
    else:
      self.batch_mean = self.running_mean
      self.batch_var = self.running_var
    x_hat = (x - self.batch_mean) / torch.sqrt(self.batch_var + self.eps)
    if self.affine:
      x_hat = x_hat * self.weight + self.bias
    return x_hat
  
  #Part 4: Model Architectures

  # Define Model NoBN (replace with your specific architecture if needed)
class ModelNoBN(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super(ModelNoBN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define Model WithBN (replace with your specific architecture if needed)
class ModelWithBN(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, num_classes=10):
        super(ModelWithBN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = CustomBatchNorm(hidden_size)  # Use the custom BatchNorm class
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # Apply BatchNorm after the first linear layer
        x = self.relu(x)
        x = self.fc2(x)
        return x

#Part 5: Training Function

def train(model, train_loader, optimizer, criterion, epochs):
  loss_history = []
  for epoch in range(epochs):
    for data, target in train_loader:
      optimizer.zero_grad() # sets the gradients of all model parameters to zero before each forward pass and backward pass
      # Flatten the input data
      data = data.view(data.size(0), -1)  # Reshape to (batch_size, -1)
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      loss_history.append(loss.item())
  return loss_history


# Part 6: Train Models and Plot Loss For Both Models

# Training parameters
learning_rate = 0.001
epochs = 1

# Train model without BatchNormalization
model_no_bn = ModelNoBN()
criterion = nn.CrossEntropyLoss()
optimizer_no_bn = optim.Adam(model_no_bn.parameters(), lr=learning_rate)
loss_history_no_bn = train(model_no_bn, train_loader, optimizer_no_bn, criterion, epochs)

# Train model with Custom BatchNorm
model_custom_bn = ModelWithBN()
optimizer_custom_bn = optim.Adam(model_custom_bn.parameters(), lr=learning_rate)
loss_history_custom_bn = train(model_no_bn, train_loader, optimizer_custom_bn, criterion, epochs)

# Plot loss curves
plt.plot(loss_history_no_bn, label="Without BatchNorm")
plt.plot(loss_history_custom_bn, label="With Custom BatchNorm")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("MNIST - Training Loss Comparison (Custom BatchNorm)")
plt.savefig("MNIST - Training Loss Comparison (Custom BatchNorm).png", dpi=300)
plt.show()

n = 20
first_n_Epoch_no_NB = loss_history_no_bn[:n]
first_n_Epoch_custom_NB = loss_history_custom_bn[:n]

# Plot loss curves
plt.plot(first_n_Epoch_no_NB, label="Without BatchNorm")
plt.plot(first_n_Epoch_custom_NB, label="With Custom BatchNorm")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("MNIST - Training Loss Comparison First " + str(n) +" (Custom BatchNorm)")
plt.savefig("MNIST - Training Loss Comparison First " + str(n) +" (Custom BatchNorm).png", dpi=300)
plt.show()

# END #

# Stop Spark session
spark.stop()
