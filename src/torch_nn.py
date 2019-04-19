#!/usr/bin/env python3
# coding: utf-8
"""
neural_net.py
04-03-19
jack skrable
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

def torch_deep_nn(X, y):

    # Convert target to categorical
    # y = to_categorical(y, num_classes=np.unique(y).size)

    # n_input, n_output = X.shape[1], np.unique(y).size
    # n_hidden_1, n_hidden_2 = X.shape[1], X.shape[1] // 2

    # # Get input and output layer sizes from input data
    # in_size = X.shape[1]
    # # Modify this when increasing artist list target
    # out_size = np.unique(y).size

    # Split up input to train/test/validation
    print('Splitting to train, test, and validation sets...')
    X_train, X_test, X_valid = np.split(X, [int(.6 * len(X)), int(.8 * len(X))])
    y_train, y_test, y_valid = np.split(y, [int(.6 * len(y)), int(.8 * len(y))])


    # Create ANN Model
    class ANNModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(ANNModel, self).__init__()
            # Linear function 1: input --> hidden1
            self.fc1 = nn.Linear(input_dim, hidden_dim) 
            # Non-linearity 1
            self.relu1 = nn.ReLU()
            
            # Linear function 2: hidden1 --> hidden1
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            # Non-linearity 2
            self.tanh2 = nn.Tanh()
            
            # Linear function 3: hidden1 --> hidden1
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            # Non-linearity 3
            self.elu3 = nn.ELU()
            
            # Linear function 4 (readout): hidden1 --> output
            self.fc4 = nn.Linear(hidden_dim, output_dim)  
        
        def forward(self, x):
            # Linear function 1
            out = self.fc1(x)
            # Non-linearity 1
            out = self.relu1(out)
            
            # Linear function 2
            out = self.fc2(out)
            # Non-linearity 2
            out = self.tanh2(out)
            
            # Linear function 2
            out = self.fc3(out)
            # Non-linearity 2
            out = self.elu3(out)
            
            # Linear function 4 (readout)
            out = self.fc4(out)
            return out

    # instantiate ANN
    n_input, n_output = X.shape[1], np.unique(y).size
    n_hidden_1, n_hidden_2 = X.shape[1], X.shape[1] // 2


    # Create ANN
    model = ANNModel(n_input, 200, n_output)

    # Cross Entropy Loss 
    error = nn.CrossEntropyLoss()

    # SGD Optimizer
    learning_rate = 0.02
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)

    # create feature and targets tensor for test set.
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # batch_size, epoch and iteration
    batch_size = 100
    n_iters = 10000
    num_epochs = n_iters / (len(X_train) / batch_size)
    num_epochs = int(num_epochs)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    test = torch.utils.data.TensorDataset(X_test,y_test)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_loader):

            train = Variable(X).float()
            labels = Variable(y).type(torch.LongTensor)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            count += 1
            
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Predict test dataset
                for X, y in test_loader:

                    test = Variable(X).float()
                    
                    # Forward propagation
                    outputs = model(test)
                    
                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]
                    
                    # Total number of labels
                    total += len(y)

                    # Total correct predictions
                    correct += (predicted == y).sum()
                
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))