import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import argparse

from models import *
from utils import progress_bar
import matplotlib.pyplot as plt

## Load data
print ("=============== Load Data ===============")

X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")
print(X_train.shape)
print(Y_train.shape)
# Train-validate split
# random shuffle
x_train = np.zeros((45000, 32, 32, 3))
y_train = np.zeros((45000, 1))
x_val = np.zeros((5000, 32, 32, 3))
y_val = np.zeros((5000, 1))
np.random.seed(2)
arr = np.arange(50000)
np.random.shuffle(arr)
for i in range(45000):
    num = arr[i]
    x_train[i] = X_train[num]
    y_train[i] = Y_train[num]
for i in range(5000):
    num = arr[i+4999]
    x_val[i] = X_train[num]
    y_val[i] = Y_train[num]

print(x_train[-1])
print(y_train[-1])
print(x_val[-1])
print(y_val[-1])

#non-random shuffle
# x_train, x_val = x_train[:45000,:], x_train[45000:,:]
# y_train, y_val = y_train[:45000,:], y_train[45000:,:]

x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')

# Transform to DataLoader
x_train_tensor = torch.Tensor(x_train).permute(0, 3, 1, 2)
x_val_tensor = torch.Tensor(x_val).permute(0, 3, 1, 2)
x_test_tensor = torch.Tensor(x_test).permute(0, 3, 1, 2)
y_train_tensor = torch.Tensor(y_train).view(-1)
y_val_tensor = torch.Tensor(y_val).view(-1)
y_test_tensor = torch.Tensor(y_test).view(-1)
y_train_tensor = y_train_tensor.type(torch.LongTensor)
y_val_tensor = y_val_tensor.type(torch.LongTensor)
y_test_tensor = y_test_tensor.type(torch.LongTensor)
trainloader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64)
valloader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=1)
testloader = DataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=1)


# It's a multi-class classification problem 
class_index = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
               'dog': 5, 'frog': 6,'horse': 7,'ship': 8, 'truck': 9}
print(np.unique(y_train))


## Build Model (pytorch)
print ('============ Environment Setting ==========')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print ('Environment done.')

print ('================ Build Model =================')
net = ResNet50()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,400,450], gamma=0.1, last_epoch=-1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    hist_train.append(train_loss/(batch_idx+1))
    
    return hist_train
        
# Validating
def valid(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        hist_val.append(val_loss/(batch_idx+1))
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = { 
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        
    return hist_val

# Testing
def test():
    # global best_acc
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    print ("\nfinal acc: ", correct/total)

hist_train = []
hist_val = []
for epoch in range(start_epoch, start_epoch+200):
    hist_train = train(epoch)
    hist_val = valid(epoch)
    scheduler.step()
    
    #Plot loss chart
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(hist_val,label="val")
    plt.plot(hist_train,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("Loss Chart.png")

test()













# # In[10]:


# y_pred = model.predict(x_test)
# print(y_pred.shape)


# # In[11]:


# y_pred[0]


# # In[12]:


# np.argmax(y_pred[0])


# # In[13]:


# y_pred = np.argmax(y_pred, axis=1)


# # ## DO NOT MODIFY CODE BELOW!
# # please screen shot your results and post it on your report

# # In[14]:


# assert y_pred.shape == (10000,)


# # In[15]:


# y_test = np.load("y_test.npy")
# print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))


# # In[ ]:




