#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.autograd import Variable


# In[85]:


num_digits=10
BATCH_SIZE=128


# In[86]:


def input_binary(file_name):
    train_input=[]
    with open(file_name) as file_object:
        for line in file_object:
            line=line.rstrip("\n")
            train_input.append(np.array([int(line) >> (9-d) & 1 for d in range(num_digits)]))
    return torch.tensor(train_input, dtype=torch.float)


# In[87]:


def prepare_output(file_name):
    train_output=[]
    with open(file_name) as file_object:
        for line in file_object:
            line=line.rstrip("\n")
            if line == "fizz":
                train_output.append(3)
            elif line == "buzz":
                train_output.append(2)
            elif line=="fizzbuzz":
                train_output.append(1)
            else:
                train_output.append(0)
    train_output=torch.tensor(train_output, dtype=torch.float)
    return train_output


# In[88]:


def accuracy(test_input_file,test_output_file):
    test_input=Variable(input_binary(test_input_file))
    test_output=Variable(prepare_output(test_output_file)).long().numpy()
    pred_output=np.argmax(net(test_input).detach().numpy(),axis=1)
    acc=(test_output==pred_output)
    return np.mean(acc)


# In[90]:


#print(prepare_output("train_output.txt"))
#print(input_binary("train_input.txt"))
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(10,100)
        #self.fc2 = nn.Linear(12,12)
        self.fc2 = nn.Linear(100,4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
       # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x


# In[97]:


net= NeuralNet()
w = torch.tensor([0.2,2,1,0.6],dtype=torch.float32)
optimizer=torch.optim.Adam(net.parameters(),lr=0.005)
loss_fn= nn.CrossEntropyLoss(weight=w)
x=Variable(input_binary("train_input.txt"))
y=Variable(prepare_output("train_output.txt")).long()
#print(y.shape)
#print(x[800:900,:])


# In[98]:


#print(x)
#print(y)
#print(len(y))
for epoch in range(5000):
    p=np.random.permutation(range(len(x)))
    x,y=x[p],y[p] 
    for start in range(0,len(x),BATCH_SIZE):
        end=start+BATCH_SIZE
        batchx=x[start:end]
        batchy=y[start:end]
        optimizer.zero_grad()
        output_pred=net(batchx)
        #print(output_pred.shape)
        loss=loss_fn(output_pred,batchy)
        loss.backward()
        optimizer.step()
    
    if epoch%100==0:
        #print(net(x).shape)
        loss = loss_fn(net(x),y).item()
        print ('Epoch:', epoch, 'Loss:', loss)
#printing the accuracy for test data..
print(accuracy("test_input.txt","test_output.txt"))
#print(accuracy(crossx,crossy))
#pickle.dump(net,open("software2",'wb'))


# In[ ]:




