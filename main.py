

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import getopt, sys
from torch.autograd import Variable


# In[85]:


num_digits=10
BATCH_SIZE=128


# In[86]:

def usage():
  print ("\nThis is the usage function\n")
  print ('Usage: '+sys.argv[0]+' -i <file1> [option]')

def software1(filename):
    a,test_input = input_binary(filename)
    output = test_input
    for i in range(len(test_input)):
        if(test_input[i]%15 == 0):
            output[i] = 'fizzbuzz'
        elif(test_input[i]%5 == 0):
            output[i] = 'buzz'
        elif(test_input[i]%3 == 0):
            output[i] = 'fizz'
    with open('Software1.txt', 'w') as f:
        for item in output:
            f.write("%s\n" % item)       

def software2(filename):
    net = pickle.load(open("Model/software2","rb"))
    test_input,raw_input_form= input_binary(filename)
    pred_output=np.argmax(net(test_input).detach().numpy(),axis=1)
    test_output = [str(i) for i in pred_output]
    for i in range(len(pred_output)):
        if(pred_output[i]==0):
            test_output[i] = raw_input_form[i]
        elif(pred_output[i]==1):
            test_output[i] = "fizzbuzz"
        elif(pred_output[i]==2):
            test_output[i] = "buzz"
        elif(pred_output[i]==3):
            test_output[i] = "fizz"
    with open('Software2.txt', 'w') as f:
        for item in test_output:
            f.write("%s\n" % item) 


def input_binary(file_name):
    train_input=[]
    raw_input_form=[]
    with open(file_name) as file_object:
        for line in file_object:
            line=line.rstrip("\n")
            raw_input_form.append(int(line))
            train_input.append(np.array([int(line) >> (9-d) & 1 for d in range(num_digits)]))
    return torch.tensor(train_input, dtype=torch.float),raw_input_form



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

def main(argv):
    try:
        opts, args = getopt.getopt(argv, '', ['help', 'test-data='])
        if not opts:
            print ('No options supplied')
            usage()
    except getopt.GetoptError as e:
        print (e)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        else:
            software1(arg)
            software2(arg)
        print("generated both the software1.txt and the software2.txt files..")
    
    
    

if __name__ =='__main__':
    main(sys.argv[1:])


