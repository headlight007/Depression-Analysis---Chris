# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:47:53 2020

@author: moore

Work in progress - see end for references required moving forward..
"""

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class DepressionDataset(Dataset):
     

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        vals = annotations[idx]   
        video_name = '204_1_cut_combined.mp4'
        frame_id = 0
        level = 1
        
        image = io.imread(img_name)
         
        sample = {'image': image, 'level': level}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epochs, trainloader, optimizer, net, criterion):
    net.to(device)
    net.train()
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    '''
    print('Finished Training')
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    net = Net()
    net.load_state_dict(torch.load(PATH))
    
    outputs = net(images)
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    
    ''' 
if __name__ == '__main__':
     print  (torch.__version__)
     #torch.multiprocessing.freeze_support()
     #backbone = torchvision.models.mobilenet_v2(pretrained=True)
     net = Net()
     criterion = nn.CrossEntropyLoss()
     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
     
     transform = transforms.Compose(
     [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     root_directory = (r"D:\\OneDrive\University\2020 S2\9785 ITS Capstone Project Semester 2 2020\Projects\AVEC2013\Development")
     trainset = DepressionDataset(root_dir=root_directory)
     
     #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
     
     trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=False, num_workers=2)
     
     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
     
     # Detect if we have a GPU available
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     epochs = 1
     # get some random training images
     dataiter = iter(trainloader)
     images, labels = dataiter.next()
      
     # show images
     #imshow(torchvision.utils.make_grid(images))
     # print labels
     #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
     
     
     
     
     
     
     
     train( epochs, trainloader, optimizer, net, criterion)
     
     '''
    
     PATH = './cifar_net.pth'
     torch.save(net.state_dict(), PATH) 
     
     
     
     
     
     #added to resolve Spyder error
     __spec__ = None
     
     classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
     
    ''' 
    
    #useful resources moving forward:
        #https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
        #https://www.geeksforgeeks.org/python-os-path-join-method/
        #https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#finetuning-the-convnet
        #https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=writing%20custom
        