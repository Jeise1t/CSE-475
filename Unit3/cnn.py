# import torch and other necessary modules from torch
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
...
# import torchvision and other necessary modules from torchvision
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# recommended preprocessing steps: resize to square -> convert to tensor -> normalize the image
# if you are resizing, 100 is a good choice otherwise GradeScope will time out
# you could use Compose (https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) from transforms module to handle preprocessing more conveniently


preprocess = transforms.Compose([
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# thanks to torchvision, this is a convenient way to read images from folders directly without writing datasets class yourself (you should know what datasets class is as mentioned in the documentation)
dataset = datasets.ImageFolder("./petimages", transform=preprocess)

#generator for random permutation of split
generator = torch.Generator().manual_seed(42)

# now we need to split the data into training set and evaluation set
# use 20% of the dataset as test
test_set, train_set = torch.utils.data.random_split(dataset, [0.2, 0.8], generator)

# model hyperparameter
learning_rate = 0.01
batch_size = 500
epoch_size = 10


# Removed because it was taken care of by the random split
# n_test = len(dataset)*10
# test_set = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
# train_set = torch.utils.data.Subset(dataset, range(n_test, len(dataset)))  # take the rest
# trainloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = TRUE)  
# testloader = torch.utils.data.Dataloader(test_set, batch_size = batch_size, shuffle = FALSE) 

# prepare dataloader for training set and evaluation set
trainloader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True) 
testloader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False) 



# model design goes here
class CNN(nn.Module):

    # there is no "correct" CNN model architecture for this lab, you can start with a naive model as follows:
    # convolution -> relu -> pool -> convolution -> relu -> pool -> convolution -> relu -> pool -> linear -> relu -> linear -> relu -> linear
    # you can try increasing number of convolution layers or try totally different model design
    # convolution: nn.Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    # pool: nn.MaxPool2d (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
    # linear: nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)
        self.fc1 = nn.Linear(64*10*10, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether your device has GPU
cnn = CNN().to(device) # move the model to GPU
# search in official website for CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = optim.Adam(cnn.parameters(), lr = learning_rate)


# start model training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size): # begin with trying 10 epochs

    loss = 0.0 # you can print out average loss per batch every certain batches
    i = 0
    for data in trainloader:
        # get the inputs and label from dataloader
        inputs,labels = data[0], data[1]
        
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # zero the parameter gradients using zero_grad()

        outputs = cnn(inputs) # forward -> compute loss -> backward propagation -> optimize (see tutorial mentioned in main documentation)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print some statistics
        loss += loss.item()# add loss for current batch
        if i > 100:    # print out average loss every 100 batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0
            i=0
        i+=1

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() # turn on evaluation model, also a good practice to do
with torch.no_grad(): # since we're not training, we don't need to calculate the gradients for our outputs, so turn on no_grad mode
    for data in testloader:
        inputs, labels = data[0], data[1]
        inputs = inputs.to(device)
        ground_truth += labels.tolist()# convert labels to list and append to ground_truth
        # calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        # the class with the highest logit is what we choose as prediction
        _, predicted = torch.max(outputs,1)
        prediction += predicted.tolist() # convert predicted to list and append to prediction


# GradeScope is chekcing for these three variables, you can use sklearn to calculate the scores
from sklearn.metrics import accuracy_score, recall_score, precision_score
accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='weighted')
precision = precision_score(ground_truth, prediction, average='weighted')

print("accuracy:", accuracy)
print("recall:", recall)
print("precision:", precision)
