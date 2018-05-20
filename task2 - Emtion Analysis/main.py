import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import dataloader
from convnet import ConvNet
from torch.autograd import Variable

def main():
    train_loader, dataset = dataloader.generateLoadedData()
    model = ConvNet(len(dataset.word2idx), dataset.longestTweetLen, 50)

    for epoch in range(1, 10):
        train(epoch, model, train_loader)
        #test(model, train_loader)

def train(epoch, model, train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(torch.LongTensor(data))
        target = Variable(torch.LongTensor(target.squeeze(1)))
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    print("Total loss is: {}".format(total_loss/train_loader.__len__()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(torch.LongTensor(data), volatile=True),Variable(torch.LongTensor(target.squeeze(1)))
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target,
        size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    main()
