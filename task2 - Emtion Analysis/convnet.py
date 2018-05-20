import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ConvNet(torch.nn.Module):
    def __init__(self, vocab_size, longestTweetLen, embedding_dim):
        super(ConvNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #self.conv_drop = nn.Dropout(p = 0.2)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = longestTweetLen, out_channels = embedding_dim, kernel_size=9, stride=5),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = embedding_dim, out_channels = embedding_dim, kernel_size=9, stride=5),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=3)
        )

        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, inputs):
        in_size = inputs.size(0)
        embeds = self.embedding(inputs)

        #out = F.dropout(out, training=self.training)
        out = self.conv1(embeds)
        out = self.conv2(out)
        out = out.view(50, -1)
        out = self.linear(out)
        return out
