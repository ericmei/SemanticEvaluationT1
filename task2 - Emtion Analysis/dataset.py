import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import collections
import re
import math
from torchvision import datasets, transforms
from collections import Counter
from collections import deque
#from sklearn.model_selection import StratifiedShuffleSplit
import os
import operator
import spacy
import en_core_web_sm

#This file uses Spacy-- use pip install for installation. instructions at https://github.com/explosion/spaCy#updating-spacy
#trimLenMultiple is used to trim the data to be a nice length for training
class MyData(Dataset):
    def __init__(self, path, train, trimLenMultiple):
        wholeText = ""
        wordCounter = Counter()
        nlp = en_core_web_sm.load()
        self.word2idx = {'UNK': 0, 'PAD': 1} # UNK is for unknown word
        self.emotionDict = {'anger': 0, 'fear': 1, 'joy': 2, 'sadness': 3}

        vectorizedTweets = [] #2-D matrix of tweets
        tweetEmotions = []

        #take all files in the folder and train/test on it
        for filename in os.listdir(path):
            print(path + filename)

            with open(path + filename, "rt", encoding="utf8") as f:
                for line in f.readlines()[1:]:
                    id, *tweet, affectDimension, intensityScore = line.strip().split()
                    currentTweet = " ".join(tweet)
                    tokenizedTweet = nlp(currentTweet)

                    currentTweetSplit = currentTweet.split()
                    processedTweet = combine_hashtags(tokenizedTweet)
                    add_to_vocab(processedTweet, self.word2idx)
                    currentSentenceVectorized = get_word_indices(processedTweet, self.word2idx)

                    vectorizedTweets.append(currentSentenceVectorized)
                    tweetEmotions.append([self.emotionDict[affectDimension]])

        maxTweetLen = get_max_tweet_length(vectorizedTweets)
        pad_sentences(maxTweetLen, vectorizedTweets, self.word2idx)

        nearestTrimSize = math.floor(len(vectorizedTweets) / trimLenMultiple) * trimLenMultiple
        vectorizedTweets = vectorizedTweets[:nearestTrimSize]

        self.longestTweetLen = maxTweetLen
        self.x_data = torch.LongTensor(vectorizedTweets) #all sentences
        self.y_data = torch.LongTensor(tweetEmotions) #all sentiments related to these sentences

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)

    def __getWordDict__(self):
        return self.word2idx

def combine_hashtags(tokenizedTweet):
    allWords = []
    currentWordIsHashTag = False

    for index in range(len(tokenizedTweet)):
        if currentWordIsHashTag != True and tokenizedTweet[index].text == '#' and index != len(tokenizedTweet) - 1:
            allWords.append('#' + tokenizedTweet[index + 1].text)
            currentWordIsHashTag = True
            #print(("The hashtag value is: {}".format('#' + tokenizedTweet[index + 1].text)).encode('utf-8'))
        elif currentWordIsHashTag == True:
            currentWordIsHashTag = False
        else:
            allWords.append(tokenizedTweet[index].text)

    return allWords

#turn the words into indices
def get_word_indices(tokenizedSentence, dictionary):
    allWords = [] #list of mapped words to index in dictionary

    for word in tokenizedSentence:
        if (word in dictionary):
            allWords.append(dictionary[word])
        else:
            allWords.append(dictionary['UNK'])
    return allWords

def get_max_tweet_length(tweets):
    max = -1
    for tweet in tweets:
        if len(tweet) > max:
            max = len(tweet)

    return max

#sentencesToPad- list of all sentences to pad
def pad_sentences(longestSentenceLen, sentencesToPad, vocabulary):
    padWord = "PAD"

    for sentenceIndex in range(len(sentencesToPad)):
        currentSentenceLen = len(sentencesToPad[sentenceIndex])

        padSentenceLen = longestSentenceLen - currentSentenceLen

        for i in range(padSentenceLen):
            (sentencesToPad[sentenceIndex]).append(vocabulary[padWord])

#if word not in vocab, add to vocabulary
def add_to_vocab(allWords, word2idx):
    for word in allWords:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

#takes in a list of words and merges the hashtags
#E.g. ['we', 'win', '#', 'winners'] -> ['we', 'win', '#winners']
def createTweetsWithMergedHashTags(tweet):
    newTweet = []
    index = 0

    while index < len(tweet):
        if tweet[index] == '#' and (index != len(tweet) - 1):
            newTweet.append(tweet[index] + tweet[index + 1])
            index = index + 2
        else:
            newTweet.append(tweet[index])
            index = index + 1

    return newTweet
