import json

from nltk_utils import tokenize, stem, bagOfWords
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import  NeuralNetwork


with open('intents.json', 'r') as f:
    intents = json.load(f)

allWords = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        allWords.extend(w)
        xy.append((w, tag))

ignoreWords = ['?', ':', '.', ',', "'", '"', '!']
allWords = [stem(w) for w in allWords if w not in ignoreWords]
allWords = sorted(set(allWords))
tags = sorted(tags)


xTrain = []
yTrain = []

for [patternSentence, tag] in xy:
    bag = bagOfWords(patternSentence, allWords)
    xTrain.append(bag)

    label = tags.index(tag)
    yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(xTrain)
        self.xData = xTrain
        self.yData = yTrain

    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx]

    def __len__(self):
        return self.n_samples


# hyperparameters
batchSize = 8
hiddenSize=8
num_epochs=1000
learning_rate=0.001
inputSize=len(allWords)
outputSize=len(tags)


dataset = ChatDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=NeuralNetwork(inputSize, hiddenSize, outputSize)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data={
    "model-state": model.state_dict(),
    "input-size": inputSize,
    "output-size":outputSize,
    "hidden-size": hiddenSize,
    "all-words": allWords,
    "tags": tags
}

file="data.pth"
torch.save(data, file)

print(f'training complete. File saved to {file}')

