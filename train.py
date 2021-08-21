# -*- coding: utf-8 -*-
"""
Created on Fri May 28 00:53:59 2021

@author: iftik
"""

# importing libraries
import nltk
from nltk import punkt , wordnet
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
#import pandas
import json 
import pickle 

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

import random

# Initializing lists
words = []
classes = []
documents = []
ignored_words=['-','%','@','#','&','!','$','?','.','(',')','1','2',"'",'/',]
responses=[]

# loading json file 
data_file=open('D:/AprojectIfti/ChatBot Project/intents.json', encoding="utf8").read()
intents= json.loads(data_file)
#loop for entering into the json file " intents "
for intent in intents["intents"]:
    #By this loop we get into the patterns inside intents
    for pattern in intent["patterns"]:
      
        #bricking the patterns into single word
        word=nltk.word_tokenize(pattern)
        
        #it takes the word from word list of tokenized words and make lemma of it
        stem_word=[stemmer.stem(w.lower()) for w in word if w not in ignored_words  ]
        #adding the lemmatized word into list words 
        words.extend(stem_word)
        #append the lemma & thier tag in documents
        documents.append((stem_word,intent['tag']))
        # adding classes to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    for response in intent['responses']:
            responses.append(response)            
            
        
#here i am making the list of sort and unique words ,removing duplicate and same for classes
words=sorted(list(set(words)))

classes= sorted(list(set(classes)))
#saving words & classes in pkl file
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


# create training data
training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output = list(output_empty)
    output[classes.index(doc[1])] = 1

    training.append([bag, output])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)
# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(200, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y),
                 epochs=330, batch_size=13, verbose=1)


model.save('chatbot_model.h5', hist)

print("model created")
