# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 00:04:25 2021

@author: iftik
"""

#importing libraries required
import nltk
from nltk.stem import LancasterStemmer
stemmer= LancasterStemmer()
import pickle
import numpy as np

#loading the saved trained model
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

#loading the json dataset 
intents = json.loads(open('D:/AprojectIfti/ChatBot Project/intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#importing sequencaial model from keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


#this function will tokenize and stemm the words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bag_of_word(sentence, words, show_details=True):
    # clean_up_sentence function is called
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    #iterate over sentence_words
    for s_word in sentence_words:
        #getting the word from words list with index
        for i,word in enumerate(words):
            if word == s_word:
                # assign 1 in the same index if word is present in words list
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s_word" % word)
    #convert in np array for feeding into the model
    return(np.array(bag))
#this function will predict the respective class 
def predict_class(sentence, model):
    # calling the bag of word function
    B = bag_of_word(sentence, words, show_details=False)
    #all classes probability will be stored in res
    res = model(np.array([B]))[0]
    #adding threshold value for better result
    ERROR_THRESHOLD = 0.25
    #the classes with probability greater than error_threshold 
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # sort by strength of probability & reverse 
    #why x:x[1] it will sort the list results withrespect to index 1
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    #iterate over results
    for r in results:
# =============================================================================
#         append the class name for r[0] index and probablity of r[1] index from results .let results=[23,0.9778]
#         the 23 number class will be search for 23 index and that class will be get
# =============================================================================
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#getting the response from chatbot      
def getResponse(predict_label, intents_json):
    #the class predicted by predict_class function will be saved in tag
    tag = predict_label[0]['intent']
    #making the list from intents for iteration
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        #the intents.json tag will be compare with predicted tag
        if(i['tag']== tag):
            #if tags are equal the response will be given from same intent
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

def chatbot_response(msg):
    #calling the below two functions
    predict_label = predict_class(msg, model)
    response = getResponse(predict_label, intents)
    return response


from flask import Flask, render_template, request
app=Flask(__name__,template_folder='templates')
app.static_folder = 'static'
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
if __name__ == "__main__":
    app.run()














