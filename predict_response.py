import nltk
import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing import get_stem_words

model=tensorflow.keras.models.load_model("chatbot_model.h5")

intents=json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl','rb'))
classes=pickle.load(open('./classes.pkl','rb'))

def preprocess_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input) #["how are you?"] ["how","are","you","?"]
    input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2))) #are how you

    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_prediction(user_input):

