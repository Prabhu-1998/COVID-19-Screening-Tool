#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:45:52 2020

@author: prabhu
"""
import nltk
#import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

f=open('covid.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you. Want to try something else?"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response      
    
def main():
    
    flag=True
    print("Are you experiencing any of these symptoms?")
    print(" 1. Fever or sweating \n 2. Difficulty breathing \n 3. Prolonged Cough \n 4. Sore throat \n 5. Body ache \n 6. Vomiting and Diarrhea")
    while(flag==True):
        user_response = input()
        #user_response=user_response.lower()
        if(user_response!='4' and user_response!='5' and user_response!='6'):
            if(user_response=='1' or user_response=='2' or user_response=='3'):
                print("Med-ROBO: Have you travelled internationally in last 14 days? Yes or No")
                user_response = input()
                if(user_response=='yes'):
                    flag=False
                    print("Med-ROBO: Consult Doctor Immediately..CORONA (COVID 19) HELPLINE: 011-23978046 OR 1075")
                else:
                    print("Take general tablets, wait for some more days. If symptoms persists, Contact Doctor")
                    print("Med-ROBO: Any more queries on COVID-19, please ask me")
                    user_response = input()
                    print("Med-ROBO: ",end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
            else:
                if(greeting(user_response)!=None):
                    print("Med-ROBO: "+greeting(user_response))
                else:
                    print("Med-ROBO: ",end="")
                    print(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            print("Med-ROBO: No need to worry. Take Home Remedies. Stay Home.. take care..\n")
            print("Med-ROBO: Any more queries on COVID-19, please ask me")
            user_response = input()
            print(response(user_response))
            sent_tokens.remove(user_response)
    
    
if __name__ == "__main__":
    main()
