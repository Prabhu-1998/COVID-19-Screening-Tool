from flask import Flask, render_template, request
import nltk
#import numpy as np
import random
import string # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

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


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    flag=True
    while(flag==True):
        user_response = request.args.get('msg')
        if(user_response=='1' or user_response=='2' or user_response=='3'):
            return "Med-ROBO: Have you travelled internationally in last 14 days? Yes or No"
        elif(user_response == 'yes'):
            return "Med-ROBO: Consult Doctor Immediately..CORONA (COVID 19) HELPLINE: 011-23978046 OR 1075"
        elif(user_response == 'no' or user_response == '4' or user_response == '5' or user_response == '6'):
            return "Take general tablets, wait for some more days. If symptoms persists, Contact Doctor. <br> Do you have any queries on COVID-19?"
        else:
            return str(response(user_response)) 


if __name__ == '__main__':
    app.run()