import tweepy
import re
import pickle
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import load_model

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
vocabulary_size = 5000
max_words = 50

with open("word2id", "rb") as f:
    word2id = pickle.load(f)

# attempt authentication 
# Let's say this is a web app, so we need to re-build the auth handler
# first...
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
tweets = []

for i, tweet in enumerate(tweepy.Cursor(api.search,q="#Stadia",count=5,lang="en").items(100)):
    
    print("reading", i+1, "\r", end="")
    text = tweet.text.replace("'", "")
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()).lower()
    text = text.split()
    if text[0] == "rt":
        continue
    tweets.append(text)

print()
scraped_X = list()
                
for i in range(len(tweets)):
    scraped_X.append([word2id[word] if word2id[word] < vocabulary_size else 0 for word in tweets[i]])
scraped_X = np.array(scraped_X)

scraped_X = sequence.pad_sequences(scraped_X, maxlen=max_words)

for i in range(5):
    print(scraped_X[i])

model = load_model('Model1.h5')
cnn = load_model('CNNModel.h5')
rnn = load_model('RNNModel.h5')

results = zip(model.predict(scraped_X), cnn.predict(scraped_X), rnn.predict(scraped_X))
print("Feedforward     CNN         RNN")

for i, pred in enumerate(results):
    # prints MODEL1, CNN, TWEET
    print(pred[0], pred[1], pred[2], ' '.join(tweets[i]))