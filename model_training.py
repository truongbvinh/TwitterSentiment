# MLP for the twitter problem
# CNN for the twitter problem
import re 
import numpy as np
import pickle
from collections import defaultdict
from keras.layers import Masking
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# params
vocabulary_size = 5000
max_words = 50

# Generate dataset
X = list(np.genfromtxt("sentiment140.csv", delimiter="\",\"", skip_header=1, usecols=(5), dtype=str, max_rows=500000))
Y = np.genfromtxt("sentiment140.csv", delimiter=",", skip_header=1, usecols=(0), dtype=str, max_rows= 500000)
    
# Process data, strip tweets
for i in range(Y.shape[0]):
    Y[i] = (Y[i][1])
Y = Y.astype(int)
Y = Y//4
for i in range(len(X)):
    X[i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", X[i][:-1]).split()).lower()
    X[i] = X[i].split(" ")

# Convert words to int based on frequency
word2id = defaultdict(int)
for tweet in X:
    for word in tweet:
        word2id[word] += 1

order = sorted(word2id.items(), key=lambda x:-x[1])
print(order[0], order[-1])
for i in range(len(order)):
    word2id[order[i][0]] = i+1
for i in range(len(X)):
    X[i] = [word2id[word] if word2id[word] < vocabulary_size else 0 for word in X[i]]
print(X[0])

# Tweet length info
print('Maximum tweet length: {}'.format(len(max((X), key=len))))
print('Minimum tweet length: {}'.format(len(min((X), key=len))))

X = sequence.pad_sequences(X, maxlen=max_words)

with open("word2id", "ab") as f:
    pickle.dump(word2id, f)

# Split into test/train
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, shuffle=False)
print(vocabulary_size)

# Create the model
model = Sequential()
w2v = Embedding(vocabulary_size, 32, input_length=max_words)
model.add(w2v)
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create the cnn model
cnn = Sequential()
cnn.add(Embedding(vocabulary_size, 32, input_length=max_words))
cnn.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Flatten())
cnn.add(Dense(250, activation='relu'))
cnn.add(Dense(1, activation='sigmoid'))
cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(cnn.summary())

# Create the rnn model
rnn = Sequential()
rnn_batch_size = 32
rnn.add(Masking(mask_value=0., input_shape=(max_words,)))
rnn.add(Embedding(vocabulary_size, 32, input_length=max_words))
NUM_RNN_LAYERS = 1
rnn.add(CuDNNLSTM(250, return_sequences=True))
rnn.add(CuDNNLSTM(1))
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(rnn.summary())

# Fit and print accuracy
model.fit(Xtr, Ytr, validation_data=(Xte, Yte), epochs=2, batch_size=64, verbose=2)
cnn.fit(Xtr, Ytr, validation_data=(Xte, Yte), epochs=2, batch_size=128, verbose=2)
rnn.fit(Xtr, Ytr, validation_data=(Xte, Yte), epochs=2, batch_size=rnn_batch_size, verbose=1)
# Final evaluation of the model
model_scores = model.evaluate(Xte, Yte, verbose=0)
cnn_scores = cnn.evaluate(Xte, Yte, verbose=0)
rnn_scores = rnn.evaluate(Xte, Yte, verbose=0)
print("Model1 Accuracy: {:.2f}\nCNN Accuracy: {:.2f}\nRNN Accuracy: {:.2f}".format(model_scores[1]*100,cnn_scores[1]*100,rnn_scores[1]*100))

# Save the models for later use
model.save('Model1.h5')
cnn.save('CNNModel.h5')
rnn.save('RNNModel.h5')
