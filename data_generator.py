# MLP for the IMDB problem
# CNN for the IMDB problem
import re 
import numpy as np
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.layers import Dropout
print_info = False

vocabulary_size = 5000
max_words = 50

X = list(np.genfromtxt("sentiment140.csv", delimiter="\",\"", skip_header=1, usecols=(5), dtype=str, max_rows=500000))
Y = np.genfromtxt("sentiment140.csv", delimiter=",", skip_header=1, usecols=(0), dtype=str, max_rows= 500000)
    
for i in range(Y.shape[0]):
    Y[i] = (Y[i][1])
Y = Y.astype(int)
Y = Y//4
for i in range(len(X)):
    X[i] = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", X[i][:-1]).split()).lower()
    X[i] = X[i].split(" ")

word2id = defaultdict(int)
for tweet in X:
    for word in tweet:
        word2id[word] += 1

order = sorted(word2id.items(), key=lambda x:-x[1])
print(order[0], order[-1])
for i in range(len(order)):
    word2id[order[i][0]] = i
for i in range(len(X)):
    X[i] = [word2id[word] if word2id[word] < 5000 else 0 for word in X[i]]
print(X[0])

print('Maximum tweet length: {}'.format(len(max((X), key=len))))
print('Minimum tweet length: {}'.format(len(min((X), key=len))))

X = sequence.pad_sequences(X, maxlen=max_words)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.1, shuffle=False)

if print_info:
    print("Xtr shape",Xtr.shape)
    print("Xte shape",Xte.shape)
    print("Ytr shape",Ytr.shape)
    print("Yte shape",Yte.shape)
    print("First 5 Y vals:",Y[:5])
    print("Y==0 shape (neg):",Y[Y==0].shape) # Y == 0 is negative
    print("Y==2 shape (neu):",Y[Y==2].shape) # Y == 2 is neutral
    print("Y==4 shape (pos):",Y[Y==4].shape) # Y == 4 is positive
    with open("sentiment140.csv") as fname:
        fname.readline()
        print("First 5 lines")    
        for i in range(5):
            print(fname.readline(),end="")
            print(Y[i], X[i])
            print("----------------")

NN = []
max_layers = 20
num_layers = range(0, max_layers + 1)



for layers in num_layers:
	NN.append(Sequential())
	NN[layers].add(Embedding(vocabulary_size, 32, input_length=max_words))
	NN[layers].add(Flatten())
	
	for _ in range(layers):
		NN[layers].add(Dense(250, activation='relu'))
	
	NN[layers].add(Dense(1, activation='sigmoid'))
	NN[layers].compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for layers in num_layers:
	print()
	print()
	print("Now training the neural network with ", layers, " layers")
	print()
	
	NN[layers].fit(Xtr, Ytr, validation_data=(Xte, Yte), epochs=2, batch_size=128, verbose=2)
	nn_scores = NN[layers].evaluate(Xte, Yte, verbose=0)
	print("NN Accuracy: {:.2f}".format(nn_scores[1]*100))

'''
rnn = Sequential()
rnn.add(Embedding(vocabulary_size, 32, input_length=max_words))
rnn.add(Flatten())

rnn.add(Dense(250, activation='sigmoid'))

rnn.add(Dense(250, activation='sigmoid'))

rnn.add(Dense(1, activation='sigmoid'))
print(rnn.summary())
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


rnn.fit(Xtr, Ytr, validation_data=(Xte, Yte), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
rnn_scores = rnn.evaluate(Xte, Yte, verbose=0)

print("RNN Accuracy: {:.2f}".format(rnn_scores[1]*100))
'''


