
from collections import defaultdict
from keras.layers import Masking
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
NN = []
max_layers = 3
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
 

