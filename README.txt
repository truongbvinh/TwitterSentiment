All of the executable code is abailable with contributions from 
Vinh Truong, Sebastian Dumbrava, and Bryce Blanton.

The dataset is too big to be submitted through canvas, so if retraining is 
desired (already trained models are already included in this folder), then
download the sentiment140.csv publicly available dataset and once unzipped,
resave with encoding: "utf-8 with BOM", shuffle the lines, and run the
model_training.py file.

To train and save the models, the following dependencies are needed:
Scikit-learn, Keras, numpy, Python libs
and run the file, model_training.py
This creates 4 files to be used in Twitter.py, 3 of which are models, and 1
of which is for term-frequency information.

To test different numbers of hidden levels, the following dependencies are needed:
Scikit-learn, Keras, numpy, Python libs
and run the file, data_generator.py


To run the models on the Twitter API, the following dependencies are needed:
Tweepy, Keras, Python libs
and run the file, Twitter.py
This will pull at MOST 100 unique tweets, and run predictions through each
model. Additionally, each model will print out their outputs with the
corresponding tweet.
NOTE: This file uses my personal keys and tokens from the Twitter API, so please
don't share this file outside of the required scope of people