import data
#print("data loaded")
#import os

import tensorflow as tf
from tensorflow import keras
#print(tf.__version__)
import numpy as np

modelpath='models/my_model'

#'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(20, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(6)
])

model.compile(optimizer='adam',
	loss=tf.keras.losses.MeanSquaredError())
'''
model = tf.keras.models.load_model(modelpath)
#'''
testresult=np.array(model(data.train_data))
print(np.average(testresult-data.train_labels)/0.00001)

model.evaluate(data.train_data, data.train_labels, verbose=2)

model.fit(data.train_data, data.train_labels, epochs=30,batch_size=len(data.train_data), verbose=1)
model.evaluate(data.train_data, data.train_labels, verbose=2)

model.save(modelpath)

testresult=np.array(model(data.train_data))
print(np.average(testresult-data.train_labels)/0.00001)

#model.summary()
#'''
