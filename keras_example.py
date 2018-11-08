from __future__ import print_function
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)





# model = tf.keras.Sequential([
# # Adds a densely-connected layer with 64 units to the model:
# layers.Dense(64, activation='relu'),
# # Add another:
# layers.Dense(64, activation='relu'),
# # Add a softmax layer with 10 output units:
# layers.Dense(10, activation='softmax')])

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# import numpy as np

# data = np.random.random((1000, 32))
# labels = np.random.random((1000, 10))

# val_data = np.random.random((100, 32))
# val_labels = np.random.random((100, 10))

# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data=(val_data, val_labels))

# result = model.predict(data, batch_size=32)
# print(result.shape)




#
