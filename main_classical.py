import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import make_dataset

model = keras.Sequential([
    keras.Input(shape=(2)),
    layers.Dense(2, activation=keras.activations.relu),
    layers.Dense(1, activation=keras.activations.sigmoid),
])

model.compile(loss=keras.losses.mean_squared_error,
              optimizer='adam',
              metrics=[keras.metrics.binary_accuracy])

model.summary()

x, t = make_dataset()
model.fit(x, t, nb_epoch=500, verbose=2)

# x = tf.Variable([[1], [1]])
# y = model(x)
