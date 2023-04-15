import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from classes.cornea import CorneaReader
cr = CorneaReader()

eyesMetrics, frames, y = cr.preProcess('initTest')

convInput = tf.keras.layers.Input(shape=(40,120, 1), name="frames")
denseInput = tf.keras.layers.Input(shape=(33), name='eyesMetrics')

x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=convInput.shape)(convInput)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(12, (3,3), activation='relu', input_shape=x.shape)(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.concatenate([x, denseInput])
x = tf.keras.layers.Dense(160, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='relu', name="mousePos")(x)

model = tf.keras.Model(inputs=[convInput, denseInput], outputs=output, name='nesqafe2x1')
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.summary()

model.fit(
    {"eyesMetrics": eyesMetrics, "frames": frames},
    {"mousePos": y},
    epochs=100,
    verbose=1,
    batch_size=24
)


# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)