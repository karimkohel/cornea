import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from classes.cornea import CorneaReader
cr = CorneaReader()

eyesMetrics, frames, y = cr.preProcess('realTest')

convInput = tf.keras.layers.Input(shape=(40,120, 1), name="frames")
denseInput = tf.keras.layers.Input(shape=(33), name='eyesMetrics')

x1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=convInput.shape)(convInput)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=x.shape)(x1)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Flatten()(x1)

x2 = tf.keras.layers.Dense(80, activation='relu')(denseInput)

x = tf.keras.layers.concatenate([x1, x2])
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(60, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='relu', name="mousePos")(x)

model = tf.keras.Model(inputs=[convInput, denseInput], outputs=output, name='nesqafe2x1')
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.fit(
    {"eyesMetrics": eyesMetrics, "frames": frames},
    {"mousePos": y},
    epochs=15,
    verbose=1,
    batch_size=24,
    validation_split=0.1
)

model.save("models/convModelTest1.h5")


# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)