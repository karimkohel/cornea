import tensorflow as tf
from classes.cornea import CorneaReader
from sklearn import preprocessing
import numpy as np

cr = CorneaReader()
eyesMetrics, frames, y = cr.loadData('datesetTest')

convInput = tf.keras.layers.Input(shape=(cr.TARGET_IMG_SIZE[0],cr.TARGET_IMG_SIZE[1], 1), name="frames")
denseInput = tf.keras.layers.Input(shape=(cr.FACE_METRICS_LEN), name='eyesMetrics')

CNN_KERNEL_SIZE = (3, 5)

x1 = tf.keras.layers.Conv2D(40, CNN_KERNEL_SIZE, activation='swish', input_shape=convInput.shape)(convInput)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Conv2D(30, CNN_KERNEL_SIZE, activation='swish', input_shape=x1.shape)(x1)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Flatten()(x1)

x2 = tf.keras.layers.Dense(50, activation='tanh')(denseInput)

x = tf.keras.layers.concatenate([x1, x2])
x = tf.keras.layers.Dense(80, activation='swish')(x)
x = tf.keras.layers.Dense(50, activation='swish')(x)
output = tf.keras.layers.Dense(2, activation='relu', name="mousePosition")(x)

model = tf.keras.Model(inputs=[convInput, denseInput], outputs=output, name='multEye')
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.summary()
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir="modelLogs/epochs60_batch32_swishConv_3,5kernel_lessNodes_tanhMetrics_40,30ConvNodes",
    histogram_freq=1,
    update_freq='epoch',
    write_graph=True
)


model.fit(
    {"eyesMetrics": eyesMetrics, "frames": frames},
    {"mousePosition": y},
    epochs=60,
    verbose=2,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard],
    shuffle=True
)

model.save("models/convModelTest8.h5")