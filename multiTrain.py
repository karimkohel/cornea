import tensorflow as tf
from classes.cornea import CorneaReader

cr = CorneaReader()
eyesMetrics, frames, y = cr.loadData('tataHome')

convInput = tf.keras.layers.Input(shape=(cr.TARGET_IMG_SIZE[0],cr.TARGET_IMG_SIZE[1], 1), name="frames")
denseInput = tf.keras.layers.Input(shape=(33), name='eyesMetrics')

x1 = tf.keras.layers.Conv2D(42, (3,3), activation='relu', input_shape=convInput.shape)(convInput)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Conv2D(42, (3,3), activation='relu', input_shape=x1.shape)(x1)
x1 = tf.keras.layers.MaxPool2D()(x1)
x1 = tf.keras.layers.Flatten()(x1)

x2 = tf.keras.layers.Dense(80, activation='relu')(denseInput)

x = tf.keras.layers.concatenate([x1, x2])
x = tf.keras.layers.Dense(120, activation='relu')(x)
x = tf.keras.layers.Dense(80, activation='relu')(x)
output = tf.keras.layers.Dense(2, activation='relu', name="mousePosition")(x)

model = tf.keras.Model(inputs=[convInput, denseInput], outputs=output, name='multEye')
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model.summary()

model.fit(
    {"eyesMetrics": eyesMetrics, "frames": frames},
    {"mousePosition": y},
    epochs=35,
    verbose=1,
    batch_size=24,
    validation_split=0.2
)

model.save("models/convModelTest1.h5")


# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)