import tensorflow as tf
import keras_cv

images = tf.ones(shape=(1, 512, 512, 3))
labels = {
    "boxes": tf.constant([
        [
            [0, 0, 100, 100],
            [100, 100, 200, 200],
            [300, 300, 100, 100],
        ]
    ], dtype=tf.float32),
    "classes": tf.constant([[1, 1, 1]], dtype=tf.int64),
}

# Create the model with ResNet50 backbone
model = keras_cv.models.YOLOV8Detector.from_preset(
    num_classes=20,
    bounding_box_format="xywh",
    preset="mobilenet_v3_large",
)


# Evaluate model without box decoding and NMS
model(images)

# Prediction with box decoding and NMS
model.predict(images)

# Train model
model.compile(
    classification_loss='binary_crossentropy',
    box_loss='ciou',
    optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
    jit_compile=False,
)
model.fit(images, labels)