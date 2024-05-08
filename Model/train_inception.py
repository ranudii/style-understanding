import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def feature_extraction_InV3(img_width, img_height, train_data_dir, num_image):
    base_model = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=15,
        class_mode="categorical",
        shuffle=False)

    y_train = train_generator.classes
    y_train1 = np.zeros((num_image, 4))
    y_train1[np.arange(num_image), y_train] = 1

    train_generator.reset()
    X_train = model.predict(train_generator, verbose=1)
    print(X_train.shape, y_train1.shape)
    return X_train, y_train1


def train_last_layer(img_width, img_height, train_data_dir, num_image, epochs=50):
    X_train, y_train = feature_extraction_InV3(img_width, img_height, train_data_dir, num_image)
    my_model = Sequential([
        BatchNormalization(input_shape=X_train.shape[1:]),
        Dense(1024, activation="relu"),
        Dense(4, activation='softmax')
    ])

    my_model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    my_model.fit(X_train, y_train, epochs=epochs, batch_size=30, verbose=1)
    return my_model


if __name__ == "__main__":
    img_width = 299
    img_height = 299
    train_data_dir = "../image_data"
    num_image = 1800
    epochs = 10
    model = train_last_layer(img_width, img_height, train_data_dir, num_image, epochs)
    model.save('inV3_last_layer.h5')
