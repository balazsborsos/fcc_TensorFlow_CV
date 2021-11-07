import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_logging_ops import Print

def predict_with_model(model, img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # also does the normalization automatically
    image = tf.image.resize(image, [60,60]) # (60,60,3)
    image = tf.expand_dims(image, axis=0) # (1,60,60,3)

    predictions = model.predict(image) # [0.004, 0.003, 0.93,...]
    predictions = np.argmax(predictions) # index of the max value

    return predictions


if __name__ == '__main__':

    img_path = 'D:/Downloads/gtsr/Test/0/06854.png'
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(prediction)