import os
import glob
import tensorflow

from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from utils import split_data, order_test_set, create_generators
from dl_models import streetsigns_model


if __name__ == '__main__':

    # path_to_data = 'D:/Downloads/gtsr/Train'
    # path_to_save_train = 'D:/Downloads/gtsr/training_data/train'
    # path_to_save_val = 'D:/Downloads/gtsr/training_data/val'
    # split_data(path_to_data, path_to_save_train, path_to_save_val)

    # path_to_images = 'D:/Downloads/gtsr/Test'
    # path_to_csv = 'D:/Downloads/gtsr/Test.csv'
    # order_test_set(path_to_images, path_to_csv)

    path_to_train = 'D:/Downloads/gtsr/training_data/train'
    path_to_val = 'D:/Downloads/gtsr/training_data/val'
    path_to_test = 'D:/Downloads/gtsr/Test'
    path_to_save_model = './Models'
    batch_size = 256
    epochs = 15
    TRAIN = False

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)

    if TRAIN:

        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )
        
        early_stop = EarlyStopping(monitor = 'val_accuracy', patience=10)

        model = streetsigns_model(train_generator.num_classes)

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        model.fit(train_generator,
                workers=8, # workers helped to speed up the training by orders of magnitude
                epochs = epochs,
                batch_size = batch_size,
                validation_data = val_generator,
                callbacks = [ckpt_saver, early_stop]
        )

    else:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print('Evaluating validation set:')
        model.evaluate(val_generator)

        print('Evaluating test set:')
        model.evaluate(test_generator)
