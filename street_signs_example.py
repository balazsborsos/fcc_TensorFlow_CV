import os
import glob
from sklearn.model_selection import train_test_split
import shutil

from utils import split_data, order_test_set


if __name__ == '__main__':

    # path_to_data = 'D:/Downloads/gtsr/Train'
    # path_to_save_train = 'D:/Downloads/gtsr/training_data/train'
    # path_to_save_val = 'D:/Downloads/gtsr/training_data/val'

    # split_data(path_to_data, path_to_save_train, path_to_save_val)

    path_to_images = 'D:/Downloads/gtsr/Test'
    path_to_csv = 'D:/Downloads/gtsr/Test.csv'

    order_test_set(path_to_images, path_to_csv)