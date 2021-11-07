# freeCodeCamp TensorFlow for Computer Vision
(Course: https://www.youtube.com/watch?v=cPmjQ9V6Hbk)

## Installation
```bash
git clone https://github.com/balazsborsos/fcc_TensorFlow_CV.git
```

Create a virtual environment (or in your base environment) with Anaconda or any virtual environment manager (see Anaconda example below): 

```bash
conda create -n ffc-cv python=3.8
conda activate ffc-cv
pip install -r requirements.txt
```

## Project description
In this project I practiced how to use TensorFlow with Keras for Computer Vision with building two applications. 

### MNIST
MNIST is considered the "Hello world!" example of computer vision tasks. Here, after exploring the dataset, I've tried out 3 different ways on how to build a neural network with _tf_, namely the sequential, functional and the Model Class way. Then set up the training pipeline and ran evaluation on it, achieveing 98.46% accuracy on the validation set with just 3 epochs.

### [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. Here I also had to prepare the dataset with scripts to have it better organized, and had experience with creating and using data generators. Then built a CNN the functional way with 12 layers to perform the classification. Training the model in 15 epochs resulted in the best model performing 96.28% on the test set. I've also prepared a script that get's this model closer to production, as it just loads the trained model and performs inference on single images. 