import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
#Used to automatically split the data intro training sets for models
from keras.utils import np_utils
#Used for utilities related to numpy, particularly for transforming class labels to one-hot encoding.
from keras.models import Sequential
#Imports a linear stack of layers for building nueral networks layer by layer
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from IPython.display import display, Image
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping

#from keras.modelsimport Sequential: This imports the 'Sequential' class from Keras, which is a linear stack of layers for building neural networks layer by layer.
#from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout: These import various layer types from Keras, which are building blocks for constructing neural networks.
#Dense is a fully connected layer.
#Conv2D is a 2D convolutional layer, commonly used for image processing tasks.
#MaxPooling2D is a max pooling layer, used for downsampling feature maps.
#Flatten is a layer that flattens the input, which is often used before connecting to fully connected layers.
#Dropout is a regularization technique that randomly drops a fraction of connections during training to prevent overfitting.


