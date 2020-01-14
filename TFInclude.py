import tensorflow as tf
#import keras
#import keras.layers
#import keras.backend as K
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D, SeparableConvolution2D
#from keras.utils import np_utils
#from keras.models import model_from_yaml
#import keras.regularizers
#from keras.layers.normalization import BatchNormalization
#import keras.callbacks
#from keras.regularizers import WeightRegularizer, RankRegularizer
#import keras.utils.visualize_util
#import keras.applications

import numpy as np
import scipy
import scipy.io
import scipy.linalg
import sklearn.decomposition
from sklearn.utils.extmath import randomized_svd

import tempfile
import uuid

import os
os.environ['TF_CUDNN_USE_AUTOTUNE']='0'
#1 to disable info, 2 to disable warning, 3 to disable error
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'



