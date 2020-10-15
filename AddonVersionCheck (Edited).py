# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 00:05:57 2020

@author: ashleigh - edited by Chris
"""

# Print Versions Used
print('|----------------------------------------------')
print('|     Base Python')
print('|------------|---------------------------------')
    #Python
import platform
print('|Python      |{}'.format(platform.python_version()))
    #scipy
import scipy
print('|scipy       |{}'.format(scipy.__version__))
    #numpy
import numpy
print('|numpy       |{}'.format(numpy.__version__))
    #matplotlib
import matplotlib
print('|matplotlib  |{}'.format(matplotlib.__version__))
    #pandas
import pandas
print('|pandas      |{}'.format(pandas.__version__))
    #sklearn
import sklearn
print('|sklearn     |{}'.format(sklearn.__version__))
    # pytorch
import torch
print('|PyTorch     |{}'.format(torch.__version__))
    # OpenCV
import cv2
print('|OpenCV      |{}'.format(cv2.__version__))
print('|----------------------------------------------')
print('|     Base Python')
print('|------------|---------------------------------')
# import tensorflow as tf
# print('|TensorFlow  |{}'.format(tf.___version__))
# import keras
# print('|Keras  |{}'.format(keras.___version__))

import tensorflow
print('|TensorFlow  |',tensorflow.__version__)
import keras
print('|Keras       |',keras.__version__)
