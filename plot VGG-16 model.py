# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:56:37 2020

Source: https://towardsdatascience.com/a-demonstration-of-transfer-learning-of-vgg-convolutional-neural-network-pre-trained-model-with-c9f5b8b1ab0a
@author: moore
"""

from keras.applications.vgg16 import VGG16
from keras.utils import  plot_model
model = VGG16()
plot_model(model)