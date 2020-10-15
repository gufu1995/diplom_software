# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:27:27 2020

@author: Gunnar
"""

import tensorflow as tf
import numpy as np
from keras.models import Input, Model
from keras.layers import MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array, array_to_img

image_input = Input( shape = ( 500, 500, 1 ) )
pooling_layer = MaxPooling2D( pool_size = ( 2, 2 ) )( image_input )

model = Model( inputs = [ image_input ], outputs = [ pooling_layer ] )

model.compile( loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'] )

test_image = load_img( "png/1.png", grayscale = True, color_mode = "L" )
test_image.show( )
test_array = img_to_array( test_image ) / 255.0
test_array = 1.0 - test_array
test_array = np.expand_dims( test_array, 0 )


new_array = model.predict( test_array )[ 0 ]
new_image = array_to_img( new_array )
new_image.show( )