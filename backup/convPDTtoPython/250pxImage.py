# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:11:40 2020

@author: Gunna
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


mRange, angleVec, X, Y = pickle.load( open( "dataset/xyscan_range_angle_x_y.p", "rb" ) )

xMax = 8.0
yMax = 8.0 / 2

imSize = 250

xIncs = np.linspace( 0, xMax, imSize + 1 )
yIncs = np.linspace( -yMax, yMax, imSize + 1 )

imList = [ ]

for i in range( 565 ):
    x = X[ i, : ]
    y = Y[ i, : ]
    
    binImage = np.zeros( ( imSize, imSize ) )
    
    for j in range( 761 ):
        
        xVal = x[ j ]
        yVal = y[ j ]
        
        if xVal > xMax or xVal < 0:
            
            continue
        
        if yVal > yMax or yVal < -yMax:
            
            continue
        
        xInds = np.where( xVal <= xIncs )
        yInds = np.where( yVal <= yIncs )
        
        xI = xInds[ 0 ][ 0 ] - 1
        yI = yInds[ 0 ][ 0 ] - 1
        
        binImage[ 249 - yI, xI ] = 1
        # binImage = binImage.astype( np.uint8 )
    
    imList.append( binImage )
    
    imArray = binImage * 255

    myImage = Image.fromarray( imArray )
    myImage = myImage.convert("1")
    myImage.save(f'dataset/250pngPIL/{i+1}.png')
    
    
# fig, axes = plt.subplots( )
# axes.scatter( X[ 1, : ], Y[ 1, : ], s = 0.8, c = "k" )
# axes.axis( "equal" )
# axes.set_xlim( [ 0, xMax ] )
# axes.set_ylim( [ -yMax, yMax ] )

pickle.dump( imList, open( "dataset/xyscan_to_array_6_by_6.p", "wb" ) )