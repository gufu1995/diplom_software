# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:11:40 2020

@author: Gunna
"""

import pickle
import numpy as np
from PIL import Image
import os


#################### Path Declaration ####################
homePath = os.path.join( os.getcwd( )[ : os.getcwd( ).find( "Programming") ], "Programming" )
trainPath = os.path.join( homePath, "training" )
prepPath = os.path.join( homePath, "prep" )
pngPath = os.path.join( homePath, "prep/png" )
packagePath = os.path.join( homePath, "packages" )
tempPath = os.path.join( homePath, "temp" )
##########################################################

mRange, angleVec, X, Y = pickle.load( open( os.path.join( prepPath, "xyscan_range_angle_x_y.p" ), "rb" ) )

xRange = 10
yRange = 10

xMin = - int( np.ceil( np.tan( np.deg2rad( 5 ) ) * yRange / 2 ) )
xMax = xRange + xMin

yMin = - yRange / 2
yMax = yRange / 2

imArea = xRange * yRange
desPxArea = 4
scaleFactor = imArea * 100 ** 2 / desPxArea
scaleFactor = int( np.sqrt( scaleFactor ) )

imSize = scaleFactor

xIncs = np.linspace( xMin, xMax, imSize + 1 )
yIncs = np.linspace( yMin, yMax, imSize + 1 )

imList = [ ]

for i in range( 340 ):
    
    x = X[ i, : ]
    y = Y[ i, : ]
    
    binImage = np.ones( ( imSize, imSize ) )
    
    for j in range( 761 ):
        
        xVal = x[ j ]
        yVal = y[ j ]
        
        if xVal > xMax or xVal < xMin:
            
            continue
        
        if yVal > yMax or yVal < yMin:
            
            continue
        
        xInds = np.where( xVal <= xIncs )
        yInds = np.where( yVal <= yIncs )
        
        xI = xInds[ 0 ][ 0 ] - 1
        yI = yInds[ 0 ][ 0 ] - 1
        
        binImage[ imSize - 1 - yI, xI ] = 0
    
    imList.append( binImage )
    
    imArray = binImage * 255

    myImage = Image.fromarray( imArray )
    myImage = myImage.convert("RGB")
    myImage.save( os.path.join( pngPath, f'{i+1}.png' ) )
