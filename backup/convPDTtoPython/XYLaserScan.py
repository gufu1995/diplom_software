# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:56:40 2020

@author: Gunna
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

cwd = os.getcwd()
os.chdir( "../" )
os.chdir( "data" )


numSamples = 761
numScans = 565
FOV = 190.0
FOV = np.deg2rad( FOV )
res = FOV / numSamples

startAngle = -FOV / 2
endAngle = FOV / 2

angleVector = [ startAngle + res * i for i in range( numSamples ) ]

angleVector = np.array( angleVector )

dataArray = np.zeros( ( numScans, numSamples ) )

for i in range( 1, numScans + 1 ):
    filePath = f"Scan{i}.txt"
    fileData = np.loadtxt( filePath )
    fileData = fileData[ :, 1 ]
    fileData = fileData.reshape( numSamples, 4, order = "F" )
    fileData = fileData.mean( axis = 1 )

    
    dataArray[ i - 1, : ] = fileData
    
    
X = np.zeros( ( numScans, numSamples ) )
Y = np.zeros( ( numScans, numSamples ) )

for i, row in enumerate( dataArray ):
    x = row * np.cos( angleVector )
    y = row * np.sin( angleVector )
    
    X[ i, : ] = x
    Y[ i, : ] = y

 

pickle.dump( ( dataArray, angleVector, X, Y ), open( os.path.join( cwd, "dataset", "xyscan_range_angle_x_y.p" ), "wb" ) )
