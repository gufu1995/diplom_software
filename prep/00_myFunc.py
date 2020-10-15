# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:55:25 2020

@author: Gunna
"""

import numpy as np
from PIL import Image
import os

#################### Path Declaration ####################
homePath = os.path.join( os.getcwd( )[ : os.getcwd( ).find( "Programming") ], "Programming" )
trainPath = os.path.join( homePath, "training" )
prepPath = os.path.join( homePath, "prep" )
pngPath = os.path.join( homePath, "pngData" )
packagePath = os.path.join( homePath, "packages" )
tempPath = os.path.join( homePath, "temp" )
##########################################################

numSamples = 761
FOV = 190.0
FOV = np.deg2rad( FOV )
res = FOV / numSamples
startAngle = -FOV / 2
endAngle = FOV / 2
angleVector = [ startAngle + res * i for i in range( numSamples ) ]
angleVector = np.array( angleVector )


def readScan( path ):
    fileData = np.loadtxt( path )
    fileData = fileData[ :, 1 ]
    fileData = fileData.reshape( numSamples, 4, order = "F" )
    fileData = fileData.mean( axis = 1 )
    
    return fileData

    
def convRangeScantoXY( Scan ):
    x = Scan * np.cos( angleVector )
    y = Scan * np.sin( angleVector )

    return x, y


def createBinArray( x, y, size = 250,  xlim = 6, ylim = 3 ):
    binArray = np.zeros( ( size, size ) )
    
    xIncs = np.linspace( 0, xlim, size + 1 )
    yIncs = np.linspace( - ylim, ylim, size + 1 )
    
    for j in range( len( x ) ):
        
        xVal = x[ j ]
        yVal = y[ j ]
        
        if xVal > xlim or xVal < 0:
            
            continue
        
        if yVal > ylim or yVal < - ylim:
            
            continue
        
        xInds = np.where( xVal <= xIncs )
        yInds = np.where( yVal <= yIncs )
        
        xI = xInds[ 0 ][ 0 ] - 1
        yI = yInds[ 0 ][ 0 ] - 1
        
        binArray[ 249 - yI, xI ] = 1
    
    binArray = binArray.astype( np.uint8 )
    
    return binArray

def createBinImage( binArray ):
    
    imArray = binArray * 255

    myImage = Image.fromarray( imArray )
    myImage = myImage.convert("L")
    
    return myImage
    
    
    
def scanToImage( scanPath, Destination ):
    
    myScan = readScan( scanPath )
    x, y = convRangeScantoXY( myScan ) 
    myImageArray = createBinArray( x = x, y = y, size = 250, xlim = 6, ylim = 3 )
    myImage = createBinImage( myImageArray )
    myImage.save( Destination )
    
    return myImageArray, myImage

scanToImage( "data/Scan1.txt", "temp/test1.png" )