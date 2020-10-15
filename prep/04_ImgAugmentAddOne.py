# -*- coding: utf-8 -*-

import pandas as pd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches

cwd = os.getcwd( )
pngPath = os.path.join( cwd, "png" )

cutPath = os.path.join( cwd, "pngCutout" )
testPath = os.path.join( cwd, "pngTest" )

myDF = pd.read_csv( "trainLabel2.csv", sep = ";", index_col = 0 )

newList = [ ]

for index, row in myDF.iterrows( ):
    path = row.path
    img = Image.open( path )
    
    imgNp = np.array( img )
    
    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax 
    
    cutNmbr = np.random.randint( 1, 680 )
    
    imgCut = Image.open( os.path.join( cutPath, f"{cutNmbr}.png" ) )
    imgCutNp = np.array( imgCut )
    
    height = imgCutNp.shape[ 0 ]
    width = imgCutNp.shape[ 1 ]
    
    Border = 25
    Spacing = 25
    
    xmaxP = 499 - width - Border
    ymaxP = 499 - height - Border
    xminP = Border
    yminP = Border
    
    while True:
        xminN = np.random.randint( xminP, xmaxP )
        yminN = np.random.randint( yminP, ymaxP )
        xmaxN = xminN + width
        ymaxN = yminN + height
        
        if xminN > xmax + Spacing:
            break
        if yminN > ymax + Spacing: 
            break
        
        if xmaxN < xmin - Spacing:
            break
        if ymaxN < ymin - Spacing:
            break
        
    imgNpAlt = imgNp
    imgNpAlt[ yminN : ymaxN, xminN : xmaxN ] = imgCutNp
    
    myAugImg = Image.fromarray( imgNpAlt )
    myAugImg.save( os.path.join( pngPath, f"{index + 680}.png"))
    
    
    newList.append( [ os.path.join( pngPath, f"{index + 680}.png" ), xmin, ymin, xmax, ymax, "pallet" ] )
    newList.append( [ os.path.join( pngPath, f"{index + 680}.png" ), xminN, yminN, xmaxN, ymaxN, "pallet" ] )
    

    
newDF = pd.DataFrame( newList, columns = [ "path", "xmin", "ymin", "xmax", "ymax", "name" ], index = range( 681, len( newList ) + 681 ) )

newDF2 = myDF.append( newDF )

newDF2.to_csv( "trainLabel3.csv", sep = ";" ) 