# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:02:27 2020

@author: Gunna
"""

import pandas as pd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches

cwd = os.getcwd( )
pngPath = os.path.join( cwd, "png" )

myDF = pd.read_csv( "trainLabel.csv", sep = ";", index_col = 0 )

newList = [ ]

for index, row in myDF.iterrows( ):
    path = row.path
    img = Image.open( path )
    
    imgNp = np.asarray( img )
    
    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax 
    
    height = ymax - ymin
    
    imgFlUp = np.flip( imgNp, 0 )
    
    
    myImgFlUp = Image.fromarray( imgFlUp )

    
    myImgFlUp.save( f"png/{index + 340}.png" )
    
    ymin = 499 - ymax
    ymax = ymin + height
    
    newList.append( [ os.path.join( pngPath, f"{index + 340}.png" ), xmin, ymin, xmax, ymax, "pallet" ] )
    
    
    
newDF = pd.DataFrame( newList, columns = [ "path", "xmin", "ymin", "xmax", "ymax", "name" ], index = range( 341, 681 ) )

newDF2 = myDF.append( newDF )

newDF2.to_csv( "trainLabel2.csv", sep = ";" ) 
