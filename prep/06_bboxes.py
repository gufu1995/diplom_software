# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 18:28:59 2020

@author: Gunna
"""

import pandas as pd
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
%matplotlib inline

myDF = pd.read_csv( "trainLabel4.csv", sep = ";", index_col = 0 )

uniqueImgs = myDF.path.unique( )

for i, Img in enumerate( uniqueImgs ):
    fig = plt.figure( )
    ax = fig.add_axes( [ 0, 0, 1, 1 ] )
    image = plt.imread( Img )
    plt.imshow( image, cmap = "gray" )
    plt.axis( "off" )
    
    entrys = myDF.path == Img
    smallDF = myDF.loc[ entrys ]
    
    for index, row in smallDF.iterrows():
        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax 
        name = row.name 
        
        width = xmax - xmin
        height = ymax - ymin
        
        edgecolor = 'r'
        ax.annotate( 'pallet', xy = ( xmin - 1, ymin - 3 ) )
        
        rect = patches.Rectangle( ( xmin, ymin ), width, height, edgecolor = edgecolor, facecolor = 'none' )
        ax.add_patch( rect )
        
    fig.tight_layout( )
    
    plt.savefig( f"pngLabeled/{ i + 1 }.jpg", bbox_inches = 'tight' )
    plt.close( fig )
    