# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:45:41 2020

@author: Gunna
"""

import pandas as pd
from PIL import Image
import numpy as np

myDF = pd.read_csv( "trainLabel2.csv", sep = ";", index_col = 0 )

for index, row in myDF.iterrows( ):
    path = row.path
    img = Image.open( path )
    
    img = np.asarray( img )
    
    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax 
    
    newImg = img[ ymin : ymax, xmin : xmax ]
    
    myImage = Image.fromarray( newImg )
    
    myImage.save( f"pngCutout/{index}.png")
    