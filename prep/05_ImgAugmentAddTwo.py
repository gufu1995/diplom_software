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

myDF = pd.read_csv( "trainLabel3.csv", sep = ";", index_col = 0 )

newList = [ ]

def checkOverlap( box1, box2, spacing ):
    xmin1 = box1[ 0 ]
    ymin1 = box1[ 1 ]
    xmax1 = box1[ 2 ]
    ymax1 = box1[ 3 ]
    
    xmin2 = box2[ 0 ]
    ymin2 = box2[ 1 ]
    xmax2 = box2[ 2 ]
    ymax2 = box2[ 3 ]
    
    if xmin1 > xmax2 + spacing:
        return False
    if xmax1 < xmin2 - spacing:
        return False
    if ymax1 < ymin2 - spacing:
        return False
    if ymin1 > ymax2 + spacing:
        return False
    
    return True

def createBox( imArray, border ):
    height = imArray.shape[ 0 ]
    width = imArray.shape[ 1 ]
    
    xmaxP = 499 - width - border
    ymaxP = 499 - height - border
    xminP = border
    yminP = border
    
    xminN = np.random.randint( xminP, xmaxP )
    yminN = np.random.randint( yminP, ymaxP )
    xmaxN = xminN + width
    ymaxN = yminN + height
    
    return [ xminN, yminN, xmaxN, ymaxN ]
    

for index, row in myDF.iterrows( ):
    path = row.path
    img = Image.open( path )
    
    if index > 680:
        continue
    
    imgNp = np.array( img )
    
    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax 
    
    bboxO = [ xmin, ymin, xmax, ymax ]
    
    cutNmbr1 = np.random.randint( 1, 680 )
    cutNmbr2 = np.random.randint( 1, 680 )

    imgCut1 = Image.open( os.path.join( cutPath, f"{cutNmbr1}.png" ) )
    imgCut2 = Image.open( os.path.join( cutPath, f"{cutNmbr2}.png" ) )
    imgCutNp1 = np.array( imgCut2 )
    imgCutNp2 = np.array( imgCut2 )
    
    Border = 25
    Spacing = 25
    
    
    
    while True:
        bbox1 = createBox( imgCutNp1, border = Border )
        bbox2 = createBox( imgCutNp2, border = Border )
        
        ovO1 = checkOverlap( bboxO, bbox1, spacing = Spacing )
        ov12 = checkOverlap( bbox1, bbox2, spacing = Spacing )
        ovO2 = checkOverlap( bbox2, bboxO, spacing = Spacing )
        
        if not( ovO1 or ovO2 or ov12 ) == True:
            break
            
        
        
    imgNpAlt = imgNp
    imgNpAlt[ bbox1[ 1 ] : bbox1[ 3 ], bbox1[ 0 ] : bbox1[ 2 ] ] = imgCutNp1
    imgNpAlt[ bbox2[ 1 ] : bbox2[ 3 ], bbox2[ 0 ] : bbox2[ 2 ] ] = imgCutNp2 
    
    myAugImg = Image.fromarray( imgNpAlt )
    myAugImg.save( os.path.join( pngPath, f"{1360 + index}.png"))
    
    
    newList.append( [ os.path.join( pngPath, f"{1360 + index}.png" ), bboxO[ 0 ], bboxO[ 1 ], bboxO[ 2 ], bboxO[ 3 ], "pallet" ] )
    newList.append( [ os.path.join( pngPath, f"{1360 + index}.png" ), bbox1[ 0 ], bbox1[ 1 ], bbox1[ 2 ], bbox1[ 3 ], "pallet" ] )
    newList.append( [ os.path.join( pngPath, f"{1360 + index}.png" ), bbox2[ 0 ], bbox2[ 1 ], bbox2[ 2 ], bbox2[ 3 ], "pallet" ] )
    

    
newDF = pd.DataFrame( newList, columns = [ "path", "xmin", "ymin", "xmax", "ymax", "name" ], index = range( 2041, len( newList ) + 2041 ) )

newDF2 = myDF.append( newDF )

newDF2.to_csv( "trainLabel4.csv", sep = ";" ) 