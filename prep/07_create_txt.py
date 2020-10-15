# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:09:10 2020

@author: Gunna
"""

import os
import pandas as pd
import numpy as np

def create_txt( df ):
    outputDF = pd.DataFrame( )
    outputDF[ "path" ] = df.path
    outputDF.path = outputDF.path.apply( lambda x: x.split( "\\" )[ -1 ] )
    outputDF.path = "/data/" + outputDF.path + ","
    for index, row in df.iterrows( ):
        outputDF.path[ index ] = outputDF.path[ index ] + str( row[ "xmin" ] ) + "," + str( row[ "ymin" ] ) + "," + str( row[ "xmax" ] ) + "," +  str( row[ "ymax" ] ) + "," + row[ "name" ]
    
    
    return outputDF


cwd = os.getcwd( )
os.chdir( "../" )
mainDir = os.getcwd( )
trainDir = os.path.join( mainDir, "training" )
prepDir = os.path.join( mainDir, "prep" )
pngDir = os.path.join( prepDir, "png" )

myDF = pd.read_csv( os.path.join( prepDir, "trainLabel4.csv"), sep = ";", index_col = 0 )

dataSet = np.array( myDF.values ) 
path = dataSet[ :, 0 ]
xmin = dataSet[ :, 1 ]
ymin = dataSet[ :, 2 ]
xmax = dataSet[ :, 3 ]
ymax = dataSet[ :, 4 ]

lenData = len( dataSet )

train = myDF.sample(frac = 0.8, random_state = 200 )
test = myDF.drop( train.index )

trainTXT = create_txt( train )
testTXT = create_txt( test )

trainTXT.to_csv( os.path.join( trainDir, "train.txt" ), header = None, index = None, sep = " " )
testTXT.to_csv( os.path.join( trainDir, "test.txt" ), header = None, index = None, sep = " " )