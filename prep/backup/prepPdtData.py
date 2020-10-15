# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:51:02 2020

@author: Gunna
"""

import os
import pandas as pd
import pickle
import numpy as np

workingDir = os.getcwd( )

os.chdir( "../../" )
os.chdir( r"PDT\Pallet_Detection" )
pdtPath = os.getcwd( ) 
pdtData = "AllData"
dataPath = os.path.join( pdtPath, pdtData )

hasPallet = { 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 52, 53, 54, 55, 56, 58, 60, 61, 74, 75, 76, 77, 79, 82, 86, 87, 97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 115, 117, 118, 119, 120, 121, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 191, 192, 193, 206, 207, 208, 209, 213, 214, 225 }

classes = [ "Class2", "Class1" ]

data = { 0 : None, 1 : None }

for i, lClass in enumerate( classes ):
    actPath = os.path.join( dataPath, lClass )
    
    dataFiles = [ file for file in os.listdir( actPath ) if file.endswith(".txt") ]
    
    data[ i ] = dataFiles
    
prepData = { }    
pdList = [ ]

for i in range( 2 ):

    for file in data[ i ]:
        
        name = file[ :-4 ]
        scanNmb = int( name[ 4: ] )
        
        
        fileData = pd.read_csv( os.path.join( dataPath, classes[ i ], file ), sep = " ", header = None, names = [ "blank", "inc" , "range" ] )
        fileData = fileData.drop( columns = [ "blank" ] )
        newData = fileData.groupby( "inc" ).mean()
        
        newData["deg"] = newData.apply( lambda row: row.name * 0.25, axis = 1 )
        newData[ "rad" ] = newData.apply( lambda row: np.deg2rad( row[ "deg" ] - 5 ), axis = 1 )
        newData[ "x" ] = newData.apply( lambda row: np.cos( row[ "rad" ] ) * row[ "range" ], axis = 1 )
        newData[ "y" ] = newData.apply( lambda row: np.sin( row[ "rad" ] ) * row[ "range" ], axis = 1 )
        
        if i == 0 and ( scanNmb not in hasPallet ):
            j = 0
        else:
            j = 1
            
        if i == 0:
            scanNmb += 340
            
        prepData[ (j, i, scanNmb ) ] = newData
        pdList.append( ( j, i, newData, scanNmb ) )
  
        
pdDf = pd.DataFrame( pdList, columns = [ "hasPallet", "reachable", "data", "scnNmb"])

newDf = pdDf.set_index( "scnNmb" ).sort_index( axis = 0 )

pickle.dump( newDf, open( os.path.join( workingDir, "preppedLabeledData.p" ), "wb" ) )
pickle.dump( prepData, open( os.path.join( workingDir, "preppedData.p" ), "wb" ) )