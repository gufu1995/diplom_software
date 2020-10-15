# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:28:58 2020

@author: Gunna
"""

import os
import xmltodict

#################### Path Declaration ####################
homePath = os.path.join( os.getcwd( )[ : os.getcwd( ).find( "Programming") ], "Programming" )
trainPath = os.path.join( homePath, "training" )
prepPath = os.path.join( homePath, "prep" )
pngPath = os.path.join( homePath, "pngData/test/" )
packagePath = os.path.join( homePath, "packages" )
tempPath = os.path.join( homePath, "temp" )
##########################################################


path = os.path.join( prepPath, "labeled" )

xmlFiles = os.listdir( path )

labelList = []

for file in xmlFiles:
    
    with open( os.path.join( path, file ) ) as f:
        doc = xmltodict.parse( f.read( ) )
    
    imName = doc[ "annotation" ][ "filename" ]
    imPath = doc[ "annotation" ][ "path" ]
    bndBox = doc[ "annotation" ][ "object" ][ "bndbox" ]
    name = doc[ "annotation" ][ "object" ][ "name" ]
    xmin = int( bndBox[ "xmin" ] )
    xmax = int( bndBox[ "xmax" ] )
    ymin = int( bndBox[ "ymin" ] )
    ymax = int( bndBox[ "ymax" ] )
    
    labelList.append( [ imPath, imName, xmin, ymin, xmax, ymax ] )

csvString = "Bild Pfad;Bild Name;xMin;yMin;xMax;yMax\n"

for entry in labelList:
    csvString += f"{entry[ 0 ]};{entry[ 1 ]};{entry[ 2 ]};{entry[ 3 ]};{entry[ 4 ]};{entry[ 5 ]}\n"
    
with open( os.path.join( prepPath, "labelList.csv" ), "w+" ) as f:
    f.write( csvString )