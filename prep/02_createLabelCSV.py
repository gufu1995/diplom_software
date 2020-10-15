# -*- coding: utf-8 -*-

import os, glob
%matplotlib inline
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

cwd = os.getcwd( )

imgPath = os.path.join( cwd, "png" )
labPath = os.path.join( cwd, "labeled" )

myList = [ ]

for i in range( 1, 341 ):
    
    tree = ET.parse( os.path.join( labPath, f"{i}.xml" ) )
    root = tree.getroot()
    
    for x in root.iter( "xmin" ):
        xmin = int( x.text )
        
    for y in root.iter( "ymin" ):
        ymin = int( y.text )
        
    for x in root.iter( "xmax" ):
        xmax = int( x.text )
        
    for y in root.iter( "ymax" ):
        ymax = int( y.text )
        
    for p in root.iter( "path" ):
        path = p.text 
        
    for n in root.iter( "name" ):
        name = n.text
        
    entry = [ path, xmin, ymin, xmax, ymax, name ]
    
    myList.append( entry )
    
    
myDF = pd.DataFrame( myList, columns = [ "path", "xmin", "ymin", "xmax", "ymax", "name" ], index = range( 1, 341 ) )

myDF.to_csv( "trainLabel.csv", sep = ";" )
