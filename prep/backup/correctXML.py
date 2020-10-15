# -*- coding: utf-8 -*-

import os
import glob
import xml.etree.ElementTree as ET

cwd = os.getcwd( )
labelPath = os.path.join( cwd, "labeled" )
files = glob.glob( labelPath + "/*.xml" )
imPath = os.path.join( cwd, "png" )


def newX( x ):
    return int( ( ( ( x / 250 ) * 8 ) + 1 ) / 10 * 500 )

def newY( y ):
    return int( ( ( y / 250 ) * 8 + 1 ) / 10 * 500 )

for file in files:
    
    number = int( file[ : -4 ].split( "\\" )[ -1 ] )
    
    tree = ET.parse( file )
    root = tree.getroot()
    
    for path in root.iter( "path" ):
        path.text = os.path.join( imPath, f"{number}.png" )
        # path.set('updated', 'yes')
    
    for folder in root.iter( "folder" ):
        folder.text = "png"
        # folder.set('updated', 'yes')
        
    for width in root.iter( "width" ):
        width.text = "500"
        # width.set('updated', 'yes')
        
    for height in root.iter( "height" ):
        height.text = "500"
        # height.set('updated', 'yes')
        
    for xmin in root.iter( "xmin" ):
        xmin.text = str( newX( int( xmin.text ) ) )

        
    for ymin in root.iter( "ymin" ):
        ymin.text = str( newY( int( ymin.text ) ) )
    
    
    for xmax in root.iter( "xmax" ):
        xmax.text = str( newX( int( xmax.text ) ) )
        
    for ymax in root.iter( "ymax" ):
        ymax.text = str( newY( int( ymax.text ) ) )
        
    tree.write( f"labeled/{number}.xml" )