# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:48:17 2020

@author: Gunnar
"""

import glob, shutil, os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


xml = glob.glob( "./labeled/*.xml" )

# counter = 341

# for item in xml:
    
#     item = int( item.split( "\\")[ -1 ].split( "." )[ 0 ] ) 
    
#     if item > 340:
        
#         shutil.copyfile( f"lid_data_others/Scan{item}.txt", f"lid_data/Scan{counter}.txt" )
#         os.remove( f"lid_data_others/Scan{item}.txt" )
        
#         shutil.copyfile( f"labeled/{item}.xml", f"labeled_others/{counter}.xml" )
#         os.remove( f"labeled/{item}.xml" )
        
#         counter += 1

df = pd.read_csv( "bboxes_m.csv", sep = ";", decimal = "," )

bbvalues = df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values
pathvalues = df[ "path" ].values

pathtop = list( pathvalues[ : 340 ] )
bbtop = bbvalues[ : 340 ]


def bboxes_m_to_px( bbox, xlim, ylim, pxx, pxy ):
        
    r_x = xlim[ 1 ] - xlim[ 0 ]
    r_y = ylim[ 1 ] - ylim[ 0 ]
    
    px_m = pxx / r_x 
    py_m = pxy / r_y 
    
    
    bbox = bbox - np.ones( ( len( bbox ), 4 ) ) * np.array( [ [ xlim[ 0 ], ylim[ 0 ], xlim[ 0 ], ylim[ 0 ] ] ] )
    
    bbox = bbox * ( np.ones( ( len( bbox ), 4 ) ) * np.array( [ [ px_m, py_m, px_m, py_m ] ] ) )
    bbox = bbox.astype( np.int )
    bbox = bbox - np.ones( ( len( bbox), 4 ) )
    
    return bbox.astype( np.int )

n = [ ]
p = [ ]
up = [ ]
pup = [ ]

for item in range( 341, 391 ):
        
    tree = ET.parse( f"./labeled/{item}.xml" )
    root = tree.getroot()
    
    root = root.find( "object" )
    
    entry = root.find( "bndbox" )
    
    xmin = int( entry.find( "xmin" ).text )
    ymin = int( entry.find( "ymin" ).text )
    xmax = int( entry.find( "xmax" ).text )
    ymax = int( entry.find( "ymax" ).text )
    
    row = np.array( [ [ xmin, ymin, xmax, ymax ] ] )
    row_up = np.array( [ [ xmin, 498 - ymax, xmax, 498 - ymin ] ] )

    # m = bboxes_m_to_px( row, xlim = [ -1, 9 ], ylim = [ -5, 5 ], pxx = 500, pxy = 500 )
    # m_up = bboxes_m_to_px( row_up, xlim = [ -1, 9 ], ylim = [ -5, 5 ], pxx = 500, pxy = 500 )
    
    row = ( row + 1 ) / 500 * 10
    row_up = ( row_up + 1 ) / 500 * 10
    
    corr = np.ones( ( 1, 4 ) ) * np.array( [ -1, -5, -1, -5 ] )
    
    row = row + corr
    row_up = row_up + corr
    
    row = row.reshape( (4, ) )
    row_up = row_up.reshape( ( 4, ) )
    
    n.append( row )
    up.append( row_up )
    p.append( f"./png_err/{item}.png" )
    pup.append( f"./png_err/{item+390}.png")
    
    
pathbottom = [ f"./png_err/{391 + i}.png" for i in range( 340 ) ]
bbbottom = bbvalues[ 340 : ]

n = np.array( n )
up = np.array( up )

finalpath = pathtop + p + pathbottom + pup 
finalbboxes = np.vstack( ( bbtop, n, bbbottom, up ) )

new_df = pd.DataFrame( )
new_df[ "path" ] = finalpath 
new_df[ [ "xmin", "ymin", "xmax", "ymax" ] ] = finalbboxes

new_df.to_csv( "bboxes_mn.csv", sep = ";", decimal = ",", index = False )