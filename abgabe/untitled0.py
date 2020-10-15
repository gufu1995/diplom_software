# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:18:10 2020

@author: Gunnar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

ratios = np.array( [ 2/3, 1, 1.5 ] )
scales = np.array( [ 1.2, 1.7 ] )

# area covered in picture
area_width = 10 
area_height = 10

# img size
img_width = 500
img_height = 500

px_m_width = img_width /area_width
px_m_height = img_height / area_height

px_m = ( px_m_width + px_m_height ) / 2

euro_pal = np.array( [ 1.2, 0.8 ] ) # width, height
pos_euro_pal = np.array( [ 5, 5 ] )
pos_euro_pal_0 = pos_euro_pal - euro_pal / 2
pos_euro_pal_90 = pos_euro_pal - np.flip( euro_pal ) / 2


area = euro_pal[ 0 ] * euro_pal[ 1 ]
area_ratios = area / ratios 
w = np.round( np.sqrt( area_ratios ) * px_m )
h = np.round( w * ratios )

anchors = _mkanchors( w, h, pos_euro_pal[ 0 ] * px_m, pos_euro_pal[ 1 ] * px_m)
sanchors = [ _scale_enum( anchors[ i, : ], scales) for i in range( anchors.shape[ 0 ] ) ]

base_width = np.sqrt( area )


fig, axes = plt.subplots( )

axes.set_aspect( 'equal' )
axes.axis( [ 4, 6, 4, 6 ] )

euro_pal_img_0 = Rectangle( pos_euro_pal_0, euro_pal[ 0 ], euro_pal[ 1 ], fill = False, color = "brown", linewidth = 4 )
euro_pal_img_90 = Rectangle( pos_euro_pal_90, euro_pal[ 1 ], euro_pal[ 0 ], fill = False, color = "brown", linewidth = 4 )
axes.add_patch( euro_pal_img_0 )
axes.add_patch( euro_pal_img_90 )

colorDict = [ "blue", "green", "gray" ]

for scale in sanchors:
    
    for i in range( scale.shape[ 0 ] ):
        myAnchor = scale[ i, : ] / px_m
        x, y = myAnchor[ 0 ], myAnchor[ 1 ]
        width = myAnchor[ 2 ] - myAnchor[ 0 ]
        height = myAnchor[ 3 ] - myAnchor[ 1 ]
        
        rec = Rectangle( ( x, y ), width, height, fill = False, color = colorDict[ i ] )
        axes.add_patch( rec )
        
        
