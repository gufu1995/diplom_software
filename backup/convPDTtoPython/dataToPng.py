# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:35:58 2020

@author: Gunna
"""

import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


myRange, angleVec, X, Y = pickle.load( open( "dataset/xyscan_range_angle_x_y.p", "rb" ) )



for i in range( len( myRange ) ):
    
    fig = plt.figure( figsize = ( 10, 7 ) )
    axes = plt.subplot( 1, 1, 1 )
    axes.scatter( X[ i, : ], Y[ i, : ], s = 1 )
    
    axes.xaxis.set_major_locator( MultipleLocator( 1 ) )
    axes.xaxis.set_minor_locator( MultipleLocator( 0.25 ) )
    axes.yaxis.set_major_locator( MultipleLocator( 1 ) )
    axes.yaxis.set_minor_locator( MultipleLocator( 0.25 ) )
    axes.set_xlabel( "x Achse in [m]" )
    axes.set_ylabel( "y Achse in [m]" )
    axes.set_title( f"Versuch Nr. {i}" )
    
    axes.grid( )
    
    fig.tight_layout( )
    
    fig.savefig( f"dataset/png/{i}.png", bbox_inches='tight')
    plt.close(fig) 