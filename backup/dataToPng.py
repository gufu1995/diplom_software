# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:35:58 2020

@author: Gunna
"""

import pickle, os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

palDict = { 0 : "nicht vorhanden", 1 : "vorhanden" }
recDict = { 0 : "nicht erreichbar", 1 : "erreichbar" }

myData = pickle.load( open( os.path.join( os.getcwd( ), "preppedData.p" ), "rb" ) )

randLabel, randData = random.choice( list( myData.items ( ) ) )


for label, data in myData.items():
    fig = plt.figure( figsize = ( 10, 7 ) )
    axes = plt.subplot( 1, 1, 1 )
    axes.scatter( data[ "x" ], data[ "y" ], s = 1 )
    
    axes.xaxis.set_major_locator( MultipleLocator( 1 ) )
    axes.xaxis.set_minor_locator( MultipleLocator( 0.25 ) )
    axes.yaxis.set_major_locator( MultipleLocator( 1 ) )
    axes.yaxis.set_minor_locator( MultipleLocator( 0.25 ) )
    axes.set_xlabel( "x Achse in [m]" )
    axes.set_ylabel( "y Achse in [m]" )
    axes.set_title( f"Palette { palDict[ label[ 0] ] } | ist {recDict[ label[ 1 ] ] } | Punktewolke - Scan { label[ 2 ] }" )
    
    axes.grid( )
    
    fig.tight_layout( )
    
    fig.savefig( os.path.join( os.getcwd() , "dataPNGs", f"{label[2]}_{label[0]}_{label[1]}.png" ), bbox_inches='tight')
    plt.close(fig) 