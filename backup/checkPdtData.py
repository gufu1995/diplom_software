# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:35:58 2020

@author: Gunna
"""

import pickle, os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

decDict = { 0 : "nicht vorhanden", 1 : "vorhanden" }

myData = pickle.load( open( os.path.join( os.getcwd( ), "preppedData.p" ), "rb" ) )

randLabel, randData = random.choice( list( myData.items ( ) ) )

# fig = plt.figure( figsize = ( 10, 7 ) )
# axes = plt.subplot( 1, 1, 1, projection = "polar" )
# axes.set_thetamin( -5 )
# axes.set_thetamax( 185 )
# axes.scatter( randData[ "rad" ], randData[ "range" ], s = 0.5 )
# fig.tight_layout( )

fig = plt.figure( figsize = ( 10, 7 ) )
axes = plt.subplot( 1, 1, 1 )
axes.scatter( randData[ "x" ], randData[ "y" ], s = 1 )

axes.xaxis.set_major_locator( MultipleLocator( 1 ) )
axes.xaxis.set_minor_locator( MultipleLocator( 0.25 ) )
axes.yaxis.set_major_locator( MultipleLocator( 1 ) )
axes.yaxis.set_minor_locator( MultipleLocator( 0.25 ) )
axes.set_xlabel( "x Achse in [m]" )
axes.set_ylabel( "y Achse in [m]" )
axes.set_title( f"Punktewolke - { randLabel[ 1 ] } | Palette { decDict[ randLabel[ 0] ] }" )

axes.grid( )

fig.tight_layout( )