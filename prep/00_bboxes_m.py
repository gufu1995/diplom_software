# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:07:10 2020

@author: Gunnar
"""

import pandas as pd
import numpy as np

df = pd.read_csv( "bboxes_s1.csv", sep = ";", index_col = 0 )

values = df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values
values = np.vstack( ( values[ 340 : ], values[ : 340 ] ) )

values_m = ( values + 1 ) / 500 * 10

corr = np.ones( ( len( values ), 4 ) ) * np.array( [ -1, -5, -1, -5 ] )

values_m = values_m + corr

new_df = pd.DataFrame( df[ "path" ] )
new_df[ [ "xmin", "ymin", "xmax", "ymax" ] ] = values_m

new_df.to_csv( "bboxes_m.csv", sep = ";", decimal = ",", index = False )