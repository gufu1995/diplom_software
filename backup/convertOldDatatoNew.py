# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:06:50 2020

@author: Gunna
"""

import os
import shutil

curDir = os.getcwd( )

files = [ file for file in os.listdir( "temp" ) if file.endswith( ".txt" ) ] 

for file in files:
    srcDir = os.path.join( curDir, "temp", file )
    nmbr = int( file[ :-4 ][ 4: ] ) + 340 
    desDir = os.path.join( curDir, "data", f"Scan{nmbr}.txt")
    
    shutil.copy( srcDir, desDir )