# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 20:36:02 2020

@author: Gunnar
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import pandas as pd

path_prep = os.getcwd( )
path_lid = os.path.join( path_prep, "lid_data" )

create_new_ims = False
create_cutouts = False
create_bckgrnd = False
add_one_bckgrnd = False
add_two_bckgrnd = False
add_others = False
add_one = False
add_two = False
create_bbox_imgs = True


def read_txt( filepath ):
    incs = np.linspace( - 95, 95, num = 761, endpoint = True )
    incs = np.deg2rad( incs )
    data = np.loadtxt( filepath )
    ran = data[ :, 1 ].reshape( ( int( len( data ) / 4 ), 4 ), order = "F" )
    ran = np.mean( ran, axis = 1 )
    x = ran * np.cos( incs )
    y = ran * np.sin( incs )
    
    return x, y

def map_xy( x, y, xlim, ylim, error, pxx, pxy ):
    array_out = np.zeros( ( pxy, pxx ) )
    
    cm_pxx = ( xlim[ 1 ] - xlim[ 0 ] ) * 100 / pxx
    cm_pxy = ( ylim[ 1 ] - ylim[ 0 ] ) * 100 / pxy
    
    px_errx = np.round( error * 100 / cm_pxx ).astype( np.int )
    px_erry = np.round( error * 100 / cm_pxx ).astype( np.int )
    
    entry_array = np.zeros( ( 2 * px_erry + 1, 2 * px_errx + 1 ) )
    i_m = np.array( [ px_erry, px_errx ] )
    
    for e_x in range( entry_array.shape[ 1 ] ):
        for e_y in range( entry_array.shape[ 0 ] ):
            err = np.linalg.norm( i_m - np.array( [ e_y, e_x ] ) ) * cm_pxx / 100
            
            value = np.max( [ 1 - ( err / error ) ** ( 0.1 ), 0 ] )
            
            entry_array[ e_y, e_x ] = value
                                                   
            
    
    x_grid = np.linspace( xlim[ 0 ], xlim[ 1 ], num = pxx + 1 )
    y_grid = np.linspace( ylim[ 0 ], ylim[ 1 ], num = pxy + 1 )
    
    for i in range( len( x ) ):
        try:
            x_c = np.where( x[ i ] >= x_grid )[ 0 ][ - 1 ]
            y_c = np.where( y[ i ] >= y_grid )[ 0 ][ - 1 ]
        except: 
            continue
        
        # if x_c >= px_errx & x_c < pxx - px_errx:
        #     if y_c >= px_erry & y_c < pxy - px_erry:
                
                # array_out[ y_c - px_erry : y_c + px_erry + 1, x_c - px_errx : x_c + px_errx + 1 ] = array_out[ y_c - px_erry : y_c + px_erry + 1, x_c - px_errx : x_c + px_errx + 1 ] + entry_array
        try:
            array_out[ y_c - px_erry : y_c + px_erry + 1, x_c - px_errx : x_c + px_errx + 1 ] = array_out[ y_c - px_erry : y_c + px_erry + 1, x_c - px_errx : x_c + px_errx + 1 ] + entry_array
        except:
            continue
    
    array_out = np.clip( array_out, a_max = 1.0, a_min = 0.0 )
    array_out = array_out * 255
    
    return array_out.astype( np.int8 )

def array_to_im( array, filepath ):
    
    myImage = Image.fromarray( array, mode = "L" ) 
    myImage = myImage.convert( mode = "RGB" )
    myImage.save( filepath )


def checkOverlap( box1, box2, spacing ):
    xmin1 = box1[ 0 ]
    ymin1 = box1[ 1 ]
    xmax1 = box1[ 2 ]
    ymax1 = box1[ 3 ]
    
    xmin2 = box2[ 0 ]
    ymin2 = box2[ 1 ]
    xmax2 = box2[ 2 ]
    ymax2 = box2[ 3 ]
    
    if xmin1 > xmax2 + spacing:
        return False
    if xmax1 < xmin2 - spacing:
        return False
    if ymax1 < ymin2 - spacing:
        return False
    if ymin1 > ymax2 + spacing:
        return False
    
    return True

def createBox( imArray, border ):
    height = imArray.shape[ 0 ]
    width = imArray.shape[ 1 ]
    
    xmaxP = 499 - width - border - 100
    ymaxP = 499 - height - border
    xminP = border
    yminP = border
    
    xminN = np.random.randint( xminP, xmaxP )
    yminN = np.random.randint( yminP, ymaxP )
    xmaxN = xminN + width
    ymaxN = yminN + height
    
    return [ xminN, yminN, xmaxN, ymaxN ]

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

if __name__ == "__main__":
    
    xlim = [ -1, 9 ]
    pxx = 500
    ylim = [ -5, 5 ]
    pxy = 500
    err = 0.05
    
    
    if create_new_ims == True:
        for i in range( 1, 391 ):
            
            x,y = read_txt( os.path.join( path_lid, f"Scan{i}.txt" ) )
            array = map_xy( x = x, y = y, xlim = xlim, ylim = ylim, error = err, pxx = pxx, pxy = pxy )
            array_to_im( array, os.path.join( path_prep, "png_err", f"{i}.png" ) )
            
            array = np.flipud( array )
            array_to_im( array, os.path.join( path_prep, "png_err", f"{i+390}.png" ) )
            
        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_mn.csv" ), sep = ";", decimal = "," )
        
        bboxes_m = data_df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values
        bboxes_px = bboxes_m_to_px( bbox = bboxes_m, xlim = xlim, ylim = ylim, pxx = pxx, pxy = pxy )

        data_df[ ["xmin", "ymin", "xmax", "ymax" ] ] = bboxes_px
        
        data_df.to_csv( "bboxes_s1.csv", sep = ";", decimal = ",", index = False )
    
    if create_cutouts == True:
        
        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_mn.csv" ), sep = ";", decimal = "," )
        
        bboxes_m = data_df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values
        bboxes_px = bboxes_m_to_px( bbox = bboxes_m, xlim = xlim, ylim = ylim, pxx = pxx, pxy = pxy )
        
        paths = data_df[ "path" ].values
        
        for i, row in enumerate( bboxes_px ):
            path = paths[ i ]
            img = Image.open( path )
            
            img = np.asarray( img )
            
            xmin, ymin, xmax, ymax = row
            
            newImg = img[ ymin : ymax, xmin : xmax ]
            
            myImage = Image.fromarray( newImg )
            
            myImage.save( f"./png_err_cutout/{i}.png")
            
    if create_bckgrnd == True:
        
        txt_list = glob.glob( "lid_data_others/*.txt" )
        
        for i, item in enumerate( txt_list ):
            
            x,y = read_txt( item )
            array = map_xy( x = x, y = y, xlim = xlim, ylim = ylim, error = err, pxx = pxx, pxy = pxy )
            array_to_im( array, os.path.join( path_prep, "png_err_back", f"{i + 1}.png" ) )
            
            array = np.flipud( array )
            array_to_im( array, os.path.join( path_prep, "png_err_back", f"{i + 1 + 175}.png" ) )
        
    if add_others == True:
        
        for i in range( 341, 566 ):
            x,y = read_txt( os.path.join( path_prep, f"lid_data_others/Scan{i}.txt" ) )
            array = map_xy( x = x, y = y, xlim = xlim, ylim = ylim, error = err, pxx = pxx, pxy = pxy )
            array_to_im( array, os.path.join( path_prep, "png_err_others", f"{i}.png" ) )
            
    if add_one_bckgrnd == True:
        
        back_list = glob.glob( "png_err_back/*.png" )
        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_s3.csv" ), sep = ";", decimal = "," )
        
        new_list = [ ]
        for i, elem in enumerate( back_list ):
            
            img = Image.open( elem )
            img = np.array( img )
            
            cutNmbr = np.random.randint( 1, 780 )
            
            imgCut = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr}.png" ) )
            imgCut = np.array( imgCut )
            
            height = imgCut.shape[ 0 ]
            width = imgCut.shape[ 1 ]
            
            Border = 25
            Spacing = 25
            
            xmaxP = pxx - 1 - width - 150
            ymaxP = pxy - 1 - height - Border
            xminP = Border
            yminP = Border
            
            xmin = np.random.randint( xminP, xmaxP )
            ymin = np.random.randint( yminP, ymaxP )
            xmax = xmin + width
            ymax = ymin + height
            
            img[ ymin : ymax, xmin : xmax] = imgCut
            
            img = Image.fromarray( img )
            img.save( os.path.join( "./png_err", f"{ 2341 + i }.png"))
            
            
            new_list.append( [ f"./png_err/{ 2341 + i }.png", xmin, ymin, xmax, ymax ] )
            
        data_df_append = pd.DataFrame( new_list, columns = [ "path", "xmin", "ymin", "xmax", "ymax" ] )
        
        data_df = data_df.append( data_df_append )
        
        data_df.to_csv( "bboxes_s4.csv", sep = ";", decimal = ",", index = False )
            
    if add_two_bckgrnd == True:    
    
        new_list = [ ]

        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_s4.csv" ), sep = ";", decimal = "," )
        
        paths = glob.glob( "png_err_back/*.png" )
        
        for i, path in enumerate( paths ):
            
            img = Image.open( path )
            
            img = np.array( img )
            
            cutNmbr1 = np.random.randint( 1, 780 )
            cutNmbr2 = np.random.randint( 1, 780 )
        
            imgCut1 = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr1}.png" ) )
            imgCut2 = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr2}.png" ) )
            imgCut1 = np.array( imgCut1 )
            imgCut2 = np.array( imgCut2 )
            
            Border = 25
            Spacing = 25
            
            
            while True:
                bbox1 = createBox( imgCut1, border = Border )
                bbox2 = createBox( imgCut2, border = Border )
                
                ov = checkOverlap( bbox1, bbox2, spacing = Spacing )
                
                if ov == False:
                    break
                
            img[ bbox1[ 1 ] : bbox1[ 3 ], bbox1[ 0 ] : bbox1[ 2 ] ] = imgCut1
            img[ bbox2[ 1 ] : bbox2[ 3 ], bbox2[ 0 ] : bbox2[ 2 ] ] = imgCut2 
            
            img = Image.fromarray( img )
            img.save( f"./png_err/{ 2691 + i }.png" )
            
            new_list.append( [ f"./png_err/{ 2691 + i }.png", bbox1[ 0 ], bbox1[ 1 ], bbox1[ 2 ], bbox1[ 3 ] ] )
            new_list.append( [ f"./png_err/{ 2691 + i }.png", bbox2[ 0 ], bbox2[ 1 ], bbox2[ 2 ], bbox2[ 3 ] ] )

            
        
            
        data_df_append = pd.DataFrame( new_list, columns = [ "path", "xmin", "ymin", "xmax", "ymax" ] )
        
        data_df = data_df.append( data_df_append )
        
        data_df.to_csv( "bboxes_s5.csv", sep = ";", decimal = ",", index = False ) 
    
    if add_one == True:
        
        newList = [ ]

        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_s1.csv" ), sep = ";", decimal = "," )
        
        paths = data_df[ "path" ].values
        
        for i, row in enumerate( data_df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values ):
            path = paths[ i ]
            img = Image.open( path )
            
            imgNp = np.array( img )
            
            xmin, ymin, xmax, ymax = row
            
            
            cutNmbr = np.random.randint( 1, len( paths ) )
            
            imgCut = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr}.png" ) )
            imgCutNp = np.array( imgCut )
            
            height = imgCutNp.shape[ 0 ]
            width = imgCutNp.shape[ 1 ]
            
            Border = 25
            Spacing = 25
            
            xmaxP = pxx - 1 - width - Border
            ymaxP = pxy - 1 - height - Border
            xminP = Border
            yminP = Border
            
            while True:
                xminN = np.random.randint( xminP, xmaxP )
                yminN = np.random.randint( yminP, ymaxP )
                xmaxN = xminN + width
                ymaxN = yminN + height
                
                if xminN > xmax + Spacing:
                    break
                if yminN > ymax + Spacing: 
                    break
                
                if xmaxN < xmin - Spacing:
                    break
                if ymaxN < ymin - Spacing:
                    break
                
            imgNpAlt = imgNp
            imgNpAlt[ yminN : ymaxN, xminN : xmaxN ] = imgCutNp
            
            myAugImg = Image.fromarray( imgNpAlt )
            myAugImg.save( os.path.join( "./png_err", f"{i + len( paths ) + 1}.png"))
            
            
            newList.append( [ f"./png_err/{i + len( paths )+ 1}.png", xmin, ymin, xmax, ymax ] )
            newList.append( [ f"./png_err/{i + len( paths )+ 1}.png" , xminN, yminN, xmaxN, ymaxN ] )
            
        
            
        data_df_append = pd.DataFrame( newList, columns = [ "path", "xmin", "ymin", "xmax", "ymax" ] )
        
        data_df = data_df.append( data_df_append )
        
        data_df.to_csv( "bboxes_s2.csv", sep = ";", decimal = ",", index = False )
        
    if add_two == True:
        
        newList = [ ]

        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_s1.csv" ), sep = ";", decimal = "," )
        
        paths = data_df[ "path" ].values
        
        for i, row in enumerate( data_df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values ):
            path = paths[ i ]
            img = Image.open( path )
            
            imgNp = np.array( img )
            
            xmin, ymin, xmax, ymax = row
            
            bboxO = [ xmin, ymin, xmax, ymax ]
            
            cutNmbr1 = np.random.randint( 1, len( paths ) )
            cutNmbr2 = np.random.randint( 1, len( paths ) )
        
            imgCut1 = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr1}.png" ) )
            imgCut2 = Image.open( os.path.join( "./png_err_cutout", f"{cutNmbr2}.png" ) )
            imgCutNp1 = np.array( imgCut1 )
            imgCutNp2 = np.array( imgCut2 )
            
            Border = 25
            Spacing = 25
            
            
            
            while True:
                bbox1 = createBox( imgCutNp1, border = Border )
                bbox2 = createBox( imgCutNp2, border = Border )
                
                ovO1 = checkOverlap( bboxO, bbox1, spacing = Spacing )
                ov12 = checkOverlap( bbox1, bbox2, spacing = Spacing )
                ovO2 = checkOverlap( bbox2, bboxO, spacing = Spacing )
                
                if not( ovO1 or ovO2 or ov12 ) == True:
                    break
                    
                
                
            imgNpAlt = imgNp
            imgNpAlt[ bbox1[ 1 ] : bbox1[ 3 ], bbox1[ 0 ] : bbox1[ 2 ] ] = imgCutNp1
            imgNpAlt[ bbox2[ 1 ] : bbox2[ 3 ], bbox2[ 0 ] : bbox2[ 2 ] ] = imgCutNp2 
            
            myAugImg = Image.fromarray( imgNpAlt )
            myAugImg.save( f"./png_err/{ len( paths ) * 2 + 1 + i}.png" )
            
            newList.append( [ f"./png_err/{ len( paths ) * 2 + 1 + i}.png", bboxO[ 0 ], bboxO[ 1 ], bboxO[ 2 ], bboxO[ 3 ] ] )
            newList.append( [ f"./png_err/{ len( paths ) * 2 + 1 + i}.png", bbox1[ 0 ], bbox1[ 1 ], bbox1[ 2 ], bbox1[ 3 ] ] )
            newList.append( [ f"./png_err/{ len( paths ) * 2 + 1 + i}.png", bbox2[ 0 ], bbox2[ 1 ], bbox2[ 2 ], bbox2[ 3 ] ] )
            
        
            
        data_df_append = pd.DataFrame( newList, columns = [ "path", "xmin", "ymin", "xmax", "ymax" ] )
        data_df = pd.read_csv( os.path.join( path_prep, "bboxes_s2.csv" ), sep = ";", decimal = "," )
        
        data_df = data_df.append( data_df_append )
        
        data_df.to_csv( "bboxes_s3.csv", sep = ";", decimal = ",", index = False ) 
        
    if create_bbox_imgs == True:
        
        data_df = pd.read_csv( "bboxes_s5.csv", sep = ";", decimal = "," )

        uniqueImgs = data_df.path.unique( )
        
        for i, Img in enumerate( uniqueImgs ):
            fig = plt.figure( figsize = ( 5, 5 ), dpi = 100 )
            ax = fig.add_axes( [ 0, 0, 1, 1 ] )
            image = plt.imread( Img )
            plt.imshow( image, cmap = "gray" )
            plt.axis( "off" )
            
            entrys = data_df.path == Img
            smallDF = data_df.loc[ entrys ]
            
            for index, row in smallDF.iterrows():
                xmin = row.xmin
                ymin = row.ymin
                xmax = row.xmax
                ymax = row.ymax 
                
                width = xmax - xmin
                height = ymax - ymin
                
                edgecolor = 'r'
                
                rect = patches.Rectangle( ( xmin, ymin ), width, height, edgecolor = edgecolor, facecolor = 'none' )
                ax.add_patch( rect )
                
            fig.tight_layout( )
            
            plt.savefig( f"png_err_labeled/{ i + 1 }.png", bbox_inches = 'tight', pad_inches = 0.0 )
            plt.close( fig )

    