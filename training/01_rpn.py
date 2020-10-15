# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:29:53 2020

@author: Gunna
"""

import numpy as np
import numpy.random as npr
import os
npr.seed( 42 )
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input 
from keras.models import Input, Model
from keras.layers import Conv2D

train_path = os.getcwd( )
os.chdir( "../" )
main_path = os.getcwd( )

def gen_anchors( base_width, base_height, ratios = [ 2/3, 1, 1.5 ],
                     scales = np.asarray( [ 0.8, 1.1, 1.6 ] ) ):
    """
    Erzeugt Anker für ein Fenster
    """
    px_1_value = 50
    scale_50 = ( px_1_value / base_width + px_1_value / base_height ) / 2.0
    scales = scales * scale_50
    # base_anchor = np.array( [ ( window_width - 1 ) / 2 - base_width / 2, ( window_height - 1 ) / 2 - base_width / 2, 
    #                          ( window_width - 1 ) / 2 + base_width / 2, ( window_height - 1 ) / 2 + base_width / 2 ] )
    base_anchor = np.array( [ 1, 1, base_width, base_height ] ) - 1
    ratio_anchors = gen_ratio_anchors( base_anchor, ratios )
    anchors = np.vstack( [ gen_scale_anchors( ratio_anchors[ i, : ], scales )
                         for i in range( ratio_anchors.shape[ 0 ] ) ] )
    return anchors

def gen_ratio_anchors( anchor, ratios ):
    """
    erzeugt zu eingegebenen Anker die Anker mit diversen Seitenverhaeltnissen
    """

    w, h, x_ctr, y_ctr = whctrs( anchor )
    size = w * h
    size_ratios = size / ratios
    ws = np.sqrt(size_ratios)
    hs = ws * ratios
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def mkanchors( ws, hs, x_ctr, y_ctr ):
    """
    erstellt Anker abhängig von breite, höhe und Zentralkoordinaten
    """

    ws = ws[ :, np.newaxis ]
    hs = hs[ :, np.newaxis ]
    anchors = np.hstack( ( x_ctr - 0.5 * ( ws - 1 ),
                         y_ctr - 0.5 * ( hs - 1 ),
                         x_ctr + 0.5 * ( ws - 1 ),
                         y_ctr + 0.5 * ( hs - 1 ) ) )
    return anchors

def whctrs( anchor ):
    """
    Gibt breite, höhe und die Zentrumskoordinaten eines Ankers aus
    """

    w = anchor[ 2 ] - anchor[ 0 ] + 1
    h = anchor[ 3 ] - anchor[ 1 ] + 1
    x_ctr = anchor[ 0 ] + 0.5 * ( w - 1 )
    y_ctr = anchor[ 1 ] + 0.5 * ( h - 1 )
    return w, h, x_ctr, y_ctr

def gen_scale_anchors( anchor, scales ):
    """
    erstellt aus den Ankern die skalierten Anker
    """

    w, h, x_ctr, y_ctr = whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def bbox_IoU( anchors, gt_bboxes ):
    """
    Parameters
    ----------
    anchors: (N, 4) beinhaltet alle Anker die innerhalb des Bildes liegen
    gt_boxes: (K, 4) beinhaltet alle GT Boxen
    Returns
    -------
    IoU: gibt den IoU wert für jeden Anker und jede GT Box aus
    """
    anchors = anchors.astype( int )
    N = anchors.shape[ 0 ]
    K = gt_bboxes.shape[ 0 ]

    IoU = np.zeros( ( N, K ), dtype=np.float )

    for k in range( K ):
        box_area = ( ( gt_bboxes[ k, 2 ] - gt_bboxes[ k, 0 ] + 1 ) * ( gt_bboxes[ k, 3 ] - gt_bboxes[ k, 1 ] + 1 ) )
        
        for n in range( N ):
            iw = ( min( anchors[ n, 2 ], gt_bboxes[k, 2]) - max(anchors[n, 0], gt_bboxes[k, 0]) + 1 )
            if iw > 0:
                ih = (min(anchors[n, 3], gt_bboxes[k, 3]) - max(anchors[n, 1], gt_bboxes[k, 1]) + 1)

                if ih > 0:
                    ua = float( ( anchors[n, 2] - anchors[n, 0] + 1) * (anchors[n, 3] - anchors[n, 1] + 1) + box_area - iw * ih )
                    IoU[ n, k ] = iw * ih / ua

    return IoU

def unmap( data, count, inds, fill = 0 ):
    """ 
    mapt das neue Subset auf das alte vollständige set zurück
    """
    if len( data.shape ) == 1:
        ret = np.empty( ( count, ), dtype=np.float32 )
        ret.fill( fill )
        ret[ inds ] = data
    else:
        ret = np.empty( ( count, ) + data.shape[ 1: ], dtype=np.float32 )
        ret.fill( fill )
        ret[ inds, : ] = data
    return ret

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))

    targets = np.transpose(targets)

    return targets

vgg = VGG16( weights = "imagenet", include_top = False )
model = Model( inputs = vgg.input, outputs = vgg.get_layer( 'block5_conv1' ).output )
k = 9

def create_batch( file, gt_bboxes, cnn ):
    
    fg_bg_ratio = 3
    
    #### Bild Einlesen
    img = load_img( file )
    img_width = np.shape( img )[ 1 ]
    img_height = np.shape( img )[ 0 ] 
    img = img_to_array( img )
    img = np.expand_dims( img, 0 )
    
    #### Feature Map erstellen
    img = preprocess_input( img )
    feature_map = cnn.predict( img )
    feat_width = feature_map.shape[ 2 ]
    feat_height = feature_map.shape[ 1 ]
    num_feat = feat_width * feat_height
    total_anchors = num_feat * k
    
    #### Feature Map auf originales Bild mappen
    stride_width = img_width / feat_width
    stride_height = img_height / feat_height
    img_x_marks = np.arange( 0, feat_width ) * stride_width 
    img_y_marks = np.arange( 0, feat_height ) * stride_height 
    img_x_marks, img_y_marks = np.meshgrid( img_x_marks, img_y_marks )
    marks = np.vstack( ( img_x_marks.ravel( ), img_y_marks.ravel( ), img_x_marks.ravel( ), img_y_marks.ravel( ) ) ).transpose( )
    
    
    #### Anker für normales Bild erstellen
    base_anchors = gen_anchors( base_width = stride_width, base_height = stride_height )
    all_anchors = ( base_anchors.reshape( ( 1, k, 4 ) ) + marks.reshape( ( 1, num_feat, 4 ) ).transpose( ( 1, 0, 2 ) ) )
    all_anchors = all_anchors.reshape( ( total_anchors, 4 ) )
    
    #### IoU Sektion
    border = 0
    i_inside = np.where( 
        ( all_anchors[ :, 0 ] >= -border ) &
        ( all_anchors[ :, 1 ] >= -border ) &
        ( all_anchors[ :, 2 ] < img_width + border ) &
        ( all_anchors[ :, 3 ] < img_height + border ) 
        )[ 0 ]
    usefull_anchors = all_anchors[ i_inside ]
    IoU = bbox_IoU( anchors = usefull_anchors, gt_bboxes = gt_bboxes ) 
    argmax_IoU = IoU.argmax( axis = 1 )
    max_IoU = IoU[ np.arange( len( i_inside ) ), argmax_IoU ]
    gt_argmax_IoU = IoU.argmax( axis = 0 )
    gt_max_IoU = IoU[ gt_argmax_IoU, np.arange( IoU.shape[ 1 ] ) ]
    gt_argmax_IoU = np.where( IoU == gt_max_IoU )[ 0 ]
    
    #### Label Sektion | 1 = fg, 0 = bg, -1 = ignore
    labels = np.empty( ( len( i_inside ), ), dtype = np.float32 )
    labels.fill( -1 )
    labels[ gt_argmax_IoU ] = 1
    labels[ max_IoU >= 0.5 ] = 1
    labels[ max_IoU <= 0.1 ] = 0
    fg_i = np.where( labels == 1 )[ 0 ]
    max_num_bg = int( len( fg_i ) * fg_bg_ratio )
    bg_i = np.where( labels == 0 )[ 0 ]
    if len( bg_i ) > max_num_bg:
        disable_i = npr.choice( bg_i, size = ( len( bg_i ) - max_num_bg ), replace = False )
        labels[ disable_i ] = -1
    full_labels = unmap( labels, total_anchors, i_inside, fill = -1 )
    
    batch_i = i_inside[ labels != -1 ]
    batch_i = ( batch_i / k ).astype( np.int )
    batch_label_targets = full_labels.reshape( -1, 1, 1, k )[ batch_i ]
    pos_anchor = all_anchors[ i_inside[ labels == 1 ] ]
    bbox_targets = bbox_transform( pos_anchor, gt_bboxes[ argmax_IoU, : ][ labels == 1 ] )
    bbox_targets = unmap( bbox_targets, total_anchors, i_inside[ labels == 1 ], fill = 0 )
    batch_bbox_targets = bbox_targets.reshape( -1, 1, 1, 4 * k )[ batch_i ]
    
    #### Feature Map - Input Sektion
    padded_fm = np.pad( feature_map, ( ( 0, 0 ), ( 1, 1 ), ( 1, 1 ), ( 0, 0 ) ), mode = 'constant' )
    padded_fm = np.squeeze( padded_fm )
    batch_tiles = [ ]
    for i in batch_i:
        x = i % feat_width
        y = int( i / feat_width )
        fm_tile = padded_fm[ y: y + 3, x : x + 3, : ]
        batch_tiles.append( fm_tile )
    return np.asarray( batch_tiles ), batch_label_targets.tolist( ), batch_bbox_targets.tolist( )
    


#### RPN Sektion
import keras.backend as K
import tensorflow as tf

def loss_cls(y_true, y_pred):
    condition = K.not_equal(y_true, -1)
    indices = tf.where(condition)

    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss = K.binary_crossentropy(target, output)
    return K.mean(loss)

def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.5
    x   = K.abs(y_true - y_pred)
    x   = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)


test_bboxes = np.array( [ 
    [ 92, 221, 164, 284 ],
    [ 357, 402, 428, 458 ],
    [ 302, 149,	373, 205 ]
    ] )
test_tiles, test_labels, test_bboxes = create_batch( os.path.join( train_path, "setup/2010.png" ), gt_bboxes = test_bboxes, cnn = model )


feature_map_tile = Input( shape = ( None, None, test_tiles.shape[ 3 ] ) )
conv_3x3 = Conv2D(
    filters = 512,
    kernel_size = ( 3, 3 ),
    padding = 'same',
    name = "3x3"
    )( feature_map_tile )

out_deltas = Conv2D( 
    filters = 4 * k,
    kernel_size = ( 1, 1 ),
    activation = "linear",
    kernel_initializer = "uniform" ,
    name = "deltas" 
    )( conv_3x3 )

out_scores = Conv2D(
    filters = k, 
    kernel_size = ( 1, 1 ),
    activation = "sigmoid",
    kernel_initializer = "uniform",
    name = "scores"
    )( conv_3x3 )

rpn = Model( inputs = [ feature_map_tile ], outputs = [ out_scores, out_deltas ] )
rpn.compile( optimizer = "adam", loss = { "scores" : loss_cls, "deltas" : smoothL1 } )

#### Data Section
import glob
import pandas as pd

imgs_path = os.path.join( main_path, "prep/png/" )

image_list = glob.glob( imgs_path + "*.png" )
df = pd.read_csv( os.path.join( main_path, "prep/trainLabel4.csv" ), sep = ";", index_col = 0 )
train_test_split = 0.8


df_images = df[ "path" ].values.tolist( )
df_images = np.array( [ int( path.split( '\\' )[ -1 ].split( "." )[ 0 ] ) for path in df_images ] )
all_bboxes = df[ [ "xmin", "ymin", "xmax", "ymax" ] ].values

data = [ ]

for i in np.arange( 1, np.max( df_images ) + 1 ):
    index = np.where( df_images == i )[ 0 ]
    bboxes = all_bboxes[ index, : ]
    path = os.path.join( imgs_path + f"{ i }.png" )
    data.append( [ path, bboxes ] )
    
img_nmbrs = np.arange( 1, len( data ) + 1 )
train_set = npr.choice( img_nmbrs, int( len( data ) * train_test_split ), replace = False )
test_set = np.delete( img_nmbrs, train_set - 1 )
npr.shuffle( train_set )
npr.shuffle( test_set )
train_list = [ data[ i - 1 ] for i in train_set ]
test_list = [ data[ i - 1 ] for i in train_set ]



def input_gen( ):
    batch_size = 256
    batch_tiles = [ ]
    batch_labels = [ ]
    batch_bboxes = [ ]
    
    while True:
        
        for entry in train_list:
            
            tiles, labels, bboxes = create_batch( file = entry[ 0 ], gt_bboxes = entry[ 1 ], cnn = model )
            
            for i in range( len( tiles ) ):
                
                batch_tiles.append( tiles[ i ] )
                batch_labels.append( labels[ i ] )
                batch_bboxes.append( bboxes[ i ] )
                
                if ( len( batch_tiles ) == batch_size ):
                    
                    a = np.asarray( batch_tiles )
                    b = np.asarray( batch_labels )
                    c = np.asarray( batch_bboxes )
                    
                    yield a, [ b, c ]
                    
                    batch_tiles = [ ]
                    batch_labels = [ ]
                    batch_bboxes = [ ]
        

# from keras.callbacks import ModelCheckpoint
# checkpointer = ModelCheckpoint( filepath = "rpn/weights_rpn.hdf5", verbose = 1, save_best_only = True )
# rpn.fit_generator( input_gen( ), steps_per_epoch = 10, epochs = 10, callbacks = [ checkpointer ] )
