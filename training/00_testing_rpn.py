# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 18:27:56 2020

@author: Gunnar
"""



from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_vgg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_iv3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as preprocess_irv2
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_res50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_mnet

from keras.preprocessing import image
from keras.models import Input, Model
from keras.layers import Conv2D
import numpy as np
import numpy.random as npr
import os

train_path = os.getcwd( )
os.chdir( "../" )
main_path = os.getcwd( )

#### Function Section
def generate_anchors( base_width = 16, base_height = 16, ratios = [ 2 / 3, 1, 1.5 ],
                      scales = np.asarray( [ 0.8, 1.2, 1.7 ] ) ):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, w_stride-1, h_stride-1) window.
    """

    base_anchor = np.array( [1, 1, base_width, base_height] ) - 1
    ratio_anchors = _ratio_enum( base_anchor, ratios )
    anchors = np.vstack( [ _scale_enum( ratio_anchors[ i, : ], scales )
                          for i in range( ratio_anchors.shape[ 0 ] ) ] )
    return anchors

def _ratio_enum( anchor, ratios ):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs( anchor )
    size = w * h
    size_ratios = size / ratios
    ws = np.round( np.sqrt( size_ratios ) )
    hs = np.round( ws * ratios )
    anchors = _mkanchors( ws, hs, x_ctr, y_ctr )
    return anchors

def _whctrs( anchor ):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[ 2 ] - anchor[ 0 ] + 1
    h = anchor[ 3 ] - anchor[ 1 ] + 1
    x_ctr = anchor[ 0 ] + 0.5 * ( w - 1 )
    y_ctr = anchor[ 1 ] + 0.5 * ( h - 1 )
    return w, h, x_ctr, y_ctr

def _scale_enum( anchor, scales ):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs( anchor )
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors( ws, hs, x_ctr, y_ctr )
    return anchors

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


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = boxes.astype( int )
    N = boxes.shape[ 0 ]
    K = query_boxes.shape[ 0 ]

    overlaps = np.zeros( ( N, K ), dtype=np.float )

    for k in range( K ):
        box_area = ( ( query_boxes[ k, 2 ] - query_boxes[ k, 0 ] + 1 ) * (query_boxes[k, 3] - query_boxes[k, 1] + 1) )
        
        for n in range( N ):
            iw = ( min( boxes[ n, 2 ], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1 )
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float( ( boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih )
                    overlaps[ n, k ] = iw * ih / ua

    return overlaps

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
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


#### Bildverarbeitung
imgPath = os.path.join( train_path, "setup/test.jpg" )
oWidth = 1953
oHeight = 1297
iWidth = 256
iHeight = 512

wRatio = iWidth / oWidth
hRatio = iHeight / oHeight 

xminI = ( np.array( [ 656, 666, 790 ] ) * wRatio ).astype( int ).reshape( ( 3, 1 ) )
yminI = ( np.array( [ 544, 227, 556 ] ) * hRatio ).astype( int ).reshape( ( 3, 1 ) )
xmaxI = ( np.array( [ 1158, 804, 1926 ] ) * wRatio ).astype( int ).reshape( ( 3, 1 ) )
ymaxI = ( np.array( [ 1139, 358, 686 ] ) * hRatio ).astype( int ).reshape( ( 3, 1 ) )


bbox = np.hstack( [ xminI, yminI, xmaxI, ymaxI ] )

img = image.load_img( imgPath, target_size = ( iHeight, iWidth ) )
# img.show( )
x = image.img_to_array( img )
x = np.expand_dims( x, axis = 0 )


use_model = 1
#### InceptionV3 - default 299x299 | output - 512 -> 14x14 | r = 36.5
if use_model == 0:
    modelIV3 = InceptionV3( weights = 'imagenet', include_top = False )
    model = modelIV3
    preImg = preprocess_iv3( x )

### VGG - default 224x224 | output - 512 -> 32x32 | r = 16
if use_model == 1:
    modelvgg = VGG16( weights = 'imagenet', include_top = False )
    inputVGG = Model( inputs = modelvgg.input, outputs = modelvgg.get_layer( 'block5_conv1' ).output )
    model = inputVGG
    preImg = preprocess_vgg( x )
    
#### Inception ResNet V2 - default 299x299 | output - 512 -> 14x14 | r = 36.5
if use_model == 2:
    modelirv2 = InceptionResNetV2( weights = 'imagenet', include_top = False )
    model = modelirv2
    preImg = preprocess_irv2( x )
    
#### Resnet50 - default - 224x224 | output - 512 -> 16x16 | r = 32
if use_model == 3:
    modelres = ResNet50( weights = 'imagenet', include_top = False )
    model = modelres
    preImg = preprocess_res50( x )
    
#### MobilnetV2 - default 224x224 | output - 512 -> 16x16 | r = 32
if use_model == 4:
    modelmnet = MobileNetV2( weights = 'imagenet', include_top = False )
    model = modelmnet
    preImg = preprocess_mnet( x )
    
    
output = model.predict( preImg )

k = 9
rpn_fg_frac = 0.3
rpn_batch_size = 256 
bg_fg_frac = 2
num_fg = int( rpn_fg_frac * rpn_batch_size )

shapeOut = output.shape
fHeight = shapeOut[ 1 ]
fWidth = shapeOut[ 2 ]
num_features = fHeight * fWidth 
tot_anchors = num_features * k
w_stride = iWidth / fWidth 
h_stride = iHeight / fHeight

shiftX = np.arange( 0, fWidth ) * w_stride
shiftY = np.arange( 0, fHeight ) * h_stride 

shiftX, shiftY = np.meshgrid( shiftX, shiftY )
shifts = np.vstack( ( shiftX.ravel( ), shiftY.ravel( ), shiftX.ravel( ), shiftY.ravel( ) ) ).transpose( )

base_anchors = generate_anchors( base_width = w_stride, base_height = h_stride )
all_anchors = base_anchors.reshape( (1, 9, 4 ) ) + shifts.reshape( 
    (1, num_features, 4 ) ).transpose( ( 1, 0, 2 ) )

border = 0
all_anchors = all_anchors.reshape( ( num_features * k, 4 ) )
inds_inside = np.where( 
    ( all_anchors[ :, 0 ] >= -border ) &
    ( all_anchors[ :, 1 ] >= -border ) &
    ( all_anchors[ :, 2 ] < iWidth + border ) &
    ( all_anchors[ :, 3 ] < iHeight + border ) )[ 0 ]

anchors = all_anchors[ inds_inside ]

overlaps = bbox_overlaps( anchors, bbox )
argmax_overlaps = overlaps.argmax( axis = 1 )
max_overlaps = overlaps[ np.arange( len( inds_inside ) ), argmax_overlaps ]

gt_argmax_overlaps = overlaps.argmax( axis = 0 )
gt_max_overlaps = overlaps[ gt_argmax_overlaps, np.arange( overlaps.shape[ 1 ] ) ]
gt_argmax_overlaps = np.where( overlaps == gt_max_overlaps )[ 0 ]

labels = np.empty( ( len( anchors ), ), dtype = np.float32 )
labels.fill( -1 )

labels[ gt_argmax_overlaps ] = 1 
labels[ max_overlaps >= .3 ] = 1
labels[ max_overlaps <= 0.1 ] = 0

fg_inds = np.where( labels == 1 )[ 0 ]
if len( fg_inds ) > num_fg:
    disable_inds = npr.choice( fg_inds, size = len( fg_inds ) - num_fg, replace = False )
    labels[ disable_inds ] = -1
    
fg_inds = np.where( labels == 1 )[ 0 ]
    
num_bg = int( len( fg_inds ) * bg_fg_frac )
bg_inds = np.where( labels == 0 )[ 0 ]
if len( bg_inds ) > num_bg:
    disable_inds = npr.choice( bg_inds, size = len( bg_inds ) - num_bg, replace = False )
    labels[ disable_inds ] = -1
    
batch_inds = inds_inside[ labels != -1 ]
batch_inds = ( batch_inds / k ).astype( np.int )

full_labels = unmap( labels, tot_anchors, inds_inside, fill = -1 )

batch_label_target = full_labels.reshape( -1, 1, 1, 1 * k )[ batch_inds ]
bbox_targets = np.zeros( ( len(inds_inside ), 4 ), dtype = np.float32 )

pos_anchors = all_anchors[ inds_inside[ labels == 1 ] ]

bboxes_label = bbox[ argmax_overlaps, : ][ labels == 1 ]

bbox_targets = bbox_transform( pos_anchors, bboxes_label )

bbox_targets = unmap( bbox_targets, tot_anchors, inds_inside[ labels == 1 ], fill = 0 )

batch_bbox_targets = bbox_targets.reshape( -1, 1, 1, 4 * k )[ batch_inds ]

padded_fcmap = np.pad( output, ( ( 0, 0 ), ( 1, 1 ), ( 1, 1 ), ( 0, 0 ) ), mode = 'constant' )

padded_fcmap = np.squeeze( padded_fcmap )

batch_tiles = [ ]

for ind in batch_inds: 
    x = ind % fWidth 
    y = int( ind/fWidth )
    fc_3x3 = padded_fcmap[ y : y + 3, x : x + 3, : ]
    batch_tiles.append( fc_3x3 )

out = ( np.asarray( batch_tiles ), batch_label_target.tolist( ), batch_bbox_targets.tolist( ) )
#### RPN

feature_map = Input( shape = ( None, None, 2048 ) )
convolution_3x3 = Conv2D( filters = 512, kernel_size = ( 3, 3 ), name = "3x3" )( feature_map )
output_deltas = Conv2D( filters = 4 * k, kernel_size = ( 1, 1 ), activation = "linear", 
                       kernel_initializer = "uniform", name = "deltas1" )( convolution_3x3 )
output_scores = Conv2D( filters = 1 * k, kernel_size=(1, 1), activation = "sigmoid", 
                       kernel_initializer = "uniform", name = "scores1" )( convolution_3x3 )
model = Model( inputs = [ feature_map ], outputs = [ output_scores, output_deltas ] )
# model.summary( )



#### Bounding Boxes