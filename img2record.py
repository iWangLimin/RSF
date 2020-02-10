import gdal
import numpy as np
from config import DatasetConfig as DC
import tensorflow as tf
import glob
import random
import cv2
import os
def down_sample(img,scale=4):
    return cv2.resize(img,dsize=(0,0),fx=1/scale,fy=1/scale,\
        interpolation=cv2.INTER_CUBIC)
def read_tif(name):
    ms_path,pan_path=glob.glob(DC.remote_root+'/'+name+'-MSS?.tiff')[0],\
        glob.glob(DC.remote_root+'/'+name+'-PAN?.tiff')[0]
    ms,pan=gdal.Open(ms_path),gdal.Open(pan_path)
    cols,rows=ms.RasterXSize,ms.RasterYSize
    col_patch,row_patch=(cols-DC.MS_crop_size)//DC.MS_crop_step+1,\
        (rows-DC.MS_crop_size)//DC.MS_crop_step+1
    return ms,pan,col_patch,row_patch
def read_patch(idx,ms,pan,col_patch_num,row_patch_num):
    band=ms.RasterCount
    row,col=idx//col_patch_num,idx%col_patch_num
    ms_patch,pan_patch=np.zeros((DC.MS_crop_size,DC.MS_crop_size,4),np.uint16),\
        np.zeros((DC.MS_crop_size*4,DC.MS_crop_size*4,1),np.uint16)
    for b in range(1,band+1):
        ms_patch[:,:,b-1]=ms.GetRasterBand(b).ReadAsArray(col,row,DC.MS_crop_size,DC.MS_crop_size)
    pan_patch[:,:,0]=pan.GetRasterBand(1).ReadAsArray(col*4,row*4,DC.MS_crop_size*4,DC.MS_crop_size*4)
    return ms_patch,pan_patch
def sampler(col_patch,row_patch):
    num=col_patch*row_patch
    all_idx=set(range(num))
    train_idx=random.sample(all_idx,int(num*DC.train_scale))
    all_idx=all_idx-set(train_idx)
    val_idx=random.sample(all_idx,int(num*DC.validate_scale))
    test_idx=all_idx-set(val_idx)
    return train_idx,val_idx,test_idx
def write_patch(ms,pan,writer):
    input_ms,input_pan=down_sample(ms),down_sample(pan)
    ms_feature,pan_feature,fusion=\
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_ms.tostring()])),\
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_pan.tostring()])),\
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms.tostring()]))
    features=tf.train.Features(feature={'ms':ms_feature,'pan':pan_feature,\
                                'fusion':fusion})
    example=tf.train.Example(features=features)
    writer.write(example.SerializeToString())
if __name__ == "__main__":
    for img_idx,img in enumerate(DC.img_names):
        img_idx+=1
        ms,pan,col_patch_num,row_patch_num=read_tif(img)
        sample_idx=sampler(col_patch_num,row_patch_num)
        for i,catagory in enumerate(['train','val','test']):
            record_name=os.path.join(os.getcwd(),'dataset/%s/%d.tfrecord'%(catagory,img_idx))
            with tf.io.TFRecordWriter(record_name) as writer:
                for idx in sample_idx[i]:
                    ms_patch,pan_patch=read_patch(idx,ms,pan,col_patch_num,row_patch_num)
                    write_patch(ms_patch,pan_patch,writer)
        

