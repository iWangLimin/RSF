import gdal
import numpy as np
from config import DatasetConfig as DC
import tensorflow as tf
import glob
import random
import cv2
import os
def up_sample(ms,scale=4):
    return cv2.resize(ms,dsize=(0,0),fx=scale,fy=scale,\
        interpolation=cv2.INTER_CUBIC)
def down_sample(img,scale=2):
    smoothed=cv2.GaussianBlur(img,ksize=(0,0),sigmaX=1,sigmaY=1)
    return cv2.resize(smoothed,dsize=(0,0),fx=1/scale,fy=1/scale,\
        interpolation=cv2.INTER_CUBIC)
def read_tif(name):
    left_cut=60
    ms_path,pan_path=glob.glob(DC.remote_root+'/'+name+'-MSS?.tiff')[0],\
        glob.glob(DC.remote_root+'/'+name+'-PAN?.tiff')[0]
    ms,pan=gdal.Open(ms_path),gdal.Open(pan_path)
    cols,rows=ms.RasterXSize,ms.RasterYSize
    col_patch,row_patch=(cols-DC.MS_crop_size)//DC.MS_crop_step+1,\
        (rows-DC.MS_crop_size-left_cut)//DC.MS_crop_step+1
    return ms,pan,col_patch,row_patch
def read_patch(idx,ms,pan,col_patch_num,row_patch_num):
    left_cut=60
    band=ms.RasterCount
    patch_row_idx,patch_col_idx=idx//col_patch_num,idx%col_patch_num
    patch_left_up=(patch_row_idx*DC.MS_crop_step,patch_col_idx*DC.MS_crop_step+left_cut)
    ms_patch,pan_patch=np.zeros((DC.MS_crop_size,DC.MS_crop_size,4),np.uint16),\
        np.zeros((DC.MS_crop_size*4,DC.MS_crop_size*4,1),np.uint16)
    for b in range(1,band+1):
        ms_patch[:,:,b-1]=ms.GetRasterBand(b).ReadAsArray(patch_left_up[1],patch_left_up[0],DC.MS_crop_size,DC.MS_crop_size)
    pan_patch[:,:,0]=pan.GetRasterBand(1).ReadAsArray(patch_left_up[1]*4,patch_left_up[1]*4,DC.MS_crop_size*4,DC.MS_crop_size*4)
    return ms_patch,pan_patch
def sampler(col_patch,row_patch):
    num=col_patch*row_patch
    all_idx=set(range(num))
    val_idx=random.sample(all_idx,500)
    all_idx=all_idx-set(val_idx)
    test_idx=random.sample(all_idx,500)
    train_idx=all_idx-set(test_idx)
    return train_idx,val_idx,test_idx
def write_patch(ms,pan,writer,cata='train'):
    input_pan=down_sample(pan,scale=4)
    fusion_1=down_sample(ms)
    input_ms=down_sample(fusion_1)
    # if ms_upsample:
    #     input_ms=up_sample(input_ms)
    ms_feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_ms.tostring()]))
    pan_feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[input_pan.tostring()]))
    fusion_2_feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms.tostring()]))
    if cata=='train':
        fusion_1_feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[fusion_1.tostring()]))
        features=tf.train.Features(feature={'ms':ms_feature,'pan':pan_feature,\
                                'fusion_1':fusion_1_feature,'fusion_2':fusion_2_feature})
    else:
        features=tf.train.Features(feature={'ms':ms_feature,'pan':pan_feature,\
                                'fusion':fusion_2_feature})
    example=tf.train.Example(features=features)
    writer.write(example.SerializeToString())
if __name__ == "__main__":
    for img_idx,img in enumerate(DC.img_names):
        img_idx+=1
        ms,pan,col_patch_num,row_patch_num=read_tif(img)
        sample_idx=sampler(col_patch_num,row_patch_num)
        for i,catagory in enumerate(['train','val','test']):
            record_name=os.path.join(os.getcwd(),'lapdataset/%s/%d.tfrecord'%(catagory,img_idx))
            if not os.path.exists(os.path.dirname(record_name)):
                os.mkdir(os.path.dirname(record_name))
            with tf.io.TFRecordWriter(record_name) as writer:
                for idx in sample_idx[i]:
                    ms_patch,pan_patch=read_patch(idx,ms,pan,col_patch_num,row_patch_num)
                    write_patch(ms_patch,pan_patch,writer,cata=catagory)
        

