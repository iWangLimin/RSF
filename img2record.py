import gdal
import numpy as np
from config import DatasetConfig as DC
import tensorflow as tf
import glob
import random
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
    ms_patch,pan_patch=np.zeros((DC.MS_crop_size,DC.MS_crop_size,4),np.float32),\
        np.zeros((DC.MS_crop_size*4,DC.MS_crop_size*4,1),np.float32)
    for b in range(1,band+1):
        ms_patch[:,:,b-1]=ms.GetRasterBand(b).ReadAsArray(col,row,DC.MS_crop_size,DC.MS_crop_size)
    pan_patch[:,:,0]=pan.GetRasterBand(1).ReadAsArray(col*4,row*4,DC.MS_crop_size*4,DC.MS_crop_size*4)
    return ms_patch/DC.Max_Pixel,pan_patch/DC.Max_Pixel
def sampler(col_patch,row_patch):
    num=col_patch*row_patch
    all_idx=set(range(num))
    train_idx=random.sample(all_idx,int(num*DC.train_scale))
    all_idx=all_idx-set(train_idx)
    val_idx=random.sample(all_idx,int(num*DC.validate_scale))
    test_idx=all_idx-set(val_idx)
    return train_idx,val_idx,test_idx
def write_patch(ms,pan,writer):
    ms_feature,pan_feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[ms.tostring()])),\
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[pan.tostring()]))
    features=tf.train.Features(feature={'ms':ms_feature,'pan':pan_feature})
    example=tf.train.Example(features=features)
    writer.write(example.SerializeToString())
if __name__ == "__main__":
    for img_idx,img in enumerate(DC.img_names[1:]):
        img_idx+=1
        record_names=['train%d_%d'%(DC.MS_crop_size,img_idx),'val%d_%d'%(DC.MS_crop_size,img_idx),\
            'test%d_%d'%(DC.MS_crop_size,img_idx)]
        ms,pan,col_patch_num,row_patch_num=read_tif(img)
        sample_idx=sampler(col_patch_num,row_patch_num )
        for i,record_name in enumerate(record_names):
            with tf.io.TFRecordWriter(record_name) as writer:
                for idx in sample_idx[i]:
                    ms_patch,pan_patch=read_patch(idx,ms,pan,col_patch_num,row_patch_num)
                    write_patch(ms_patch,pan_patch,writer)
        

