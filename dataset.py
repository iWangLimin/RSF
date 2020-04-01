from config import DatasetConfig as DC,TrainingConfig as TC
import tensorflow as tf
import os
import numpy as np
import cv2
class Dataset():
    def __init__(self,train_set,dir_path,ms_upsample=False):
        record_files=os.listdir(dir_path)
        self.dataset=tf.data.TFRecordDataset(os.path.join(dir_path,record_files[0]))
        for tfrecord in record_files[1:]:
            self.dataset=self.dataset.concatenate(tf.data.TFRecordDataset(
                os.path.join(dir_path,tfrecord)))
        self.train_set=train_set
        self.ms_upsample=ms_upsample
    def parse_example(self,example):
        if self.train_set:
            feature_description={'ms':tf.io.FixedLenFeature([],tf.string),\
                        'pan':tf.io.FixedLenFeature([],tf.string),\
                        'fusion_1':tf.io.FixedLenFeature([],tf.string),\
                        'fusion_2':tf.io.FixedLenFeature([],tf.string)}
        else:
            feature_description={'ms':tf.io.FixedLenFeature([],tf.string),\
                                'pan':tf.io.FixedLenFeature([],tf.string),\
                                'fusion':tf.io.FixedLenFeature([],tf.string)}
        parsed_example=tf.io.parse_single_example(example,feature_description)
        ms=tf.decode_raw(parsed_example['ms'],tf.uint16)
        pan = tf.decode_raw(parsed_example['pan'], tf.uint16)
        if self.train_set:
            fusion_1= tf.decode_raw(parsed_example['fusion_1'], tf.uint16)
            fusion_2= tf.decode_raw(parsed_example['fusion_2'], tf.uint16)
        else:
            fusion= tf.decode_raw(parsed_example['fusion'], tf.uint16)
        ms=tf.reshape(ms,[DC.MS_crop_size//4,DC.MS_crop_size//4,4])
        pan=tf.reshape(pan,[DC.MS_crop_size,DC.MS_crop_size,1])
        # if self.ms_upsample:
        #     ms=tf.expand_dims(ms,0)
        #     ms=tf.image.resize_bicubic(ms,[DC.MS_crop_size,DC.MS_crop_size])
        #     ms=tf.squeeze(ms)
        if self.train_set:
            fusion_1=tf.reshape(fusion_1,[DC.MS_crop_size//2,DC.MS_crop_size//2,4])
            fusion_2=tf.reshape(fusion_2,[DC.MS_crop_size,DC.MS_crop_size,4])
        else:
            fusion=tf.reshape(fusion,[DC.MS_crop_size,DC.MS_crop_size,4])
        if self.train_set:
            return ms,pan,fusion_1,fusion_2
        return ms,pan,fusion
    def normalize(self,*args):
        n_imgs=[]
        for img in args:
            float_img=tf.cast(img,tf.float32)
            n_imgs.append((float_img-DC.Min_Pixel)/(DC.Max_Pixel-DC.Min_Pixel))
        return n_imgs
    def batch(self):
        self.dataset=self.dataset.map(self.parse_example,num_parallel_calls=4)
        self.dataset=self.dataset.map(self.normalize,num_parallel_calls=4)
        if self.train_set:
            self.dataset=self.dataset.shuffle(20000).\
                repeat().batch(batch_size=TC.batch_size)
        else:
            self.dataset=self.dataset.batch(batch_size=TC.batch_size)
    def get_iter(self):
        self.batch()
        return tf.data.make_initializable_iterator(self.dataset)
# if __name__ == "__main__":
#     train_set=Dataset(True,'dataset/train')
#     val_set=Dataset(False,'dataset/val')
#     train_set=train_set.dataset.map(train_set.parse_example)
#     val_set=val_set.dataset.map(val_set.parse_example)
#     train_iter=tf.data.make_initializable_iterator(train_set)
#     val_iter=tf.data.make_initializable_iterator(val_set)
#     train_ms,train_pan,_,train_fusion=train_iter.get_next()
#     val_ms,val_pan,val_fusion=val_iter.get_next()
#     with tf.Session() as sess:
#         sess.run([train_iter.initializer,val_iter.initializer])
#         # try:
#         #     i=0
#         #     while True:
#         #         ms,pan,fusion=sess.run([train_ms,train_pan,train_fusion])
#         #         ms=cv2.resize(ms,dsize=(0,0),fx=4,fy=4)
#         #         np.save('/root/tfnet_pytorch-master/dataset/train/%d_lr_u.npy'%(i),np.transpose(ms,(2,0,1)))
#         #         np.save('/root/tfnet_pytorch-master/dataset/train/%d_pan.npy'%(i),np.transpose(pan,(2,0,1)))
#         #         np.save('/root/tfnet_pytorch-master/dataset/train/%d_mul.npy'%(i),np.transpose(fusion,(2,0,1)))
#         #         i+=1
#         # except tf.errors.OutOfRangeError:
#         #     pass
#         try:
#             i=0
#             while True:
#                 ms,pan,fusion=sess.run([val_ms,val_pan,val_fusion])
#                 ms=cv2.resize(ms,dsize=(0,0),fx=4,fy=4)
#                 np.save('/root/tfnet_pytorch-master/dataset/test/%d_lr_u.npy'%(i),np.transpose(ms,(2,0,1)))
#                 np.save('/root/tfnet_pytorch-master/dataset/test/%d_pan.npy'%(i),np.transpose(pan,(2,0,1)))
#                 np.save('/root/tfnet_pytorch-master/dataset/test/%d_mul.npy'%(i),np.transpose(fusion,(2,0,1)))
#                 i+=1
#         except tf.errors.OutOfRangeError:
#             pass
