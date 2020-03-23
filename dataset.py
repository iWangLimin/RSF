from config import DatasetConfig as DC,TrainingConfig as TC
import tensorflow as tf
import os
class Dataset():
    def __init__(self,train_set,dir_path):
        record_files=os.listdir(dir_path)
        self.dataset=tf.data.TFRecordDataset(os.path.join(dir_path,record_files[0]))
        for tfrecord in record_files[1:]:
            self.dataset=self.dataset.concatenate(tf.data.TFRecordDataset(
                os.path.join(dir_path,tfrecord)))
        self.train_set=train_set
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
        # if self.ms_upsample:
        #     ms=tf.reshape(ms,[256,256,4])
        # else:
        ms=tf.reshape(ms,[DC.MS_crop_size//4,DC.MS_crop_size//4,4])
        pan=tf.reshape(pan,[DC.MS_crop_size,DC.MS_crop_size,1])
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
        self.dataset=self.dataset.map(self.parse_example)
        self.dataset=self.dataset.map(self.normalize)
        if self.train_set:
            self.dataset=self.dataset.shuffle(20000).\
                repeat().batch(batch_size=TC.batch_size)
        else:
            self.dataset=self.dataset.batch(batch_size=TC.batch_size)
    def get_iter(self):
        self.batch()
        return tf.data.make_initializable_iterator(self.dataset)