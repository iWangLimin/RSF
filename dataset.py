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
        feature_description={'ms':tf.io.FixedLenFeature([],tf.string),\
                             'pan':tf.io.FixedLenFeature([],tf.string),\
                             'fusion':tf.io.FixedLenFeature([],tf.string)}
        parsed_example=tf.io.parse_single_example(example,feature_description)
        ms=tf.decode_raw(parsed_example['ms'],tf.uint16)
        pan = tf.decode_raw(parsed_example['pan'], tf.uint16)
        fusion = tf.decode_raw(parsed_example['fusion'], tf.uint16)
        return tf.reshape(ms,[64,64,4]),tf.reshape(pan,[256,256,1]),\
            tf.reshape(fusion,[256,256,4])
    def normalize(self,ms,pan,fusion):
        n_imgs,imgs=[],[ms,pan,fusion]
        for img in imgs:
            float_img=tf.cast(img,tf.float32)
            n_imgs.append((float_img-DC.Min_Pixel)/(DC.Max_Pixel-DC.Min_Pixel))
        return n_imgs
    def batch(self):
        self.dataset=self.dataset.map(self.parse_example)
        self.dataset=self.dataset.map(self.normalize)
        if self.train_set:
            self.dataset=self.dataset.shuffle(10000).\
                repeat().batch(batch_size=TC.batch_size)
        else:
            self.dataset=self.dataset.batch(batch_size=TC.batch_size)
    def get_iter(self):
        self.batch()
        return tf.data.make_initializable_iterator(self.dataset)