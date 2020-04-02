import tensorflow as tf
from module import conv
from config import TrainingConfig as TC
class PNN():
    def __init__(self,ms_size=32):
        self.ms_size=ms_size
    def forward(self,ms,pan):
        with tf.variable_scope('PNN',reuse=tf.AUTO_REUSE):
            up_ms=tf.image.resize_bicubic(ms,size=(self.ms_size*4,self.ms_size*4))
            x=tf.concat([up_ms,pan],-1)
            with tf.variable_scope('opt1',reuse=tf.AUTO_REUSE):
                feature=conv('conv1',x,64,9,activation='lrelu')
                feature=conv('conv2',feature,32,5,activation='lrelu')
            with tf.variable_scope('opt2',reuse=tf.AUTO_REUSE):
                fusion=conv('conv3',feature,4,5,activation=None)\
                    +up_ms
            return fusion
    def loss(self,fusion,gt):
        return tf.losses.mean_squared_error(gt,fusion)+\
            tf.add_n(tf.get_collection('weight_decay'))*TC.weight_dacay
    def acc(self,prediction,gt,op_name):
        return tf.metrics.mean_squared_error(\
            labels=gt,predictions=prediction,name=op_name)
class RSIFNN(PNN):
    def __init__(self,ms_size=32):
        self.ms_size=ms_size
    def forward(self,ms,pan):
        with tf.variable_scope('RSIFNN',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('ms',reuse=tf.AUTO_REUSE):
                up_ms=tf.image.resize_bicubic(ms,(self.ms_size*4,self.ms_size*4))
                feature_ms=conv('conv1',up_ms,activation='lrelu')
                feature_ms=conv('conv2',feature_ms,filter_num=32,activation='lrelu')
            with tf.variable_scope('pan',reuse=tf.AUTO_REUSE):
                feature_pan=pan
                for i in range(1,8):
                    feature_pan=conv('conv%d'%(i),feature_pan,activation='lrelu')
                feature_pan=conv('conv%d'%(i+1),feature_pan,filter_num=32,activation='lrelu')
            with tf.variable_scope('fusion',reuse=tf.AUTO_REUSE):
                features=tf.concat([feature_ms,feature_pan],-1)
                return up_ms+conv('conv',features,4,activation=None)
