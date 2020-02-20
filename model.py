import tensorflow as tf
from module import conv,dense_block,bottle_neck,conv_transpose
from config import TrainingConfig as TC
class Model():
    def residual_extraction(self,x,block_num,growth_rate,conv_num,channel_reduce=True,is_training=True):
        features=[]
        with tf.variable_scope('advance_extraction',reuse=tf.AUTO_REUSE):
            y=conv('conv1',x,is_training=is_training)
            y=conv('conv2',y,is_training=is_training)
            features.append(y)
        for i in range(1,block_num+1):
            y=dense_block('dense%d'%(i),features[-1],conv_num=conv_num,growth_rate=growth_rate,
                          is_training=is_training)
            if channel_reduce:
                y=bottle_neck('bottle_neck%d'%(i),y,is_training=is_training)
            features.append(y)
        residual=0
        for feature in features:
            residual+=feature
        return residual
    def ms_reconstruction(self,ms,is_training=True):
        with tf.variable_scope('ms_reconstruction',reuse=tf.AUTO_REUSE):
            up_ms=tf.image.resize_bilinear(ms,size=(256,256))
            residual=self.residual_extraction(ms,2,16,4,False,is_training=is_training)
            with tf.variable_scope('upsample'):
                residual=conv_transpose('deconv1',residual,scale=2,is_training=is_training)
                residual=conv_transpose('deconv2',residual,scale=2,filter_num=4,\
                    activation=None,is_training=is_training)
            return up_ms+residual
    def pan_reconstruction(self,pan,is_training=True):
        with tf.variable_scope('pan_reconstruction',reuse=tf.AUTO_REUSE):
            residual=self.residual_extraction(pan,5,12,8,True,is_training=is_training)
            residual=bottle_neck('bottle_neck',residual,4,activation=None,is_training=is_training)
            # residual=tf.expand_dims(residual,-1)
            # adjust=variable_weight('adjust',tf.initializers.ones,[4,],regularize=False)
            return residual
    def forward(self,ms,pan):
        ms_out=self.ms_reconstruction(ms)
        pan_out=self.pan_reconstruction(pan)
        return ms_out+pan_out
    def loss(self,predict,gt):
        return tf.losses.mean_squared_error(gt,predict)+\
               tf.reduce_mean(tf.get_collection('weight_decay'))*TC.weight_dacay
    def acc(self,predict,gt,op_name):
        return tf.metrics.mean_squared_error(gt,predict,name=op_name)




        


        

