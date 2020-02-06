import tensorflow as tf
from module import conv,dense_block,bottle_neck,conv_transpose,variable_weight
class Model():
    def residual_extraction(x,block_num,growth_rate,conv_num,channel_reduce=True):
        features=[]
        with tf.variable_scope('advance_extraction',reuse=tf.AUTO_REUSE):
            y=conv('conv1',x)
            y=conv('conv2',y)
            features.append(y)
        for i in range(1,block_num+1):
            y=dense_block('dense%d'%(i),features[-1],conv_num=conv_num)
            if channel_reduce:
                y=bottle_neck('bottle_neck%d'%(i),y)
            features.append(y)
        residual=0
        for feature in features:
            residual+=feature
        return residual
    def ms_reconstruction(ms):
        with tf.variable_scope('ms_reconstruction',reuse=tf.AUTO_REUSE):
            up_ms=tf.image.resize_bilinear(ms,size=(256,256))
            residual=self.residual_extraction(ms,2,16,4,False)
            with tf.variable_scope('upsample'):
                residual=conv_transpose('deconv1',residual,128)
                residual=conv_transpose('deconv2',256,4,activation=None)
            return up_ms+residual
    def pan_reconstruction(pan,ms_out):
        with tf.variable_scope('pan_reconstruction',reuse=tf.AUTO_REUSE):
            residual=self.residual_extraction(pan,5,16,8,True)
            residual=bottle_neck('bottle_neck',4,activation=None)
            # residual=tf.expand_dims(residual,-1)
            # adjust=variable_weight('adjust',tf.initializers.ones,[4,],regularize=False)
            return residual+ms_out
    def build_graph():
        


        

