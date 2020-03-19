import tensorflow as tf
from module import conv,dense_block,bottle_neck,conv_transpose,variable_weight
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

class ResidualRecursiveFusion():
    def __init__(self,ms_max_recurse,pan_max_recurse,ms_conv_depth,pan_conv_depth,\
                        ms_grow,pan_grow,reserved_recon='concat'):
        self.ms_max_recurse=ms_max_recurse
        self.pan_max_recurse=pan_max_recurse
        self.ms_conv_depth=ms_conv_depth
        self.pan_conv_depth=pan_conv_depth
        self.ms_grow=ms_grow
        self.pan_grow=pan_grow
        self.reserved_recon=reserved_recon
    def embedding(self,name,input):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            feature=conv(name='conv1',x=input)
            feature=conv(name='conv2',x=feature)
            return feature
    # def pan_embedding(self,pan):
    #     with tf.variable_scope('PAN_Embedding',reuse=tf.AUTO_REUSE):
    #         pan_feature=conv(name='conv1',x=pan)
    #         pan_feature=conv(name='conv2',x=pan_feature,activation=None,filter_num=1)
    #         return pan_feature
    def recursion(self,name,input,module):
        if module=='ms':
            growth_rate,conv_num=self.ms_grow,self.ms_conv_depth
        else:
            growth_rate,conv_num=self.pan_grow,self.pan_conv_depth
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            features=dense_block(name='dense_block',x=input,\
                growth_rate=growth_rate,conv_num=conv_num)
            features=bottle_neck('bottle_neck',x=features,out_channel=64)
            return features
    def trunk(self,name,input,module):
        reserved_feature=[]
        if module=='ms':
            recursion=self.ms_max_recurse
            reduce_channel=16
        else:
            recursion=self.pan_max_recurse
            reduce_channel=8
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for _ in range (1,recursion+1):
                out_feature=self.recursion('recursion',input,module)
                input=out_feature
                # if step%(recursion//reserved)==0:
                if self.reserved_recon=='concat':
                    reserved_feature.append(bottle_neck('bottle_neck',out_feature,reduce_channel))
                else:
                    reserved_feature.append(out_feature)
                
        return reserved_feature
    def weight_average(self,name,inputs,reversed):
         with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            w=variable_weight('weight',tf.initializers.constant(value=1/reversed),\
                shape=(reversed),regularize=False)
            # w=variable_weight('weight',tf.ones_initializer(),\
            #     shape=(reversed),regularize=False)
            return tf.reduce_sum(inputs*w,axis=-1)
    def ms_recon(self,name,up_ms,reserved_feature):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            if self.reserved_recon=='concat':
                recon_features=conv_transpose('deconv1',tf.concat(reserved_feature,-1),\
                    filter_num=32,scale=2)
                recon_features=conv_transpose('deconv2',recon_features,scale=2,filter_num=4,\
                    activation=None)
                return recon_features+up_ms
            elif self.reserved_recon=='direct':
                recon_features=[]
                for feature in reserved_feature:
                    recon_feature=conv_transpose('deconv1',feature,scale=2)
                    recon_feature=conv_transpose('deconv2',recon_feature,scale=2,filter_num=4,\
                        activation=None)
                    recon_features.append(up_ms+recon_feature)
                return recon_features
                # self.weight_average(\
                #         'result_average',tf.stack(recon_features,axis=-1),len(reserved_feature))
            else:
                recon_features=[]
                for feature in reserved_feature:
                    recon_feature=conv_transpose('deconv1',feature,scale=2)
                    recon_feature=conv_transpose('deconv2',recon_feature,scale=2,filter_num=4,\
                        activation=None)
                    recon_features.append(recon_feature)
                return up_ms+self.weight_average(\
                    'feture_average',tf.stack(recon_features,axis=-1),len(reserved_feature))
        
    def pan_recon(self,name,ms_out,reserved_feature):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            if self.reserved_recon=='concat':
                recon_features=bottle_neck('bottle_neck_1',\
                    x=tf.concat(reserved_feature,-1))
                recon_features=bottle_neck('bottle_neck_2',x=recon_features,\
                    out_channel=4,activation=None)
                return ms_out+recon_features
            elif self.reserved_recon=='direct':
                recon_features=[]
                for feature in reserved_feature:
                    recon_feature=bottle_neck(\
                        'bottle_neck',x=feature,out_channel=4,activation=None)
                    recon_features.append(recon_feature+ms_out)
                return recon_features
                # self.weight_average(\
                #     'result_average',tf.stack(recon_features,axis=-1),len(recon_features))
            else:
                recon_features=[]
                for feature in reserved_feature:
                    recon_feature=bottle_neck(\
                        'bottle_neck',x=feature,out_channel=4,activation=None)
                    recon_features.append(recon_feature)
                return ms_out+self.weight_average(\
                    'feture_average',tf.stack(recon_features,axis=-1),len(reserved_feature))
    def branch(self,name,input,module):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            embedding_feature=self.embedding('embedding',input)
            reserved_feature=self.trunk('trunk',embedding_feature,module)
            return reserved_feature
    def forward(self,ms,pan):
        up_ms=tf.image.resize_bicubic(ms,[256,256])
        with tf.variable_scope('ms',reuse=tf.AUTO_REUSE):
            ms_reserved_features=self.branch('ms_branch',ms,'ms')
            ms_out=self.ms_recon('ms_recon',up_ms,ms_reserved_features)
        with tf.variable_scope('pan',reuse=tf.AUTO_REUSE):
            pan_reserved_features=self.branch('pan_branch',pan,'pan')
            pan_out=self.pan_recon('pan_recon',ms_out[-1],pan_reserved_features)
        return ms_out,pan_out
    def loss(self,ms_reconsts,pan_reconsts,gt):
        with tf.variable_scope('loss'):
            ms_losses=[]
            pan_losses=[]
            for reconst in ms_reconsts:
                ms_losses.append(0.5*tf.losses.mean_squared_error(\
                    labels=gt,predictions=reconst))
            ms_loss=tf.add_n(ms_losses)/len(ms_losses)
            ms_losses=None
            for reconst in pan_reconsts:
                pan_losses.append(self.l1_loss(gt,reconst))
            pan_loss=tf.add_n(pan_losses)/len(pan_losses)
            return ms_loss,pan_loss
            # ms_loss=tf.losses.mean_squared_error(predictions=ms_reconst,labels=gt)
            # pan_loss=tf.reduce_mean((tf.square((pan_reconst-gt))+0.001**2))
            # return ms_loss,pan_loss
    def l1_loss(self,gt,prediction):
        return tf.reduce_mean(tf.sqrt((tf.square(prediction-gt)+0.001**2)))
    def acc(self,prediction,gt,op_name):
        with tf.variable_scope('acc'):
            return tf.metrics.mean_squared_error(\
                labels=gt,predictions=prediction,name=op_name)
        

