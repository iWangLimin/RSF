import tensorflow as tf
from module import conv,dense_block,bottle_neck,conv_transpose,variable_weight
from config import TrainingConfig as TC
from config import LapFusionConfig,DenseLapFusionConfig
class LapFusion():
    def __init__(self,ms_size=32):
        self.ms_size=ms_size
    def upsample(self,feature,out_channel=64,scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv_transpose('deconv',feature,scale,out_channel)
    def downsample(self,feature,out_channel,scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv('down_conv',feature,out_channel,strides=scale)
    def ms_feature_extraction(self,name,ms):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            # feature=conv('conv1',ms,LapFusionConfig.ms_feature_channel)
            # feature=conv('conv2',feature,LapFusionConfig.ms_feature_channel)
            # res=feature
            # for i in range(3,8):
            #     res=conv('conv%d'%(i),res,LapFusionConfig.ms_feature_channel)
            # res=conv('conv%d'%(i+1),res,LapFusionConfig.ms_feature_channel,activation=None)
            # feature=tf.nn.leaky_relu(feature+res)
            feature=ms
            for i in range (1,LapFusionConfig.ms_depth+1):
                feature=conv('conv%d'%(i),feature,LapFusionConfig.ms_feature_channel)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                ms_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                ms_feature_2=self.upsample(ms_feature_1,out_channel=64)
            return ms_feature_1,ms_feature_2
    def pan_feature_extraction(self,name,pan):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            l1,l2,l3=LapFusionConfig.pan_depth
            l1,l2=l1-1,l2-1
            feature_channel=LapFusionConfig.pan_featurec_channel
            with tf.variable_scope('block1',reuse=tf.AUTO_REUSE):
                feature=pan
                for i in range(1,l1+1):
                    feature=conv('conv%d'%(i),feature,feature_channel[0])
            with tf.variable_scope('down_sample1',reuse=tf.AUTO_REUSE):
                feature=self.downsample(feature,out_channel=feature_channel[0])
            with tf.variable_scope('block2',reuse=tf.AUTO_REUSE):
                for i in range(1,l2+1):
                    feature=conv('conv%d'%(i),feature,feature_channel[1])
                # res=feature
                # for i in range(1,l2):
                #     res=conv('conv%d'%(i),res,feature_channel[1])
                # res=conv('conv%d'%(i+1),res,64,activation=None)
                # feature=tf.nn.leaky_relu(res+feature)
            with tf.variable_scope('down_sample2',reuse=tf.AUTO_REUSE):
                feature=self.downsample(feature,out_channel=feature_channel[1])
            with tf.variable_scope('block3',reuse=tf.AUTO_REUSE):
                for i in range(1,l3+1):
                    feature=conv('conv%d'%(i),feature,feature_channel[1])
                # res=feature
                # for i in range(1,l3):
                #     res=conv('conv%d'%(i),res,feature_channel[2])
                # res=conv('conv%d'%(i+1),res,feature_channel[2],activation=None)
                # feature=tf.nn.leaky_relu(res+feature)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                pan_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                pan_feature_2=self.upsample(pan_feature_1,out_channel=64)
            return pan_feature_1,pan_feature_2
    def feature_fusion(self,name,ms_feature,pan_feature):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            fusion_feature=tf.concat([ms_feature,pan_feature],axis=-1)
            channel_in=fusion_feature.get_shape().as_list()[-1]
            fusion_feature=conv('conv1',fusion_feature,filter_num=channel_in)
            detail=conv('conv2',fusion_feature,4,activation=None)
            return detail
    def forward(self,ms,pan):
        with tf.variable_scope('LapFusion',reuse=tf.AUTO_REUSE):
            ms_feature_1,ms_feature_2=self.ms_feature_extraction('ms_feature_extraction',ms)
            pan_feature_1,pan_feature_2=self.pan_feature_extraction('pan_feature_extraction',pan)
            detail_1=self.feature_fusion('fusion1',ms_feature_1,pan_feature_1)
            detail_2=self.feature_fusion('fusion2',ms_feature_2,pan_feature_2)
            fusion_1=tf.image.resize_bicubic(ms,size=(self.ms_size*2,self.ms_size*2))+detail_1
            up_fusion_1=tf.image.resize_bicubic(fusion_1,size=(self.ms_size*4,self.ms_size*4))
            bp_stop=tf.stop_gradient(up_fusion_1)
            fusion_2=bp_stop+detail_2
            return fusion_1,fusion_2
    def loss(self,fusion_1,fusion_2,gt_1,gt_2):
        return self.l1_loss(gt_1,fusion_1),self.l1_loss(gt_2,fusion_2)
    def l1_loss(self,gt,prediction):
        return tf.reduce_mean(tf.sqrt((tf.square(prediction-gt)+0.001**2)))
    def acc(self,prediction,gt,op_name):
        # with tf.variable_scope('acc'):
        return tf.metrics.mean_squared_error(\
            labels=gt,predictions=prediction,name=op_name)
class DenseLapFusion():
    def __init__(self,ms_size=32):
        self.ms_size=ms_size
    def upsample(self,feature,out_channel=64,scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv_transpose('deconv',feature,scale,out_channel)
    def downsample(self,feature,out_channel,scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv('down_conv',feature,out_channel,strides=scale)
    def ms_feature_extraction(self,name,ms):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction',reuse=tf.AUTO_REUSE):
                feature=ms
                feature=conv('conv1',feature,64)
                feature=conv('conv2',feature,64)
            feature=dense_block('dense_block1',feature,12,conv_num=6,input_include=True)
            feature=bottle_neck('bottle_neck1',feature,64)
            feature=dense_block('dense_block2',feature,12,conv_num=6,input_include=True)
            feature=bottle_neck('bottle_neck2',feature,64)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                ms_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                ms_feature_2=self.upsample(ms_feature_1,out_channel=64)
            return ms_feature_1,ms_feature_2
    def pan_feature_extraction(self,name,pan):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction'):
                feature=pan
                feature=conv('conv1',feature,64)
                feature1=conv('conv2',feature,64)
            with tf.variable_scope('down_sample1'):
                feature2=self.downsample(feature1,64)
            feature2=dense_block('dense_block1',feature2,12,conv_num=6,input_include=True)
            feature2=bottle_neck('bottle_neck1',feature2,64)
            with tf.variable_scope('down_sample2'):
                feature3=self.downsample(feature2,64)
            feature3=dense_block('dense_block2',feature3,12,conv_num=6,input_include=True)
            feature3=bottle_neck('bottle_neck2',feature3,64)
            feature3=dense_block('dense_block3',feature3,12,conv_num=6,input_include=True)
            feature3=bottle_neck('bottle_neck3',feature3,64)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                feature3=self.upsample(feature3,out_channel=64)
                pan_feature_1=conv('conv',tf.concat([feature2,feature3],-1),filter_num=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                up_pan_feature_1=self.upsample(pan_feature_1,out_channel=64)
                pan_feature_2=conv('conv',tf.concat([up_pan_feature_1,feature1],-1),filter_num=64)
            return pan_feature_1,pan_feature_2
    def feature_fusion(self,name,ms_feature,pan_feature):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            fusion_feature=tf.concat([ms_feature,pan_feature],axis=-1)
            channel_in=fusion_feature.get_shape().as_list()[-1]
            fusion_feature=conv('conv1',fusion_feature,filter_num=channel_in)
            detail=conv('conv2',fusion_feature,4,activation=None)
            return detail
    def forward(self,ms,pan):
        with tf.variable_scope('LapFusion',reuse=tf.AUTO_REUSE):
            ms_feature_1,ms_feature_2=self.ms_feature_extraction('ms_feature_extraction',ms)
            pan_feature_1,pan_feature_2=self.pan_feature_extraction('pan_feature_extraction',pan)
            detail_1=self.feature_fusion('fusion1',ms_feature_1,pan_feature_1)
            detail_2=self.feature_fusion('fusion2',ms_feature_2,pan_feature_2)
            fusion_1=tf.image.resize_bicubic(ms,size=(self.ms_size*2,self.ms_size*2))+detail_1
            up_fusion_1=tf.image.resize_bicubic(fusion_1,size=(self.ms_size*4,self.ms_size*4))
            bp_stop=tf.stop_gradient(up_fusion_1)
            fusion_2=bp_stop+detail_2
            return fusion_1,fusion_2
    def loss(self,fusion_1,fusion_2,gt_1,gt_2):
        return self.l1_loss(gt_1,fusion_1),self.l1_loss(gt_2,fusion_2)
    def l1_loss(self,gt,prediction):
        return tf.reduce_mean(tf.sqrt((tf.square(prediction-gt)+0.001**2)))
    def acc(self,prediction,gt,op_name):
        return tf.metrics.mean_squared_error(\
            labels=gt,predictions=prediction,name=op_name)
class DenseMsScaleCom(DenseLapFusion):
    def ms_feature_extraction(self,name,ms):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction',reuse=tf.AUTO_REUSE):
                feature1=ms
                feature1=conv('conv1',feature1,64)
                feature1=conv('conv2',feature1,64)
            feature2=dense_block('dense_block1',feature1,12,conv_num=6,input_include=True)
            feature2=bottle_neck('bottle_neck1',feature2,64)
            feature3=dense_block('dense_block2',feature2,12,conv_num=6,input_include=True)
            feature3=bottle_neck('bottle_neck2',feature3,64)
            with tf.variable_scope('combanation'):
                feature=tf.concat([feature1,feature2,feature3],-1)
                feature=bottle_neck('bottle_neck',feature,64)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                ms_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                ms_feature_2=self.upsample(ms_feature_1,out_channel=64)
            return ms_feature_1,ms_feature_2
class DenseNoCom(DenseLapFusion):
    def ms_feature_extraction(self,name,ms):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction',reuse=tf.AUTO_REUSE):
                feature=ms
                feature=conv('conv1',feature,64)
                feature=conv('conv2',feature,64)
            feature=dense_block('dense_block1',feature,12,conv_num=4,input_include=True)
            feature=bottle_neck('bottle_neck1',feature,64)
            feature=dense_block('dense_block2',feature,12,conv_num=6,input_include=True)
            feature=bottle_neck('bottle_neck2',feature,64)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                ms_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                ms_feature_2=self.upsample(ms_feature_1,out_channel=64)
            return ms_feature_1,ms_feature_2
    def pan_feature_extraction(self,name,pan):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction'):
                feature=pan
                feature=conv('conv1',feature,64)
                feature=conv('conv2',feature,64)
            with tf.variable_scope('down_sample1'):
                feature=self.downsample(feature,64)
            feature=dense_block('dense_block1',feature,12,conv_num=6,input_include=True)
            feature=bottle_neck('bottle_neck1',feature,64)
            with tf.variable_scope('down_sample2'):
                feature=self.downsample(feature,64)
            feature=dense_block('dense_block2',feature,12,conv_num=8,input_include=True)
            feature=bottle_neck('bottle_neck2',feature,64)
            with tf.variable_scope('upsample1',reuse=tf.AUTO_REUSE):
                pan_feature_1=self.upsample(feature,out_channel=64)
            with tf.variable_scope('upsample2',reuse=tf.AUTO_REUSE):
                pan_feature_2=self.upsample(pan_feature_1,out_channel=64)
            return pan_feature_1,pan_feature_2
class DenseNoComV2():
    def __init__(self, ms_size=32):
        self.ms_size = ms_size

    def upsample(self, feature, out_channel=64, scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv_transpose('deconv', feature, scale, out_channel)

    def downsample(self, feature, out_channel, scale=2):
        # with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv('down_conv', feature, out_channel, strides=scale)

    def ms_feature_extraction(self, name, ms):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction', reuse=tf.AUTO_REUSE):
                feature = ms
                feature = conv('conv1', feature, 32)
                feature = conv('conv2', feature, 32)
            feature = dense_block('dense_block1', feature, 12, conv_num=4, input_include=True)
            feature = bottle_neck('bottle_neck1', feature, 64)
            feature = dense_block('dense_block2', feature, 12, conv_num=6, input_include=True)
            feature = bottle_neck('bottle_neck2', feature, 96)
            with tf.variable_scope('upsample1', reuse=tf.AUTO_REUSE):
                ms_feature_1 = self.upsample(feature, out_channel=96)
            with tf.variable_scope('upsample2', reuse=tf.AUTO_REUSE):
                ms_feature_2 = self.upsample(ms_feature_1, out_channel=96)
            return ms_feature_1, ms_feature_2

    def pan_feature_extraction(self, name, pan):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('low_level_extraction'):
                feature = pan
                feature = conv('conv1', feature, 32)
                feature = conv('conv2', feature, 32)
            with tf.variable_scope('down_sample1'):
                feature = self.downsample(feature, 32)
            feature = dense_block('dense_block1', feature, 12, conv_num=6, input_include=True)
            feature = bottle_neck('bottle_neck1', feature, 64)
            with tf.variable_scope('down_sample2'):
                feature = self.downsample(feature, 64)
            feature = dense_block('dense_block2', feature, 12, conv_num=6, input_include=True)
            feature = bottle_neck('bottle_neck2', feature, 64)
            feature = dense_block('dense_block3', feature, 12, conv_num=8, input_include=True)
            feature = bottle_neck('bottle_neck3', feature, 96)
            with tf.variable_scope('upsample1', reuse=tf.AUTO_REUSE):
                pan_feature_1 = self.upsample(feature, out_channel=96)
                # pan_feature_1 = conv('conv', tf.concat([feature2, feature3], -1), filter_num=64)
            with tf.variable_scope('upsample2', reuse=tf.AUTO_REUSE):
                pan_feature_2 = self.upsample(pan_feature_1, out_channel=96)
                # pan_feature_2 = conv('conv', tf.concat([up_pan_feature_1, feature1], -1), filter_num=64)
            return pan_feature_1, pan_feature_2

    def feature_fusion(self, name, ms_feature, pan_feature):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            fusion_feature = tf.concat([ms_feature, pan_feature], axis=-1)
            channel_in = fusion_feature.get_shape().as_list()[-1]
            fusion_feature = conv('conv1', fusion_feature, filter_num=96)
            fusion_feature = conv('conv2', fusion_feature, filter_num=64)
            detail = conv('conv3', fusion_feature, 4, activation=None)
            return detail

    def forward(self, ms, pan):
        with tf.variable_scope('DenseNoComV2', reuse=tf.AUTO_REUSE):
            ms_feature_1, ms_feature_2 = self.ms_feature_extraction('ms_feature_extraction', ms)
            pan_feature_1, pan_feature_2 = self.pan_feature_extraction('pan_feature_extraction', pan)
            detail_1 = self.feature_fusion('fusion1', ms_feature_1, pan_feature_1)
            detail_2 = self.feature_fusion('fusion2', ms_feature_2, pan_feature_2)
            fusion_1 = tf.image.resize_bicubic(ms, size=(self.ms_size * 2, self.ms_size * 2)) + detail_1
            up_fusion_1 = tf.image.resize_bicubic(fusion_1, size=(self.ms_size * 4, self.ms_size * 4))
            bp_stop = tf.stop_gradient(up_fusion_1)
            fusion_2 = bp_stop + detail_2
            return fusion_1, fusion_2

    def loss(self, fusion_1, fusion_2, gt_1, gt_2):
        return self.l1_loss(gt_1, fusion_1), self.l1_loss(gt_2, fusion_2)

    def l1_loss(self, gt, prediction):
        return tf.reduce_mean(tf.sqrt((tf.square(prediction - gt) + 0.001 ** 2)))

    def acc(self, prediction, gt, op_name):
        return tf.metrics.mean_squared_error(\
            labels=gt, predictions=prediction, name=op_name)
class LapFusionShallow(LapFusion):
    def ms_feature_extraction(self,name,ms):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            feature=ms
            for i in range(6):
                feature=conv('conv%d'%(i+1),feature,64)
            with tf.variable_scope('upsample1', reuse=tf.AUTO_REUSE):
                ms_feature_1 = self.upsample(feature, out_channel=64)
            with tf.variable_scope('upsample2', reuse=tf.AUTO_REUSE):
                ms_feature_2 = self.upsample(ms_feature_1, out_channel=64)
            return ms_feature_1,ms_feature_2
    def pan_feature_extraction(self,name,pan):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            feature=pan
            for i in range(8):
                feature=conv('conv%d'%(i+1),feature,64,5)
            with tf.variable_scope('downsample',reuse=tf.AUTO_REUSE):
                feature1=self.downsample(feature,64)
            return feature1,feature