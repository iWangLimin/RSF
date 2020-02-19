import tensorflow as tf
def variable_weight(name,initializer,shape,regularize=True):
    weight=tf.get_variable(name,shape,tf.float32,initializer)
    if regularize:
        tf.add_to_collection('weight_decay',tf.reduce_mean(tf.square(weight))*0.5)
    return weight

def conv(name,x,filter_num=64,kernel_size=3,dilation=1,strides=1,padding='SAME',\
         activation='lrelu',is_training=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        feature_shape=x.get_shape().as_list()
        filters=variable_weight(name,tf.initializers.he_normal(),\
            shape=[kernel_size,kernel_size,feature_shape[-1],filter_num])
        bias=variable_weight(name+"_bias",tf.zeros_initializer(),[1,1,filter_num],regularize=False)
        y=tf.add(tf.nn.conv2d(x,filters,strides,padding,dilations=dilation),bias)
        if activation==None:
            return y
        elif activation=='relu':
            return tf.nn.relu(y)
        elif activation=='lrelu':
            return tf.nn.leaky_relu(y)
def bottle_neck(name,x,out_channel=64,activation='lrelu',is_training=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        return conv('conv',x,filter_num=out_channel,kernel_size=1,activation=activation,is_training=is_training)

def dense_block(name,x,growth_rate=16,kernel_size=3,strides=1,padding='SAME',\
        activation='lrelu',conv_num=8,input_include=False,is_training=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        if input_include:
            all_output=[x] 
        else:
            all_output=[]
        all_output.append(conv('conv1',x,growth_rate,kernel_size=kernel_size,strides=strides,
                               padding=padding,activation=activation,is_training=is_training))
        for i in range(2,conv_num+1):
            all_output.append(conv('conv%d'%(i),tf.concat(all_output,-1),growth_rate,kernel_size=kernel_size,
                                   strides=strides,padding=padding,activation=activation,is_training=is_training))
        return tf.concat(all_output,-1)

def conv_transpose(name,x,scale,filter_num=64,kernel_size=3,dilation=1,strides=2,\
        padding='SAME',activation='lrelu',is_training=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        _,h,w,in_channel=x.get_shape().as_list()
        out_shape=tf.stack([tf.shape(x)[0],h*scale,w*scale,filter_num])
        filters=variable_weight(name,tf.initializers.he_normal(),\
            shape=[kernel_size,kernel_size,filter_num,in_channel])
        bias=variable_weight(name+"_bias",tf.zeros_initializer(),[1,1,filter_num],regularize=False)
        y=tf.add(tf.nn.conv2d_transpose(x,filters,out_shape,strides,padding,dilations=dilation),bias)
        if activation==None:
            return y
        elif activation=='relu':
            return tf.nn.relu(y)
        elif activation=='lrelu':
            return tf.nn.leaky_relu(y)


