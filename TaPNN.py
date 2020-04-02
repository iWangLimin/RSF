from module import conv
import tensorflow as tf
import os
from dataset import Dataset
from config import TrainingConfig as TC
from camparsion import PNN,RSIFNN
def init_vars(restore,sess,saver,name):
    if restore:
        states=tf.train.get_checkpoint_state('checkpoint%s/'%(name))
        checkpoint_paths=states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)
        saver.restore(sess,saver.last_checkpoints[-1])
        sess.run(tf.local_variables_initializer())
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
if __name__ == "__main__":
    g1=tf.Graph()
    with g1.as_default():
        name='RSIFNN'
        model = RSIFNN()
        restore=True
        train_set = Dataset(train_set=True,dir_path='datasetV2/train',\
            ms_upsample=True)
        val_set = Dataset(train_set=False,dir_path='datasetV2/val',\
            ms_upsample=True)
        train_iter=train_set.get_iter()
        val_iter=val_set.get_iter()
        train_ms,train_pan,_,train_fusion=train_iter.get_next()
        val_ms,val_pan,val_fusion=val_iter.get_next()
        ms,pan,gt=tf.placeholder(tf.float32, [None,32,32, 4]), \
                            tf.placeholder(tf.float32, [None, 128, 128, 1]),\
                            tf.placeholder(tf.float32, [None, 128, 128, 4])
        min_acc=tf.placeholder(tf.float32)
        fusion= model.forward(ms, pan)
        loss= model.loss(fusion,gt)
        acc,acc_update= model.acc(fusion,gt,'mse_acc')
        acc_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mse_acc")
        acc_vars_initializer = tf.variables_initializer(var_list=acc_vars)
        train_summary=tf.summary.scalar(name='loss',tensor=loss)
        val_summary=tf.summary.scalar(name='mse_acc',tensor=acc)
        step = tf.Variable(0,dtype=tf.int32,trainable=False)
        best_acc=tf.Variable(10000,trainable=False,dtype=tf.float32)
        # learning_rate = tf.train.exponential_decay(0.0001, step, 800, 0.96)
        opt=tf.train.MomentumOptimizer(0.0001,0.9)
        step_op=opt.minimize(loss,global_step=step)
        saver=tf.train.Saver(max_to_keep=100)
        best_acc_assign=tf.assign(best_acc,min_acc)
    with tf.Session(graph=g1) as sess:
        summary_writer=tf.summary.FileWriter('tensorboard_%s'%(name),\
            session=sess,graph=sess.graph)
        sess.run(train_iter.initializer)
        init_vars(restore,sess,saver,name)
        i=1
        while i<=20:
            train_ms_value,train_pan_value,train_fusion_value=\
                sess.run([train_ms,train_pan,train_fusion])
            train_loss,step_value,_=sess.run(\
                [train_summary,step,step_op],feed_dict=\
                {ms:train_ms_value,pan:train_pan_value,\
                    gt:train_fusion_value})
            summary_writer.add_summary(train_loss,step_value)
            if step_value%TC.val_step==0:
                sess.run([val_iter.initializer,acc_vars_initializer])
                try:
                    while True:
                        val_ms_value,val_pan_value,val_fusion_value=\
                            sess.run([val_ms,val_pan,val_fusion])
                        sess.run(acc_update,feed_dict=\
                            {ms:val_ms_value,pan:val_pan_value,gt:val_fusion_value})
                except tf.errors.OutOfRangeError:
                    best_acc_value,acc_value,val_acc=sess.run([best_acc,acc,val_summary])
                    summary_writer.add_summary(val_acc, step_value)
                    if acc_value<best_acc_value:
                        sess.run(best_acc_assign,feed_dict={min_acc:acc_value})
                    saver.save(sess=sess,save_path=\
                            os.path.join(os.getcwd(), './checkpoint%s/model'%(name)),global_step=step)
                i+=1
    tf.reset_default_graph()
    g2=tf.Graph()
    with g2.as_default():
        name='PNN'
        model = PNN()
        restore=True
        train_set = Dataset(train_set=True,dir_path='datasetV2/train',\
            ms_upsample=True)
        val_set = Dataset(train_set=False,dir_path='datasetV2/val',\
            ms_upsample=True)
        train_iter=train_set.get_iter()
        val_iter=val_set.get_iter()
        train_ms,train_pan,_,train_fusion=train_iter.get_next()
        val_ms,val_pan,val_fusion=val_iter.get_next()
        ms,pan,gt=tf.placeholder(tf.float32, [None,32,32, 4]), \
                            tf.placeholder(tf.float32, [None, 128, 128, 1]),\
                            tf.placeholder(tf.float32, [None, 128, 128, 4])
        min_acc=tf.placeholder(tf.float32)
        # is_training = tf.placeholder(tf.bool)
        fusion= model.forward(ms, pan)
        loss= model.loss(fusion,gt)
        # loss=0.2*loss_1+0.8*loss_2+tf.add_n(tf.get_collection('weight_decay'))*TC.weight_dacay
        acc,acc_update= model.acc(fusion,gt,'mse_acc')
        acc_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mse_acc")
        acc_vars_initializer = tf.variables_initializer(var_list=acc_vars)
        train_summary=tf.summary.scalar(name='loss',tensor=loss)
        val_summary=tf.summary.scalar(name='mse_acc',tensor=acc)
        step = tf.Variable(0,dtype=tf.int32,trainable=False)
        best_acc=tf.Variable(10000,trainable=False,dtype=tf.float32)
        # lr1 =tf.train.exponential_decay(0.0001, step, 800, 0.96)
        # lr2=tf.train.exponential_decay(0.00001, step, 800, 0.96)
        opt_1=tf.train.MomentumOptimizer(0.0001, 0.9)
        opt_2=tf.train.MomentumOptimizer(0.00001, 0.9)
        var_list_1=tf.get_collection(\
            tf.GraphKeys.TRAINABLE_VARIABLES,scope='PNN/opt1')
        var_list_2=tf.get_collection(\
            tf.GraphKeys.TRAINABLE_VARIABLES,scope='PNN/opt2')
        step_op_1 = opt_1.minimize(loss,global_step=step,var_list=var_list_1)
        step_op_2 = opt_2.minimize(loss,global_step=step,var_list=var_list_2)
        saver=tf.train.Saver(max_to_keep=100)
        best_acc_assign=tf.assign(best_acc,min_acc)     
    with tf.Session(graph=g2) as sess:
        summary_writer=tf.summary.FileWriter('tensorboard_%s'%(name),\
            session=sess,graph=sess.graph)
        sess.run(train_iter.initializer)
        init_vars(restore,sess,saver,name)
        i=1
        while i<=20:
            train_ms_value,train_pan_value,train_fusion_value=\
                sess.run([train_ms,train_pan,train_fusion])
            train_loss,step_value,_,_=sess.run(\
                [train_summary,step,step_op_1,step_op_2],feed_dict=\
                {ms:train_ms_value,pan:train_pan_value,\
                    gt:train_fusion_value})
            summary_writer.add_summary(train_loss,step_value)
            if step_value%TC.val_step==0:
                sess.run([val_iter.initializer,acc_vars_initializer])
                try:
                    while True:
                        val_ms_value,val_pan_value,val_fusion_value=\
                            sess.run([val_ms,val_pan,val_fusion])
                        sess.run(acc_update,feed_dict=\
                            {ms:val_ms_value,pan:val_pan_value,gt:val_fusion_value})
                except tf.errors.OutOfRangeError:
                    best_acc_value,acc_value,val_acc=sess.run([best_acc,acc,val_summary])
                    summary_writer.add_summary(val_acc, step_value) 
                    if acc_value<best_acc_value:
                        sess.run(best_acc_assign,feed_dict={min_acc:acc_value})
                    saver.save(sess=sess,save_path=\
                        os.path.join(os.getcwd(), './checkpoint%s/model'%(name)),global_step=step)
                i+=1

