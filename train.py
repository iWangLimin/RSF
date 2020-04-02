import tensorflow as tf
from dataset import Dataset
from model import LapFusion,DenseLapFusion,DenseMsScaleCom,DenseNoCom,DenseNoComV2,LapFusionShallow
from config import TrainingConfig as TC,DatasetConfig as DC
import os
def init_vars(restore,sess,saver,name):
    if restore:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        states=tf.train.get_checkpoint_state('checkpoint%s/'%(name))
        checkpoint_paths=states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)
        saver.restore(sess,saver.last_checkpoints[-1])
        sess.run(tf.local_variables_initializer())
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
if __name__ == '__main__':
    # args=parse_arg()
    # restore=args.restore
    restore=True
    name='DenseLapFusionNewData'
    model = DenseNoComV2()
    train_set = Dataset(train_set=True,dir_path='datasetV2/train')
    val_set = Dataset(train_set=False,dir_path='datasetV2/val')
    train_iter=train_set.get_iter()
    val_iter=val_set.get_iter()
    train_ms,train_pan,train_fusion_1,train_fusion_2=train_iter.get_next()
    val_ms,val_pan,val_fusion=val_iter.get_next()
    ms,pan,gt_1,gt_2=tf.placeholder(tf.float32, [None,32,32, 4]), \
                        tf.placeholder(tf.float32, [None, 128, 128, 1]), \
                        tf.placeholder(tf.float32, [None, 64, 64, 4]),\
                        tf.placeholder(tf.float32, [None, 128, 128, 4])
    # is_training = tf.placeholder(tf.bool)
    fusion_1,fusion_2= model.forward(ms, pan)
    loss_1,loss_2= model.loss(fusion_1,fusion_2,gt_1,gt_2)
    loss=0.2*loss_1+0.8*loss_2+tf.add_n(tf.get_collection('weight_decay'))*TC.weight_dacay
    acc,acc_update= model.acc(fusion_2,gt_2,'mse_acc')
    best_acc=tf.Variable(10000,trainable=False,dtype=tf.float32)
    acc_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mse_acc")
    acc_vars_initializer = tf.variables_initializer(var_list=acc_vars)
    train_summary=tf.summary.merge([tf.summary.scalar(name='loss1',tensor=loss_1),\
        tf.summary.scalar(name='loss2',tensor=loss_2),
        tf.summary.scalar(name='loss',tensor=loss)])
    val_summary=tf.summary.scalar(name='mse_acc',tensor=acc)
    step = tf.Variable(0,dtype=tf.int32,trainable=False)
    learning_rate=tf.train.exponential_decay(TC.learning_rate*0.8,step,1000,0.96)
    opt= tf.train.AdamOptimizer(learning_rate)
    # lr_summary=tf.summary.scalar(name='lr',tensor=learning_rate)
    # pan_opt = tf.train.AdamOptimizer(learning_rate=TC.learning_rate)
    # var_list_1=tf.get_collection(\
    #     tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms')
    # var_list_2=tf.get_collection(\
    #     tf.GraphKeys.TRAINABLE_VARIABLES,scope='pan')

    step_op = opt.minimize(loss,global_step=step)
    # var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms'))
    var = tf.global_variables()
    # saver_restore=tf.train.Saver(var_list=[val for val in var if 'combanation' not in val.name])
    saver = tf.train.Saver(max_to_keep=100)
    # variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms_branch/trunk')
    with tf.Session() as sess:
        sess.run(train_iter.initializer)
        init_vars(restore,sess,saver,name)
        summary_writer=tf.summary.FileWriter('tensorboard_%s'%(name),\
            session=sess,graph=sess.graph)
        while True:
            train_ms_value,train_pan_value,train_fusion_1_value,train_fusion_2_value=\
                sess.run([train_ms,train_pan,train_fusion_1,train_fusion_2])
            train_loss,step_value,_=sess.run([train_summary,step,step_op],feed_dict=\
                {ms:train_ms_value,pan:train_pan_value,\
                gt_1:train_fusion_1_value,gt_2:train_fusion_2_value})
            summary_writer.add_summary(train_loss,step_value)
            step_value=sess.run(step)
            if step_value%TC.val_step==0:
                sess.run([val_iter.initializer,acc_vars_initializer])
                try:
                    while True:
                        val_ms_value,val_pan_value,val_fusion_value=\
                            sess.run([val_ms,val_pan,val_fusion])
                        sess.run(acc_update,feed_dict=\
                            {ms:val_ms_value,pan:val_pan_value,gt_2:val_fusion_value})
                except tf.errors.OutOfRangeError:
                    best_acc_value,acc_value,val_acc=sess.run([best_acc,acc,val_summary])
                    summary_writer.add_summary(val_acc, step_value)
                    if acc_value<best_acc_value:
                        # best_acc.assign(acc_value)
                        sess.run(tf.assign(best_acc,acc_value))
                    saver.save(sess=sess,\
                        save_path=os.path.join(os.getcwd(), './checkpoint%s/model'%(name)),global_step=step)

