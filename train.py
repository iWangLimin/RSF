import tensorflow as tf
from dataset import Dataset
from model import ResidualRecursiveFusion
from config import TrainingConfig as TC
import argparse
import os
def summary(sess, graph_save=False):
    summary_writer = tf.summary.FileWriter('tensorboard\\AlterRecursiveFusion', session=sess)
    if graph_save:
        summary_writer.add_graph(sess.graph)
    return summary_writer
def parse_arg():
    parser = argparse.ArgumentParser(description='Test for argparse')
    # parser.add_argument('-save_graph',default=False)
    parser.add_argument('-restore',default=False)
    args = parser.parse_args()
    return args
def init_vars(restore,sess,saver):
    if restore:
        states=tf.train.get_checkpoint_state('checkpoint_ARF/')
        checkpoint_paths=states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)
        saver.restore(sess,saver.last_checkpoints[-1])
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
if __name__ == '__main__':
    with tf.device('/cpu:0'):
        args=parse_arg()
        restore=args.restore
        train_set = Dataset(train_set=True,dir_path='dataset/train',ms_upsample=False)
        # val_set = Dataset(train_set=False,dir_path='dataset/val',ms_upsample=False)
        train_iter=train_set.get_iter()
        # val_iter=val_set.get_iter()
        train_ms,train_pan,train_fusion=train_iter.get_next()
        # val_ms,val_pan,val_fusion=val_iter.get_next()
        model = ResidualRecursiveFusion(ms_max_recurse=4,ms_conv_depth=4,ms_grow=16,\
                                        pan_max_recurse=16,pan_conv_depth=4,pan_grow=16,\
                                        reserved_recon='direct')
        ms, pan, fusion = tf.placeholder(tf.float32, [None, 64, 64, 4]), \
                            tf.placeholder(tf.float32, [None, 256, 256, 1]), \
                            tf.placeholder(tf.float32, [None, 256, 256, 4])
        # is_training = tf.placeholder(tf.bool)
        ms_fusion,pan_fusion= model.forward(ms, pan)
        ms_loss,pan_loss= model.loss(ms_fusion,pan_fusion,fusion)
        # acc,acc_update= model.acc(pan_fusion, fusion,'mse_acc')
        # acc_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mse_acc")
        # acc_vars_initializer = tf.variables_initializer(var_list=acc_vars)
        train_summary=tf.summary.merge([tf.summary.scalar(name='ms_loss',tensor=ms_loss),\
            tf.summary.scalar(name='pan_loss',tensor=pan_loss)])
        # val_summary=tf.summary.scalar(name='mse_acc',tensor=acc)
        step = tf.Variable(0,dtype=tf.int32,trainable=False)
        # best_acc=tf.Variable(10000,trainable=False,dtype=tf.float32)
        opt = tf.train.AdamOptimizer(learning_rate=TC.learning_rate)
        # pan_opt = tf.train.AdamOptimizer(learning_rate=TC.learning_rate)
        # var_list_1=tf.get_collection(\
        #     tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms')
        # var_list_2=tf.get_collection(\
        #     tf.GraphKeys.TRAINABLE_VARIABLES,scope='pan')
        ms_op = opt.minimize(ms_loss, global_step=step,var_list=tf.get_collection(\
            tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms'))
        pan_op=opt.minimize(pan_loss,global_step=step,var_list=tf.get_collection(\
            tf.GraphKeys.TRAINABLE_VARIABLES,scope='pan'))
        saver=tf.train.Saver()
        # variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='ms_branch/trunk')
    with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:
        sess.run(train_iter.initializer)
        init_vars(restore,sess,saver)
        summary_writer=tf.summary.FileWriter('tensorboard_ARF',\
            session=sess,graph=sess.graph)
        while True:
            train_ms_value,train_pan_value,train_fusion_value=\
                sess.run([train_ms,train_pan,train_fusion])
            train_loss,step_value,_,_=sess.run([train_summary,step,ms_op,pan_op],feed_dict=\
                {ms:train_ms_value,pan:train_pan_value,fusion:train_fusion_value})
            summary_writer.add_summary(train_loss,step_value)
            # if step_value%TC.val_step==0:
            #     sess.run([val_iter.initializer,acc_vars_initializer])
            #     try:
            #         while True:
            #             val_ms_value,val_pan_value,val_fusion_value=\
            #                 sess.run([val_ms,val_pan,val_fusion])
            #             sess.run(acc_update,feed_dict=\
            #                 {ms:val_ms_value,pan:val_pan_value,fusion:val_fusion_value})
            #     except tf.errors.OutOfRangeError:
            #         best_acc_value,acc_value,val_acc=sess.run([best_acc,acc,val_summary])
            #         summary_writer.add_summary(val_acc, step_value)
            #         if acc_value<best_acc_value:
            #             best_acc.assign(acc_value)
            #             saver.save(sess=sess,\
            #                 save_path=os.path.join(os.getcwd(), './checkpoint_ARF/ARF'),global_step=step)

