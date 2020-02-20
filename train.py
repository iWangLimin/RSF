import tensorflow as tf
from dataset import Dataset
from model import Model
from config import TrainingConfig as TC
import argparse
import os
def summary(sess, graph_save=False):
    summary_writer = tf.summary.FileWriter('tensorboard\\residual_fusion', session=sess)
    if graph_save:
        summary_writer.add_graph(sess.graph)
    return summary_writer
def parse_arg():
    parser = argparse.ArgumentParser(description='Test for argparse')
    # parser.add_argument('-save_graph',default=False)
    parser.add_argument('-restore',default=True)
    args = parser.parse_args()
    return args
def init_vars(restore,sess,saver):
    if restore:
        states=tf.train.get_checkpoint_state('checkpoint/')
        checkpoint_paths=states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)
        saver.restore(sess,saver.last_checkpoints[-1])
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
if __name__ == '__main__':
    args=parse_arg()
    restore=args.restore
    train_set = Dataset(train_set=True,dir_path='dataset/train')
    val_set = Dataset(train_set=False,dir_path='dataset/val')
    train_iter,val_iter=train_set.get_iter(),val_set.get_iter()
    train_ms,train_pan,train_fusion=train_iter.get_next()
    val_ms,val_pan,val_fusion=val_iter.get_next()
    model = Model()
    ms, pan, fusion = tf.placeholder(tf.float32, [None, 64, 64, 4]), \
                          tf.placeholder(tf.float32, [None, 256, 256, 1]), \
                          tf.placeholder(tf.float32, [None, 256, 256, 4])
    # is_training = tf.placeholder(tf.bool)
    predict = model.forward(ms, pan)
    loss = model.loss(predict,fusion)
    acc,acc_update= model.acc(predict, fusion,'mse_acc')
    acc_vars=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="mse_acc")
    acc_vars_initializer = tf.variables_initializer(var_list=acc_vars_initializer)
    loss_summary=tf.summary.scalar(name='mse_loss',tensor=loss)
    acc_summary=tf.summary.scalar(name='mse_acc',tensor=acc)
    step = tf.Variable(0,dtype=tf.uint16,trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=TC.learning_rate)
    train_op = opt.minimize(loss, global_step=step)
    saver=tf.train.Saver()
    best_acc=float('inf')
    with tf.Session() as sess:
        sess.run(train_iter.initializer)
        init_vars(restore,sess,saver)
        summary_writer=tf.summary.FileWriter('tensorboard\\residual_fusion',\
            session=sess,graph=sess.graph)
        while True:
            train_ms_value,train_pan_value,train_fusion_value=\
                sess.run([train_ms,train_pan,train_fusion])
            train_loss,step_value,_=sess.run([loss_summary,step,train_op],feed_dict=\
                {ms:train_ms_value,pan:train_pan_value,fusion:train_fusion_value})
            summary_writer.add_summary(train_loss,step_value)
            if step_value%TC.val_step==0:
                sess.run([val_iter.initializer,acc_vars_initializer])
                try:
                    while True:
                        val_ms_value,val_pan_value,val_fusion_value=\
                            sess.run([val_ms,val_pan,val_fusion])
                        sess.run(acc_update,feed_dict=\
                            {ms:val_ms_value,pan:val_pan_value,fusion:val_fusion_value})
                except tf.errors.OutOfRangeError:
                    acc_value,val_acc=sess.run([acc,acc_summary])
                    summary_writer.add_summary(val_acc, step_value)
                    if acc_value<best_acc:
                        best_acc=acc_value
                        saver.save(sess=sess,\
                            save_path=os.path.join(os.getcwd(), './checkpoint/residual_fusion'),global_step=step)

