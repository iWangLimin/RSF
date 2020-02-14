import tensorflow as tf
from dataset import Dataset
from model import Model
from config import TrainingConfig as TC
import argparse
def summary(sess, graph_save=False):
    summary_writer = tf.summary.FileWriter('tensorboard\\residual_fusion', session=sess)
    if graph_save:
        summary_writer.add_graph(sess.graph)
    return summary_writer
def parse_arg():
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('-save_graph',default=False)
    parser.add_argument('-restore',default=False)
    args = parser.parse_args()
    return args
def init_vars(restore,sess,saver):
    if restore:
        states=tf.train.get_checkpoint_state('/checkpoint/residual_fusion')
        checkpoint_paths=states.all_model_checkpoint_paths
        saver.recover_last_checkpoints(checkpoint_paths)
        saver.restore(sess,saver.last_checkpoints[-1])
    else:
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
if __name__ == '__main__':
    args=parse_arg()
    save_graph,restore=args.save_graph,args.restore
    training_set = Dataset(dir_path=)
    val_set = Dataset(dir_path=)
    model = Model()
    ms, pan, fusion = tf.placeholder(tf.float32, [None, 64, 64, 4]), \
                          tf.placeholder(tf.float32, [None, 64, 64, 4]), \
                          tf.placeholder(tf.float32, [None, 64, 64, 4])
    is_training = tf.placeholder(tf.bool)
    predict = Model.forward(ms, pan, is_training)
    loss = Model.loss(predict, fusion)
    acc,acc_update= Model.acc(predict, fusion)
    step = tf.Variable(0)
    opt = tf.train.AdamOptimizer(learning_rate=TC.learning_rate)
    train_op = opt.minimize(loss, global_step=step)
    saver=tf.train.Saver()
    best_acc=float('INF')
    with tf.Session() as sess:
        summary_writer=summary(sess=sess,graph_save=save_graph)
        init_vars(restore,sess,saver)
        for ms,pan,fusion in training_set.dataset:
            loss_value,step_value,_=sess.run([loss,step,train_op],
                                             feed_dict={'ms':ms,'pan':pan,'fusion':fusion})
            summary_writer.add_summary({'loss':loss_value},step_value)
            if step_value%TC.val_step==0:
                for val_ms,val_pan,val_fusion in val_set.dataset:
                    sess.run(acc_update,feed_dict={'ms':val_ms,'pan':val_pan,'fusion':val_fusion})
                val_acc=sess.run(acc)
                summary_writer.add_summary({'acc': val_acc}, step_value)
                if val_acc<best_acc:
                    saver.save(sess=sess,save_path='/checkpoint/residual_fusion',global_step=step)

