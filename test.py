import tensorflow as tf
import numpy as np
from model import LapFusion,DenseLapFusion
import cv2
def normalize(img,reverse=False):
    if reverse:
        return (img*1500.0).astype(np.uint16)
    else:
        return img.astype(np.float)/1500.0
def visual_result(img):
    visual_image = np.zeros(shape=(256,256,3),dtype='uint8')
    for iband in range(1, 4):
        imgMatrix=img[:,:,iband-1]
        zeros = np.size(imgMatrix) - np.count_nonzero(imgMatrix)
        minVal = np.percentile(imgMatrix, float(zeros / np.size(imgMatrix) * 100 + 0.15))
        maxVal = np.percentile(imgMatrix, 99)

        idx1 = imgMatrix < minVal
        idx2 = imgMatrix > maxVal
        idx3 = ~idx1 & ~idx2
        imgMatrix[idx1] = imgMatrix[idx1] * 20 / minVal
        imgMatrix[idx2] = 255
        idx1 = None
        idx2 = None
        imgMatrix[idx3] = pow((imgMatrix[idx3] - minVal) / (maxVal - minVal), 0.9) * 255
        visual_image[:, :, iband-1] = imgMatrix
        imgMatrix = None
    return visual_image.transpose(2,0,1)
def load_weight(saver,sess):
    states=tf.train.get_checkpoint_state('checkpointDenseLapFusion/')
    checkpoint_paths=states.all_model_checkpoint_paths
    saver.recover_last_checkpoints(checkpoint_paths)
    saver.restore(sess,saver.last_checkpoints[-1])
def load_data():
    gt_downtown,ms_downtown,pan_downtown=\
        np.load('gt_downtown.npy'),np.load('ms_downtown.npy'),np.load('pan_downtown.npy')
    gt_suburb,ms_suburb,pan_suburb=\
        np.load('gt_suburb.npy'),np.load('ms_suburb.npy'),np.load('pan_suburb.npy')
    ms_downtown,pan_downtown=normalize(ms_downtown),normalize(pan_downtown)
    ms_suburb,pan_suburb=normalize(ms_suburb),normalize(pan_suburb)
    ms,pan=np.stack([ms_downtown,ms_suburb],0),np.stack([pan_downtown,pan_suburb],0)
    pan=np.expand_dims(pan,-1)
    return tf.convert_to_tensor(ms,tf.float32),tf.convert_to_tensor(pan,tf.float32),\
        gt_downtown,gt_suburb
ms,pan,gt_downtown,gt_suburb=load_data()
lf=DenseLapFusion(ms_size=64)
predict=lf.forward(ms,pan)
saver=tf.train.Saver()
with tf.Session() as sess:
    load_weight(saver,sess)
    result=sess.run(predict)
    result_downtown,result_suburb=result[0],result[1]
    result_downtown,result_suburb=normalize(result_downtown,True),normalize(result_suburb,True)
    gt_downtown,gt_suburb=visual_result(gt_downtown),visual_result(gt_suburb)
    result_downtown,result_suburb=visual_result(result_downtown),visual_result(result_suburb)
