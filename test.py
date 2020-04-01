import tensorflow as tf
import numpy as np
from model import LapFusion,DenseLapFusion,DenseMsScaleCom,DenseNoCom,DenseNoComV2
from dataset import Dataset
import cv2
from scipy import io
from TaPNN import PNN,RSIFNN
import os
import sewar
def normalize(img,reverse=False):
    if reverse:
        return img*1500.0
    else:
        return img.astype(np.float32)/1500.0
def visual_result(img):
    w,h,_=img.shape
    visual_image = np.zeros(shape=(w,h,3),dtype='uint8')
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
    return visual_image
def load_weight(saver,sess,name):
    states=tf.train.get_checkpoint_state('checkpoint%s/'%(name))
    checkpoint_paths=states.all_model_checkpoint_paths
    saver.recover_last_checkpoints(checkpoint_paths)
    saver.restore(sess,saver.last_checkpoints[-1])
def load_data(area,full_res=False):
    if full_res:
        img_dir='/root/FullRes/'
    else:
        img_dir='/root/RSF/test_img/'
        gt=np.load(img_dir+'gt_%s.npy'%(area))
        # gt_suburb=np.load(img_dir+'gt_suburb.npy')
    ms=\
        np.load(img_dir+'ms_%s.npy'%(area))
        # np.load(img_dir+'pan_downtown.npy'),
    pan=\
        np.load(img_dir+'pan_%s.npy'%(area))
        # np.load(img_dir+'pan_suburb.npy')
    ms=cv2.GaussianBlur(ms,ksize=(0,0),sigmaX=1,sigmaY=1)
    ms=cv2.resize(ms,dsize=(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
    pan = cv2.GaussianBlur(pan, ksize=(0, 0), sigmaX=1, sigmaY=1)
    pan=cv2.resize(pan,dsize=(0,0),fx=0.25,fy=0.25,interpolation=cv2.INTER_CUBIC)
    ms,pan=normalize(ms),\
        normalize(pan)
    # ms_suburb,pan_suburb=normalize(ms_suburb),\
    #     normalize(pan_suburb)
    # ms,pan=np.stack([ms_downtown,ms_suburb],0),\
    #     np.stack([pan_downtown,pan_suburb],0)
    pan=np.expand_dims(pan,-1)
    ms,pan=np.expand_dims(ms,0),np.expand_dims(pan,0)
    return ms,pan
    # return tf.convert_to_tensor(ms,tf.float32),\
    #     tf.convert_to_tensor(pan,tf.float32)
        # gt.astype(np.float32)

if __name__ == "__main__":
    full_res=True
    ms_size=512 if full_res else 128
    name='DenseLapFusion'
    model=DenseNoComV2(ms_size//4)
    ms,pan=tf.placeholder(tf.float32,[1,ms_size//4,ms_size//4,4]),\
        tf.placeholder(tf.float32,[1,ms_size*4//4,ms_size*4//4,1])
    if 'Lap' in name or 'Dense' in name:
        _, predict = model.forward(ms, pan)
    else:
        predict = model.forward(ms, pan)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        load_weight(saver, sess, 'DenseNoComV2')
        for area in ['downtown','suburb']:
            ms_value,pan_value=load_data(area,full_res)
            result=sess.run(predict,feed_dict={ms:ms_value,pan:pan_value})
            result=np.squeeze(result)
            result[result<0]=0
            result[result>1]=1
            result=normalize(result,True)
            # sCC=sewar.full_ref.scc(gt_downtown,result_downtown)
            if full_res:
                matfile='%s_%s%s.mat'%(name,area,'FS')
                io.savemat(matfile,{'I':result.astype(np.float64)})
            else:
                matfile = '%s_%s%s.mat' % (name, area, 'DS')
                io.savemat(matfile,{'I':result.astype(np.float64)})
            result=visual_result(result)
            if full_res:
                cv2.imwrite('%s%sFS.png' % (name, area), result)
            else:
                cv2.imwrite('%s%sDS.png' % (name, area), result)
        # gt_downtown,gt_suburb=visual_result(gt_downtown),visual_result(gt_suburb)
        # suburb=np.concatenate([result_suburb,gt_suburb],1)
        # downtown=np.concatenate([result_downtown,gt_downtown],1)
        #     cv2.imwrite('%s%s.png'%(name,area),result)
        # cv2.imwrite('downtown%s.png'%(name),downtown)
        # for i in range(30):
        #     result,gt=sess.run([predict,fusion])
        #     # result_downtown,result_suburb=result[0],result[1]
        #     result[result<0]=0
        #     result[result>1]=1
        #     # result_downtown[result_downtown>0]=1
        #     # result_suburb[result_suburb>1]=1
        #     result,gt=normalize(result,True),normalize(gt,True)
        #     result,gt=np.squeeze(result,0),np.squeeze(result,0)
        #     # result_downtown=result_downtown.astype(np.uint16),result_suburb=result_suburb.astype(np.uint16)
        #     # io.savemat('denselap_town.mat',{'I':result_downtown.astype(np.float64)})
        #     # io.savemat('denselap_sub.mat',{'I':result_suburb.astype(np.float64)})
        #     # gt_downtown,gt_suburb=visual_result(gt_downtown),visual_result(gt_suburb)
        #     result,gt=visual_result(result),visual_result(gt)
        #     img=np.concatenate([result,gt],1)
        #     cv2.imwrite('img%d.png'%(i),img)