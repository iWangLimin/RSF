class DatasetConfig():
    remote_root='/home/zhou/wlm/RSF/dataset'
    img_names=['GF2_PMS2_E117.4_N39.1_20170510_L1A0002351826','GF2_PMS1_E117.6_N39.0_20171108_L1A0002748722']
                # 'GF2_PMS1_E117.6_N39.2_20171108_L1A0002748717',\]
    train_scale,validate_scale,test_scale=0.8,0.1,0.1
    MS_crop_size=256
    MS_crop_step=64
    Max_Pixel,Min_Pixel=1500.0,0.0
class TrainingConfig():
    batch_size=4
    weight_dacay=0.0001
    learning_rate=0.001
    val_step=5000//batch_size