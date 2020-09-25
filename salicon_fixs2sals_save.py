import fixs2sals as f
import numpy as np
import scipy.io as io
from os import listdir
from os.path import isfile, join
import os 
import cv2


'''
20200924
This script is used for making saliency map from predicted fixation points. 
'''

img_s = (360, 480)
server_type = 'libigpu1'
fn = 'save_data_test_e40_caption_saliconNoIor.mat'
#model_name = 'with_ior_ks12'
model_name = 'caption_salicon_no_ior'


if server_type == 'libigpu1' or 'libigpu5':
    base_dir = '/home/min/datasets/salicon_original/image/images/test/1/'
    save_dir = '/home/libiadm/HDD1/libigpu1/minkyu/salicon_test_salmap/'
    save_dir = os.path.join(save_dir, model_name)
    print(save_dir)
else:
    print('wrong server type')


fn_test = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
fn_test = sorted(fn_test)
#print(fn_test[10][:-3])
#print(os.path.join(save_dir, fn_test[10][:-3]+'png'))


fixsss = io.loadmat(fn)['fix_history']
print(np.shape(fixsss), len(fixsss))
salmaps = f.fixs2sals(img_s, fixsss, 'max', k_size=101, std=22) # 5000, 360, 480
#salmaps = f.fixs2sals(img_s, fixsss, 'max', k_size=101, std=12) # 5000, 360, 480



for sample in range(len(salmaps)):
    cv2.imwrite(os.path.join(save_dir, fn_test[sample][:-3]+'png'), salmaps[sample]*255)
