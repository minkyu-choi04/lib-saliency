import fixs2sals as f

import numpy as np
import scipy.io as io
from os import listdir
from os.path import isfile, join
import os 
import cv2

'''
20200924 Before running this script, change the file names of the MIT300's test dataset. 
Run the script 'mit300_addIndexFron.py' first.
'''

img_s = (360, 480)
server_type = 'libigpu1'
fn = 'save_data_test_e0_MIT300noIor.mat'#'save_data_test_e0_noIor.mat'
model_name = 'noIor'


if server_type == 'libigpu1' or 'libigpu5':
    base_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/mit300/BenchmarkIMAGES_newIndex/'
    save_dir = '/home/libiadm/HDD1/libigpu1/minkyu/mit300_test_salmap/'
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

print('for loop')


for sample in range(len(salmaps)):
    ## in case of MIT 300, size of image is not fixed. 
    ## Therefore, when saving salmap, it needs to be adjusted. 

    # read image size
    target = cv2.imread(os.path.join(base_dir, fn_test[sample]))
    target_s = np.shape(target)
    print(target_s)
    # resize saliency map 
    salmap_resize = cv2.resize(salmaps[sample], (target_s[1], target_s[0]))
    # save as image file
    cv2.imwrite(os.path.join(save_dir, fn_test[sample][4:]), salmap_resize*255)
    print(fn_test[sample], sample, os.path.join(save_dir, fn_test[sample][4:]))
