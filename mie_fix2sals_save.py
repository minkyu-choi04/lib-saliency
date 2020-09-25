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
And then it calculates Cross Correlation value between target saliency maps and predicted ones. 
This calculation is only possible in MIE dataset because other datasets such as Salicon, MIT300 
does not provide GT salmaps for test data. 

In case of MIE dataset, I manually changed file names from 1.png to 01.png...
There are around 25 images and it is easy to do it by hands. 
Also, I moved images originally inside '.../stimuli' path into '.../stimuli/0/'. 
This is because pytorch dataloader requires this structure. 

Datasets downloaded: https://www-percept.irisa.fr/asperger_to_kanner/
'''


img_s = (360, 480)
server_type = 'libigpu1'
data_type = 'no'
fn = 'save_data_test_e40_caption_mieNo_withIor.mat'
#model_name = 'with_ior_ks12'
model_name = 'caption_mieNo_whthior'


if server_type == 'libigpu1' or 'libigpu5':
    if data_type == 'fo':
        # base_dir: path where the stimuli images are saved. 
        base_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_Fo/stimuli/0/'
        sal_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_Fo/saliencyMaps/'
    elif data_type == 'no':
        base_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_No/stimuli/0/'
        sal_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/ASD/MIE_No/saliencyMaps/'
    save_dir = '/home/libiadm/HDD1/libigpu1/minkyu/mie_test_salmap/'
    save_dir = os.path.join(save_dir, model_name)
    print(save_dir)
else:
    print('wrong server type')


fn_test = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
fn_test = sorted(fn_test)
print(fn_test)
#print(fn_test[10][:-3])
#print(os.path.join(save_dir, fn_test[10][:-3]+'png'))


fixsss = io.loadmat(fn)['fix_history']
print(np.shape(fixsss), len(fixsss))
salmaps = f.fixs2sals(img_s, fixsss, 'max', k_size=101, std=22) # 5000, 360, 480
#salmaps = f.fixs2sals(img_s, fixsss, 'max', k_size=101, std=12) # 5000, 360, 480



for sample in range(len(salmaps)):
    cv2.imwrite(os.path.join(save_dir, fn_test[sample][:-3]+'png'), salmaps[sample]*255)



##############################################
## calculate metric value 
import saliency_metrics as sm

cc = 0.0
for sample in range(len(salmaps)):
    print(os.path.join(sal_dir, fn_test[sample]))
    sal_target = cv2.imread(os.path.join(sal_dir, fn_test[sample])) / 255.0
    sal_target = cv2.resize(sal_target, (img_s[1], img_s[0]))
    sal_target = np.squeeze(sal_target[:, :, 0])
    print(np.shape(sal_target), np.shape(salmaps[sample]))
    #print(sal_target[180, 240, :])
    cc = cc + sm.CC(salmaps[sample], sal_target)
print('CC: ', cc/(sample+1))

