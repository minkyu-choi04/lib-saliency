import os
from os import listdir
from os.path import isfile, join
import cv2

'''
20200924
MIT300 dataset's images have names of 'i1.jpg', 'i2.jpg' ... 'i300.jpg'. 
It is not desired because when sort the file names, it is not sorted properly. 
Therefore, I need to add an index in front of the file name to make it '001_i1.jpg', '002_i2.jpg'...


After running this script, go to the directory and make dir named 0. 
And then mv all files into the new dir 0. 
Just follow the command below. Change the dir below based on the server.  
>> cd /home/libiadm/HDD1/libigpu1/minkyu/datasets/mit300/BenchmarkIMAGES_newIndex/
>> mkdir 0
>> mv * ./0
'''


base_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/mit300/BenchmarkIMAGES/'
new_base_dir = '/home/libiadm/HDD1/libigpu1/minkyu/datasets/mit300/BenchmarkIMAGES_newIndex/'


fn_test = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
fn_test_new = []

for i in range(len(fn_test)):
    target = cv2.imread(os.path.join(base_dir, fn_test[i]))

    spl = fn_test[i].split('.')
    idx = spl[0]
    idx = ''.join(jj for jj in idx if jj.isdigit())
    if len(idx) == 1:
        reg = '00' + idx
    elif len(idx) == 2:
        reg = '0' + idx
    elif len(idx) == 3:
        reg = idx
    else:
        print('problem')

    nfn = reg + '_' + spl[0] + '.jpg'
    print(fn_test[i], nfn)
    fn_test_new.append(nfn)

    cv2.imwrite(os.path.join(new_base_dir, nfn), target)
    print(os.path.join(new_base_dir, nfn))


