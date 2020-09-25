import numpy as np
import cv2


def mark_point_np(img, fix):
    '''
    Args:
        img: mono channel image, (height, width)
        fix: (2,), (x,y) order, -1~1 range
    return: 
        img: (channel, height, width)
    '''
    img_s = np.shape(img)
    x_n = int((fix[0]+1)/2.0 * img_s[1])
    y_n = int((fix[1]+1)/2.0 * img_s[0])

    img[y_n, x_n] = 1.0 ## I did not accumulate it. If there is a point occuring twice, it will only be counted once. 
    return img

def mark_fixations_seq_np(img, fixs):
    '''
    Plot fixations of a sequence (trial)
    Args:
        img: mono channel image, (height, width)
        fixs: (step, 2), (x,y) order, -1~1 range
    return: 
        img_fixs: (channel, height, width)
    '''
    n_steps = np.shape(fixs)[0]
    for step in range(n_steps):
        img = mark_point_np(img, fixs[step, :])
    return img

def mark_fixations_img_np(img, fixss):
    '''
    Plot fixations of multiple sequences (trials)
    Args:
        img: mono channel image (height, width)
        fixss: (trial, step, 2), (x,y) order, -1~1 range
    return: 
        img_fixs: (channel, height, width)
    '''
    n_trials = np.shape(fixss)[0]
    for trial in range(n_trials):
        img = mark_fixations_seq_np(img, fixss[trial, :, :])
    return img

def mark_fixations_dataset_np(img_s, fixsss):
    '''
    Plot fixations of multiple sequences (trials) of all test dataset
    Args:
        img_s: image size (int height, int width)
        fixsss: (#total test images, trial, step, 2), (x,y) order, -1~1 range
    return: 
        imgs: (#total test images, (height, width)), list of numpy arrays
    '''

    imgs = []
    n_tests = np.shape(fixsss)[0]
    for test in range(n_tests):
        img = np.zeros(img_s, dtype=float)  
        img = mark_fixations_img_np(img, fixsss[test, :, :, :])
        imgs.append(img)
    return imgs

def fixs2sals(img_s, fixsss, norm_type, k_size=101, std=22):
    '''
    Convert fixation sequences to saliency maps. 
    Args:
        img_s: image size (int height, int width)
        fixsss: (#total test images, trial, step, 2), (x,y) order, -1~1 range
        norm_type: 'max' (max pixel value will be 1) or 'sum' (sum of all pixels will be 1)
    return: 
        salmaps: (#total test images, (height, width)), list of numpy arrays
    '''
    salmaps = []
    fixmaps = mark_fixations_dataset_np(img_s, fixsss)

    n_tests = len(fixmaps)
    for test in range(n_tests):
        salmap = cv2.GaussianBlur(np.expand_dims(fixmaps[test], axis=-1),(k_size,k_size), std)
        if norm_type == 'max':
            salmap = salmap / np.max(salmap)
        if norm_type == 'sum':
            salmap = salmap / np.sum(salmap)

        salmaps.append(salmap)

    return salmaps

