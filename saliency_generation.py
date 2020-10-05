import torch

def get_normalized_fms_01(fms):
    '''
    Return normalized FMs from 0 to 1

    Args: 
        fms: (b, c, h, w)
    return:
        fms_norm: (b, c, h, w)
    '''
    fms_s = fms.size()
    fms_lin = fms.view(fms_s[0], fms_s[1], -1)

    global_max, _ = torch.max(fms_lin, 2) # (b, c)
    global_min, _ = torch.min(fms_lin, 2) # (b, c)
    div = global_max-global_min

    zero_indicator = (div == 0) # avoid division by zero
    div = div + zero_indicator
    div = div.unsqueeze(2).unsqueeze(2) # (b, c, 1, 1)
    #print(zero_indicator, zero_indicator.size())

    fms_01 = (fms - global_min.unsqueeze(2).unsqueeze(2)) / div
    return fms_01

def get_itti_normalization(fms, patch_portion):
    '''
    Function for normalizing FMs described in Itti's 1998 paper. 

    Args: 
        fms: (b, c, h, w)
        patch_portion: float 0~1. It describes the local patch for maxpool. 
                        If this value is 0.1 and the size of fms is (100x100), 
                        maxpool will be applied to every 10x10 local patches. 

    Return:
        fms_norm: (b, c, h, w) normalized fms
    '''
    ## normalize value range 0~1 independently channel by channel
    fms_s = fms.size()
    '''fms_lin = fms.view(fms_s[0], fms_s[1], -1)
    global_max, _ = torch.max(fms_lin, 2) # (b, c)
    zero_indicator = (global_max == 0) # avoid division by zero
    global_max = global_max + zero_indicator
    global_max = global_max.unsqueeze(2).unsqueeze(2) # (b, c, 1, 1)
    #print(zero_indicator, zero_indicator.size())

    fms_01 = fms / global_max'''
    fms_01 = get_normalized_fms_01(fms)
    #print('max fms_01: ', torch.max(fms_01))

    ## Calculate local maxima
    kernel_size = (int(fms_s[2]*patch_portion), int(fms_s[3]*patch_portion))
    local_max = torch.nn.functional.max_pool2d(fms_01, kernel_size=kernel_size) # (b, c, h', w')
    local_max_mean = torch.mean(local_max.view(fms_s[0], fms_s[1], -1), 2) # (b, c)
    local_max_mean = local_max_mean.unsqueeze(2).unsqueeze(2) # (b, c, 1, 1)

    ## Normalize
    fms_norm = fms_01 * (1.0 - local_max_mean)**2
    #print('inside norm01, ', torch.max(fms_01), torch.min(fms_01))

    return fms_norm

def get_itti_saliency(fms, patch_portion, threshold=0.2, isNormalize=True):
    '''
    Function for making saliency map described in Itti's 1998 paper. 

    Args: 
        fms: (b, c, h, w)
        patch_portion: float 0~1. It describes the local patch for maxpool. 
                        If this value is 0.1 and the size of fms is (100x100), 
                        maxpool will be applied to every 10x10 local patches. 

    Return:
        sal_map: (b, 1, h, w) saliency map 
    '''

    fms_norm = get_itti_normalization(fms, patch_portion)

    fms_norm = torch.nn.functional.relu(fms_norm - threshold)

    sal_map = torch.mean(fms_norm, 1)
    sal_map = sal_map.unsqueeze(1)
    if isNormalize:
        sal_map = get_normalized_fms_01(sal_map)
        #print('inside get_itti_saliency, ', torch.max(sal_map), torch.min(sal_map))
    return sal_map, fms_norm

def clear_boundary_to_zeros(salmap, device='cuda'):
    ''' Clear boundary of salmap by setting the values to 0 while maintaining the other values. 
    The size of input and output of this function are the same. 
    Args: 
        salmap: (b, c, h, w)
    Return:
        salmap_clear: (b, c, h, w)
    '''
    with torch.no_grad():
        s = salmap.size()
        mask = torch.zeros((s[0], 1, s[2], s[3]), dtype=torch.float32, device=device)
        mask[:, :, 1:-1, 1:-1] = torch.ones((s[0], 1, s[2]-2, s[3]-2), dtype=torch.float32, device=device)
        
        salmap_clear = salmap * mask
        return salmap_clear



def get_itti_saliency_multiLayer(fms3, fms4, fms5, patch_portion):
    '''
    Make saliency map using last three blocks

    Args:
    Args: 
        fms3, fms4, fms5: (b, c, h, w)
        patch_portion: float 0~1. It describes the local patch for maxpool. 
                        If this value is 0.1 and the size of fms is (100x100), 
                        maxpool will be applied to every 10x10 local patches. 

    Return:
        sal_map: (b, 1, h, w) saliency map 
    '''
    with torch.no_grad():
        layer = 0

        fms_layer = fms3
        sal_map3, fms_norm = get_itti_saliency(fms_layer, patch_portion)
        final_sal_map = clear_boundary_to_zeros(sal_map3)#sal_map3[:, :, 1:-1, 1:-1]
        fm3_s = final_sal_map.size() 

        fms_layer = fms4
        sal_map4, fms_norm = get_itti_saliency(fms_layer, patch_portion, threshold=0.3)
        sal_map4 = clear_boundary_to_zeros(sal_map4)
        #final_sal_map = final_sal_map + torch.nn.functional.interpolate(sal_map4[:, :, 1:-1, 1:-1], (fm3_s[2], fm3_s[3]), mode='bilinear')
        final_sal_map = final_sal_map + torch.nn.functional.interpolate(sal_map4, (fm3_s[2], fm3_s[3]), mode='bilinear')

        fms_layer = fms5
        sal_map5, fms_norm = get_itti_saliency(fms_layer, patch_portion)
        sal_map5 = clear_boundary_to_zeros(sal_map5)
        #final_sal_map = final_sal_map + torch.nn.functional.interpolate(sal_map5[:, :, 1:-1, 1:-1], (fm3_s[2], fm3_s[3]), mode='bilinear')
        final_sal_map = final_sal_map + torch.nn.functional.interpolate(sal_map5, (fm3_s[2], fm3_s[3]), mode='bilinear')
        final_sal_map = final_sal_map/3


    return final_sal_map, sal_map3, sal_map4, sal_map5






