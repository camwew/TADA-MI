import numpy as np
import torch

def oneMix(mask, data = None, target = None, target_2 = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    '''
    if target_2 is not None and target_3 is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target_2[0])
        target_2 = (stackedMask0*target_2[0]+(1-stackedMask0)*target_2[1]).unsqueeze(0)
        stackedMask0, _ = torch.broadcast_tensors(mask[0].unsqueeze(-1), target_3[0])
        target_3 = (stackedMask0*target_3[0]+(1-stackedMask0)*target_3[1]).unsqueeze(0)
        return data, target, target_2, target_3
    '''
    if target_2 is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target_2[0])
        target_2 = (stackedMask0*target_2[0]+(1-stackedMask0)*target_2[1]).unsqueeze(0)
        return data, target, target_2
    else:
        raise Exception("Invalid mix type!") 
        
    return data, target
    
    
    
def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N
