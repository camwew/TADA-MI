import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.utils import cropping_center
from .utils import gradient_loss, dice_loss, mse_loss, msge_loss, xentropy_loss
#from .DA import image_dis_forward
from backpack import backpack, extend
from backpack.extensions import BatchGrad, Variance, DiagGGNMC, DiagGGNExact, DiagHessian, BatchDiagGGNExact
from .image_mix import oneMix, generate_class_mask
from collections import OrderedDict

import time
def get_second_order_grad(grads, xs):
    start = time.time()
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        assert grad.size() == x.size(), "Unmatched size between first-oreder gradient and model parameters."
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(outputs=g, inputs=x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to("cuda"))
        print('Time used is ', time.time() - start)
    for grad in grads2:  # check size
        print(grad.size())

    return 

def dict2tensor(_dict):
    dict_values = [_dict[key] for key in sorted(_dict.keys())]
    return torch.cat(tuple([t.view(-1) for t in dict_values]))


def train_step(batch_data_source, batch_data_target, batch_data_target_few, run_info, warmup_epoch=1):
    # TODO: synchronize the attach protocol
    
    run_info, state_info = run_info
    epoch_count = state_info['epoch']
    step_count = state_info['step']
    bank_interval = 1
    
    loss_func_dict = {
        "bce": xentropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    # use 'ema' to add for EMA calculation, must be scalar!
    result_dict = {"EMA": {}}
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####imgs_mix
    model = run_info["net"]["desc"]
    if epoch_count >= warmup_epoch:
        optimizer = run_info["net"]["optimizer"]
    else:
        optimizer = run_info["net"]["optimizer_warmup"]
    
    ema_source_np_gradient_mean = run_info["net"]["ema_source_np_gradient_mean"]
    ema_source_np_gradient_var = run_info["net"]["ema_source_np_gradient_var"]
    ema_mix_np_gradient_mean = run_info["net"]["ema_mix_np_gradient_mean"]
    ema_mix_np_gradient_var = run_info["net"]["ema_mix_np_gradient_var"]
    ema_source_tp_3class_gradient_mean = run_info["net"]["ema_source_tp_3class_gradient_mean"]
    ema_source_tp_3class_gradient_var = run_info["net"]["ema_source_tp_3class_gradient_var"]
    ema_mix_tp_3class_gradient_mean = run_info["net"]["ema_mix_tp_3class_gradient_mean"]
    ema_mix_tp_3class_gradient_var = run_info["net"]["ema_mix_tp_3class_gradient_var"]
    ema_mix_tp_6class_gradient_mean = run_info["net"]["ema_mix_tp_6class_gradient_mean"]
    ema_mix_tp_6class_gradient_var = run_info["net"]["ema_mix_tp_6class_gradient_var"]
    
    source_np_mean_grad_bank = run_info["net"]["source_np_mean_grad_bank"]
    source_np_var_grad_bank = run_info["net"]["source_np_var_grad_bank"]
    source_tp_3_mean_grad_bank = run_info["net"]["source_tp_3_mean_grad_bank"]
    source_tp_3_var_grad_bank = run_info["net"]["source_tp_3_var_grad_bank"]
    source_tp_6_mean_grad_bank = run_info["net"]["source_tp_6_mean_grad_bank"]
    source_tp_6_var_grad_bank = run_info["net"]["source_tp_6_var_grad_bank"]
    
    flat_layer_grad_bank_tp = run_info["net"]["flat_layer_grad_bank_tp"]
    flat_layer_grad_bank_np = run_info["net"]["flat_layer_grad_bank_np"]
    
    bce_extended = run_info["net"]["bce_extended"]
    ########################################################################################################
    
    # Input configuration
    ###################
    imgs_source = batch_data_source["img"]
    true_np_source = batch_data_source["np_map"]
    true_hv_source = batch_data_source["hv_map"]

    imgs_source = imgs_source.to("cuda").type(torch.float32)  # to NCHW
    imgs_source = imgs_source.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np_source = true_np_source.to("cuda").type(torch.int64)
    true_hv_source = true_hv_source.to("cuda").type(torch.float32)

    true_np_onehot_source = (F.one_hot(true_np_source, num_classes=2)).type(torch.float32)
    true_dict_source = {
        "np": true_np_onehot_source,
        "hv": true_hv_source,
    }

    if model.num_types_source is not None:
        true_tp_source = batch_data_source["tp_map"]
        true_tp_source = torch.squeeze(true_tp_source).to("cuda").type(torch.int64)
        true_tp_onehot_source = F.one_hot(true_tp_source, num_classes=model.num_types_source)
        true_tp_onehot_source = true_tp_onehot_source.type(torch.float32)
        true_dict_source["tp"] = true_tp_onehot_source
        #true_dict_source["tp_source"] = true_tp_onehot_source
    
    
    ###################
    
    imgs_target = batch_data_target["img"]
    true_np_target = batch_data_target["np_map"]
    true_hv_target = batch_data_target["hv_map"]

    imgs_target = imgs_target.to("cuda").type(torch.float32)  # to NCHW
    imgs_target = imgs_target.permute(0, 3, 1, 2).contiguous()
    
    # HWC
    #true_np_target = true_np_target.to("cuda").type(torch.int64)
    #true_hv_target = true_hv_target.to("cuda").type(torch.float32)

    #true_np_onehot_target = (F.one_hot(true_np_target, num_classes=2)).type(torch.float32)
    #true_dict_target = {
    #    "np": true_np_onehot_target,
    #    "hv": true_hv_target,
    #}
    
    if model.num_types is not None:
        true_tp_target = batch_data_target["tp_map"]
        true_tp_target = torch.squeeze(true_tp_target).to("cuda").type(torch.int64)


    pred_dict_target = model(imgs_target)
    pred_dict_target = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict_target.items()]
    )
    
    pred_dict_target["np"] = F.softmax(pred_dict_target["np"], dim=-1)
    target_np_pseu_label = torch.argmax(pred_dict_target["np"], dim=-1)
    target_np_pseu_label[pred_dict_target["np"].max(-1)[0] < 0.7] = 0
    pred_dict_target["np"] = target_np_pseu_label
    
    pred_dict_target["tp"] = F.softmax(pred_dict_target["tp"], dim=-1)
    target_tp_pseu_label = torch.argmax(pred_dict_target["tp"], dim=-1)
    target_tp_pseu_label[pred_dict_target["tp"].max(-1)[0] < 0.4] = 0
    pred_dict_target["tp"] = target_tp_pseu_label
        
    pred_dict_target["tp_source"] = torch.argmax(F.softmax(pred_dict_target["tp_source"], dim=-1), dim=-1)
    
    ###################
    imgs_target_few = batch_data_target_few["img"]
    true_np_target_few = batch_data_target_few["np_map"]
    true_hv_target_few = batch_data_target_few["hv_map"]

    imgs_target_few = imgs_target_few.to("cuda").type(torch.float32)  # to NCHW
    imgs_target_few = imgs_target_few.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np_target_few = true_np_target_few.to("cuda").type(torch.int64)
    true_hv_target_few = true_hv_target_few.to("cuda").type(torch.float32)

    true_np_onehot_target_few = (F.one_hot(true_np_target_few, num_classes=2)).type(torch.float32)
    true_dict_target_few = {
        "np": true_np_onehot_target_few,
        "hv": true_hv_target_few,
    }

    if model.num_types is not None:
        true_tp_target_few = batch_data_target_few["tp_map"]
        true_tp_target_few = torch.squeeze(true_tp_target_few).to("cuda").type(torch.int64)
        true_tp_onehot_target_few = F.one_hot(true_tp_target_few, num_classes=model.num_types)
        true_tp_onehot_target_few = true_tp_onehot_target_few.type(torch.float32)
        true_dict_target_few["tp"] = true_tp_onehot_target_few
        
        true_tp_target_few_3 = batch_data_target_few["tp_map"]
        true_tp_target_few_3 = torch.squeeze(true_tp_target_few_3).to("cuda").type(torch.int64)
        true_tp_target_few_3[true_tp_target_few == 1] = 2
        true_tp_target_few_3[true_tp_target_few == 3] = 2
        true_tp_target_few_3[true_tp_target_few == 4] = 2
        true_tp_target_few_3[true_tp_target_few == 5] = 2
        true_tp_target_few_3[true_tp_target_few == 2] = 1
        true_tp_target_few_3[true_tp_target_few == 6] = 3
        true_tp_onehot_target_few_3 = F.one_hot(true_tp_target_few_3, num_classes=model.num_types_source)
        true_tp_onehot_target_few_3 = true_tp_onehot_target_few_3.type(torch.float32)
        true_dict_target_few["tp_source"] = true_tp_onehot_target_few_3
        
    ###################
    assert imgs_source.shape[0] == imgs_target.shape[0] == imgs_target_few.shape[0], "Inconsistent batch size!!!"
    for batch_id in range(imgs_source.shape[0]):
        #classes_mix_target = torch.unique(true_tp_target_few[batch_id])
        #classes_mix_target = classes_mix_target[classes_mix_target != 0]
        #nclasses = classes_mix_target.shape[0]
        #classes_list_target = np.random.choice(nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        #classes_mix_target = (classes_mix_target[torch.Tensor(classes_list_target).long()]).cuda()
        classes_mix_target = torch.tensor([1., 2., 3., 4., 5., 6.]).cuda()
        MixMask = generate_class_mask(true_tp_target_few[batch_id], classes_mix_target).unsqueeze(0).cuda()
        mix_image_sample, mix_tp_label_sample, mix_np_label_sample = oneMix(MixMask, data=torch.cat((imgs_target_few[batch_id].unsqueeze(0), imgs_target[batch_id].unsqueeze(0))), target=torch.cat((true_tp_target_few[batch_id].unsqueeze(0), pred_dict_target["tp"][batch_id].unsqueeze(0).detach())), target_2=torch.cat((true_np_target_few[batch_id].unsqueeze(0), pred_dict_target["np"][batch_id].unsqueeze(0).detach())))
        
        if batch_id == 0:
            imgs_mix = mix_image_sample
            tp_labels_mix = mix_tp_label_sample
            np_labels_mix = mix_np_label_sample
        else:
            imgs_mix = torch.cat([imgs_mix, mix_image_sample], 0)
            tp_labels_mix = torch.cat([tp_labels_mix, mix_tp_label_sample], 0)
            np_labels_mix = torch.cat([np_labels_mix, mix_np_label_sample], 0)
    
    tp_labels_mix = tp_labels_mix.type(torch.int64)
    np_labels_mix = np_labels_mix.type(torch.int64)
    true_dict_mix = {}
    true_dict_mix["np"] = F.one_hot(np_labels_mix, num_classes=2).type(torch.float32)
    true_dict_mix["tp"] = F.one_hot(tp_labels_mix, num_classes=model.num_types).type(torch.float32)
    
    tp_labels_mix_3 = tp_labels_mix.clone()
    tp_labels_mix_3[tp_labels_mix == 1] = 2
    tp_labels_mix_3[tp_labels_mix == 3] = 2
    tp_labels_mix_3[tp_labels_mix == 4] = 2
    tp_labels_mix_3[tp_labels_mix == 5] = 2
    tp_labels_mix_3[tp_labels_mix == 2] = 1
    tp_labels_mix_3[tp_labels_mix == 6] = 3
    true_dict_mix["tp_source"] = F.one_hot(tp_labels_mix_3, num_classes=model.num_types_source).type(torch.float32)
    
    ########################################################################################################

    ######################################
    model.train()
    
    assert imgs_source.shape[0] == imgs_mix.shape[0] == imgs_target_few.shape[0], "Unmatched batch size!!!"
    batch_size = imgs_source.shape[0]
    pred_dict_source_mix = model(torch.cat([imgs_source, imgs_mix], 0))
    
    pred_dict_source = OrderedDict(
        [[k, v[:batch_size]] for k, v in pred_dict_source_mix.items()]
    )
    pred_dict_source['hv'] = pred_dict_source['hv'].permute(0, 2, 3, 1).contiguous()
    pred_dict_source['np_feat'] = pred_dict_source['np_feat'].permute(0, 2, 3, 1).contiguous()
    pred_dict_source['tp_feat'] = pred_dict_source['tp_feat'].permute(0, 2, 3, 1).contiguous()
    pred_dict_source["tp"] = pred_dict_source["tp_source"]
    pred_dict_source.pop("tp_source")
    
    ####
    
    pred_dict_mix = OrderedDict(
        [[k, v[batch_size:]] for k, v in pred_dict_source_mix.items()]
    )
    pred_dict_mix['hv'] = pred_dict_mix['hv'].permute(0, 2, 3, 1).contiguous()
    pred_dict_mix['np_feat'] = pred_dict_mix['np_feat'].permute(0, 2, 3, 1).contiguous()
    pred_dict_mix['tp_feat'] = pred_dict_mix['tp_feat'].permute(0, 2, 3, 1).contiguous()
    
    ########################################################################################################
    if epoch_count >= warmup_epoch:
        
        optimizer.zero_grad()
        loss_gradient_np = bce_extended(pred_dict_source_mix["np"], torch.cat([true_np_source, np_labels_mix], 0))
        with backpack(BatchGrad()):
            loss_gradient_np.backward(
                inputs=list(model.np_layer.parameters()), retain_graph=True, create_graph=True
            )
        
        dict_grads_np = OrderedDict(
                [
                    (name, weights.grad_batch.clone().reshape(weights.grad_batch.size(0), -1))
                    for name, weights in model.np_layer.named_parameters()
                ]
            )
        
          
        ############
        dict_grads_source_np = OrderedDict(
                [
                    (name, weights[:batch_size])
                    for name, weights in dict_grads_np.items()
                ]
            )
        grads_source_np_mean = {}
        grads_source_np_var = {}
        for name, env_grads in dict_grads_source_np.items():
            env_mean = env_grads.mean(dim=0, keepdim=True)
            grads_source_np_mean[name] = env_mean.squeeze()
            env_grads_centered = env_grads - env_mean
            grads_source_np_var[name] = (env_grads_centered).pow(2).mean(dim=0)
        grads_source_np_mean = ema_source_np_gradient_mean.update(grads_source_np_mean)
        grads_source_np_var = ema_source_np_gradient_var.update(grads_source_np_var)
        grads_source_np_mean = dict2tensor(grads_source_np_mean)
        grads_source_np_var = dict2tensor(grads_source_np_var)
        
        source_np_mean_grad_bank.update(grads_source_np_mean.detach().cpu().numpy())
        source_np_var_grad_bank.update(grads_source_np_var.detach().cpu().numpy())
        if source_np_mean_grad_bank.U is not None:
            grads_source_np_mean = torch.mm(grads_source_np_mean.unsqueeze(0), source_np_mean_grad_bank.U).squeeze()
        if source_np_var_grad_bank.U is not None:
            grads_source_np_var = torch.mm(grads_source_np_var.unsqueeze(0), source_np_var_grad_bank.U).squeeze()
    
        ############
        dict_grads_mix_np = OrderedDict(
                [
                    (name, weights[batch_size:])
                    for name, weights in dict_grads_np.items()
                ]
            )
        grads_mix_np_mean = {}
        grads_mix_np_var = {}
        for name, env_grads in dict_grads_mix_np.items():
            env_mean = env_grads.mean(dim=0, keepdim=True)
            grads_mix_np_mean[name] = env_mean.squeeze()
            env_grads_centered = env_grads - env_mean
            grads_mix_np_var[name] = (env_grads_centered).pow(2).mean(dim=0)
        grads_mix_np_mean = ema_mix_np_gradient_mean.update(grads_mix_np_mean)
        grads_mix_np_var = ema_mix_np_gradient_var.update(grads_mix_np_var)
        grads_mix_np_mean = dict2tensor(grads_mix_np_mean)
        grads_mix_np_var = dict2tensor(grads_mix_np_var)
        
        if source_np_mean_grad_bank.U is not None:
            grads_mix_np_mean = torch.mm(grads_mix_np_mean.unsqueeze(0), source_np_mean_grad_bank.U).squeeze()
        if source_np_var_grad_bank.U is not None:
            grads_mix_np_var = torch.mm(grads_mix_np_var.unsqueeze(0), source_np_var_grad_bank.U).squeeze()
        
        
        ####################################
        loss_gradient_tp_3class = bce_extended(pred_dict_source_mix["tp_source"], torch.cat([true_tp_source, tp_labels_mix_3], 0))
        with backpack(BatchGrad()):
            loss_gradient_tp_3class.backward(
                inputs=list(model.tp_layer_source.parameters()), retain_graph=True, create_graph=True
            )
        
        dict_grads_tp_3class = OrderedDict(
                [
                    (name, weights.grad_batch.clone().squeeze())
                    for name, weights in model.tp_layer_source.named_parameters()
                ]
            )
            
        ############
        dict_grads_source_tp_3class_raw = OrderedDict(
                [
                    (name, weights[:batch_size])
                    for name, weights in dict_grads_tp_3class.items()
                ]
            )
        dict_grads_source_tp_3class = OrderedDict(
                [
                    (name, weights.reshape(weights.shape[0], -1))
                    for name, weights in dict_grads_source_tp_3class_raw.items()
                ]
            )
        
        grads_source_tp_3class_mean = {}
        grads_source_tp_3class_var = {}
        for name, env_grads in dict_grads_source_tp_3class.items():
            env_mean = env_grads.mean(dim=0, keepdim=True)
            grads_source_tp_3class_mean[name] = env_mean.squeeze()
            env_grads_centered = env_grads - env_mean
            grads_source_tp_3class_var[name] = (env_grads_centered).pow(2).mean(dim=0)
        grads_source_tp_3class_mean = ema_source_tp_3class_gradient_mean.update(grads_source_tp_3class_mean)
        grads_source_tp_3class_var = ema_source_tp_3class_gradient_var.update(grads_source_tp_3class_var)
        grads_source_tp_3class_mean = dict2tensor(grads_source_tp_3class_mean)
        grads_source_tp_3class_var = dict2tensor(grads_source_tp_3class_var)
        
        source_tp_3_mean_grad_bank.update(grads_source_tp_3class_mean.detach().cpu().numpy())
        source_tp_3_var_grad_bank.update(grads_source_tp_3class_var.detach().cpu().numpy())
        if source_tp_3_mean_grad_bank.U is not None:
            grads_source_tp_3class_mean = torch.mm(grads_source_tp_3class_mean.unsqueeze(0), source_tp_3_mean_grad_bank.U).squeeze()
        if source_tp_3_var_grad_bank.U is not None:
            grads_source_tp_3class_var = torch.mm(grads_source_tp_3class_var.unsqueeze(0), source_tp_3_var_grad_bank.U).squeeze()
            
        ############
        dict_grads_mix_tp_3class = OrderedDict(
                [
                    (name, weights[batch_size:].reshape(weights[batch_size:].shape[0], -1))
                    for name, weights in dict_grads_tp_3class.items()
                ]
            )
        grads_mix_tp_3class_mean = {}
        grads_mix_tp_3class_var = {}
        for name, env_grads in dict_grads_mix_tp_3class.items():
            env_mean = env_grads.mean(dim=0, keepdim=True)
            grads_mix_tp_3class_mean[name] = env_mean.squeeze()
            env_grads_centered = env_grads - env_mean
            grads_mix_tp_3class_var[name] = (env_grads_centered).pow(2).mean(dim=0)
        grads_mix_tp_3class_mean = ema_mix_tp_3class_gradient_mean.update(grads_mix_tp_3class_mean)
        grads_mix_tp_3class_var = ema_mix_tp_3class_gradient_var.update(grads_mix_tp_3class_var)
        grads_mix_tp_3class_mean = dict2tensor(grads_mix_tp_3class_mean)
        grads_mix_tp_3class_var = dict2tensor(grads_mix_tp_3class_var)
        
        if source_tp_3_mean_grad_bank.U is not None:
            grads_mix_tp_3class_mean = torch.mm(grads_mix_tp_3class_mean.unsqueeze(0), source_tp_3_mean_grad_bank.U).squeeze()
        if source_tp_3_var_grad_bank.U is not None:
            grads_mix_tp_3class_var = torch.mm(grads_mix_tp_3class_var.unsqueeze(0), source_tp_3_var_grad_bank.U).squeeze()
        
        
        #################################### 
        
        loss_gradient_tp_6class = bce_extended(pred_dict_source_mix["tp"][batch_size:], tp_labels_mix)
        with backpack(BatchGrad()):
            loss_gradient_tp_6class.backward(
                inputs=list(model.tp_layer.parameters()), retain_graph=True, create_graph=True
            )
        
        dict_grads_mix_tp_6class = OrderedDict(
                [
                    (name, weights.grad_batch[batch_size:].clone().squeeze())
                    for name, weights in model.tp_layer.named_parameters()
                ]
            )
        grads_mix_tp_6class_mean = {}
        grads_mix_tp_6class_var = {}
        for name, env_grads in dict_grads_mix_tp_6class.items():
            env_mean = env_grads.mean(dim=0, keepdim=True)
            grads_mix_tp_6class_mean[name] = env_mean.squeeze()
            env_grads_centered = env_grads - env_mean
            grads_mix_tp_6class_var[name] = (env_grads_centered).pow(2).mean(dim=0)
        grads_mix_tp_6class_mean = ema_mix_tp_6class_gradient_mean.update(grads_mix_tp_6class_mean)
        grads_mix_tp_6class_var = ema_mix_tp_6class_gradient_var.update(grads_mix_tp_6class_var)
        grads_mix_tp_6class_mean = grads_mix_tp_6class_mean['weight']
        grads_mix_tp_6class_var = grads_mix_tp_6class_var['weight']
        
        grads_mix_tp_6class_mean_236 = (grads_mix_tp_6class_mean[2] + grads_mix_tp_6class_mean[3] + grads_mix_tp_6class_mean[6]) / 3
        source_tp_6_mean_grad_bank.update(grads_mix_tp_6class_mean_236.detach().cpu().numpy())
        if source_tp_6_mean_grad_bank.U is not None:
            grads_mix_tp_6class_mean_236 = torch.mm(grads_mix_tp_6class_mean_236.unsqueeze(0), source_tp_6_mean_grad_bank.U).squeeze()
        grads_mix_tp_6class_mean_236_avg = grads_mix_tp_6class_mean_236.mean().detach()
        if source_tp_6_mean_grad_bank.U is not None:
            grads_mix_tp_6class_mean_1_avg = torch.mm(grads_mix_tp_6class_mean[1].unsqueeze(0), source_tp_6_mean_grad_bank.U).squeeze().mean()
            grads_mix_tp_6class_mean_4_avg = torch.mm(grads_mix_tp_6class_mean[4].unsqueeze(0), source_tp_6_mean_grad_bank.U).squeeze().mean()
            grads_mix_tp_6class_mean_5_avg = torch.mm(grads_mix_tp_6class_mean[5].unsqueeze(0), source_tp_6_mean_grad_bank.U).squeeze().mean()
        else:
            grads_mix_tp_6class_mean_1_avg = grads_mix_tp_6class_mean[1].mean()
            grads_mix_tp_6class_mean_4_avg = grads_mix_tp_6class_mean[4].mean()
            grads_mix_tp_6class_mean_5_avg = grads_mix_tp_6class_mean[5].mean()
        
        grads_mix_tp_6class_var_236 = (grads_mix_tp_6class_var[2] + grads_mix_tp_6class_var[3] + grads_mix_tp_6class_var[6]) / 3
        source_tp_6_var_grad_bank.update(grads_mix_tp_6class_var_236.detach().cpu().numpy())
        if source_tp_6_var_grad_bank.U is not None:
            grads_mix_tp_6class_var_236 = torch.mm(grads_mix_tp_6class_var_236.unsqueeze(0), source_tp_6_var_grad_bank.U).squeeze()
        grads_mix_tp_6class_var_236_avg = grads_mix_tp_6class_var_236.mean().detach()
        if source_tp_6_var_grad_bank.U is not None:
            grads_mix_tp_6class_var_1_avg = torch.mm(grads_mix_tp_6class_var[1].unsqueeze(0), source_tp_6_var_grad_bank.U).squeeze().mean()
            grads_mix_tp_6class_var_4_avg = torch.mm(grads_mix_tp_6class_var[4].unsqueeze(0), source_tp_6_var_grad_bank.U).squeeze().mean()
            grads_mix_tp_6class_var_5_avg = torch.mm(grads_mix_tp_6class_var[5].unsqueeze(0), source_tp_6_var_grad_bank.U).squeeze().mean()
        else:
            grads_mix_tp_6class_var_1_avg = grads_mix_tp_6class_var[1].mean()
            grads_mix_tp_6class_var_4_avg = grads_mix_tp_6class_var[4].mean()
            grads_mix_tp_6class_var_5_avg = grads_mix_tp_6class_var[5].mean()
    
    ####
    
    pred_dict_target_few = model(imgs_target_few)
    pred_dict_target_few['hv'] = pred_dict_target_few['hv'].permute(0, 2, 3, 1).contiguous()
    pred_dict_target_few['np_feat'] = pred_dict_target_few['np_feat'].permute(0, 2, 3, 1).contiguous()
    pred_dict_target_few['tp_feat'] = pred_dict_target_few['tp_feat'].permute(0, 2, 3, 1).contiguous()
    
    ########################################################################################################
    ####
    optimizer.zero_grad()
    key_list = ["np", "tp", "hv"]
    source_loss_weights = 1.0
    few_loss_weights = 0.1
    mix_loss_weights = 0.1
    mean_var_mseloss_weights = 1000
    var2mean = 1.0
    loss = 0
    loss_opts = run_info["net"]["extra_info"]["loss"]
    
    pred_dict_source.pop("tp_feat")
    pred_dict_source.pop("np_feat")
    #for branch_name in pred_dict_source.keys():
    for branch_name in key_list:
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict_source[branch_name], pred_dict_source[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot_source[..., 1])
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name + '_source', loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss * source_loss_weights
    
    pred_dict_target_few.pop("tp_feat")
    pred_dict_target_few.pop("np_feat")
    #for branch_name in pred_dict_target_few.keys():
    for branch_name in key_list:
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict_target_few[branch_name], pred_dict_target_few[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot_target_few[..., 1])
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name + '_target_few', loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss * few_loss_weights

    
    if epoch_count >= warmup_epoch:
        # np branch
        np_grad_list = [grads_source_np_mean, grads_mix_np_mean, grads_source_np_var, grads_mix_np_var]
        np_gradient_loss = gradient_loss(np_grad_list, true_dict_mix, pred_dict_mix, "np", mix_loss_weights, mean_var_mseloss_weights)
        track_value("loss_%s" % ('np_gradient_loss'), np_gradient_loss.cpu().item())
        loss += np_gradient_loss
    
        # tp overlap class branch
        tp_overlap_grad_list = [grads_source_tp_3class_mean, grads_mix_tp_3class_mean, grads_source_tp_3class_var, grads_mix_tp_3class_var]
        tp_overlap_gradient_loss = gradient_loss(tp_overlap_grad_list, true_dict_mix, pred_dict_mix, "tp_source", mix_loss_weights, mean_var_mseloss_weights) + gradient_loss([], true_dict_target_few, pred_dict_target_few, "tp_source", mix_loss_weights, mean_var_mseloss_weights)
        track_value("loss_%s" % ('tp_overlap_gradient_loss'), tp_overlap_gradient_loss.cpu().item())
        loss += tp_overlap_gradient_loss
        
        # tp new class branch
        tp_new_grad_list = [grads_mix_tp_6class_mean_236_avg, grads_mix_tp_6class_mean_1_avg, grads_mix_tp_6class_mean_4_avg, grads_mix_tp_6class_mean_5_avg, grads_mix_tp_6class_var_236_avg, grads_mix_tp_6class_var_1_avg, grads_mix_tp_6class_var_4_avg, grads_mix_tp_6class_var_5_avg]
        tp_new_gradient_loss = gradient_loss(tp_new_grad_list, true_dict_mix, pred_dict_mix, "tp", mix_loss_weights, mean_var_mseloss_weights)
        track_value("loss_%s" % ('tp_new_gradient_loss'), tp_new_gradient_loss.cpu().item())
        loss += tp_new_gradient_loss
        
    track_value("overall_loss", loss.cpu().item())
    # * gradient update
    
    # torch.set_printoptions(precision=10)
    loss.backward()
    
    if epoch_count >= warmup_epoch:
        for k, (name, params) in enumerate(model.decoder["tp_feat"].named_parameters()):
            #if len(params.shape) > 1 and ('u3.1' in name or 'u2.1' in name):
            if len(params.shape) > 1:
                if step_count % bank_interval == 0:
                    flat_layer_grad_bank_tp.update_singlelayer(name, params.grad.data)
                if flat_layer_grad_bank_tp.layer_bank[name].U is not None:
                    params.grad.data = params.grad.data + 0.1 * (torch.mm(torch.mm(params.grad.data.reshape(-1).unsqueeze(0), flat_layer_grad_bank_tp.layer_bank[name].U), flat_layer_grad_bank_tp.layer_bank[name].U.T).squeeze()).reshape(params.shape)
    
    optimizer.step()
    ####
    
    #sample_indices = torch.randint(0, labels_mix.shape[0], (2,))
    sample_indices = torch.tensor([li for li in range(tp_labels_mix.shape[0])])

    imgs_mix = (imgs_mix[sample_indices]).byte()  # to uint8
    imgs_mix = imgs_mix.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    
    imgs_target_few = (imgs_target_few[sample_indices]).byte()
    imgs_target_few = imgs_target_few.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    
    imgs_target = (imgs_target[sample_indices]).byte()
    imgs_target = imgs_target.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    
    # * Its up to user to define the protocol to process the raw output per step!
    result_dict["raw"] = {  # protocol for contents exchange within `raw`
        "img": imgs_mix,
        "img_target": imgs_target,
        "img_target_few": imgs_target_few,
        "label_target_few": true_tp_target_few[sample_indices].detach().cpu().numpy(),
        "tp": (true_tp_target[sample_indices].detach().cpu().numpy(), tp_labels_mix[sample_indices].detach().cpu().numpy()),
        "np": (true_np_target[sample_indices].detach().cpu().numpy(), np_labels_mix[sample_indices].detach().cpu().numpy()),
        #"hv": (pred_dict_mix["hv"], pred_dict_mix["hv"]),
        "hv": (np_labels_mix[sample_indices].detach().cpu().numpy(), tp_labels_mix[sample_indices].detach().cpu().numpy()),
    }
    
    return result_dict


def valid_step(batch_data_source, batch_data_target, batch_data_target_few, run_info):

    batch_data = batch_data_target
    
    run_info, state_info = run_info
    ####
    model = run_info["net"]["desc"]
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to("cuda").type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
    true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)

    true_dict = {
        "np": true_np,
        "hv": true_hv,
    }

    if model.num_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)
        true_dict["tp"] = true_tp

    # --------------------------------------------------------------
    with torch.inference_mode():  # dont compute gradient
        pred_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        #pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1]
        pred_dict["np"] = torch.argmax(F.softmax(pred_dict["np"], dim=-1), dim=-1)
        if model.num_types is not None:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map

    # * Its up to user to define the protocol to process the raw output per step!
    result_dict = {  # protocol for contents exchange within `raw`
        "raw": {
            "imgs": imgs.numpy(),
            "true_np": true_dict["np"].cpu().numpy(),
            "true_hv": true_dict["hv"].cpu().numpy(),
            "prob_np": pred_dict["np"].cpu().numpy(),
            "pred_hv": pred_dict["hv"].cpu().numpy(),
        }
    }
    if model.num_types is not None:
        result_dict["raw"]["true_tp"] = true_dict["tp"].cpu().numpy()
        result_dict["raw"]["pred_tp"] = pred_dict["tp"].cpu().numpy()
    return result_dict


def infer_step(batch_data, model):

    ####
    patch_imgs = batch_data

    patch_imgs_gpu = patch_imgs.to("cuda").type(torch.float32)  # to NCHW
    patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous()

    ####
    model.eval()  # infer mode

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        pred_dict = model(patch_imgs_gpu)
        pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
        if "tp" in pred_dict:
            type_map = F.softmax(pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            type_map = type_map.type(torch.float32)
            pred_dict["tp"] = type_map
        pred_output = torch.cat(list(pred_dict.values()), -1)

    # * Its up to user to define the protocol to process the raw output per step!
    return pred_output.cpu().numpy()


def viz_step_output(raw_data, num_types=7):
    """
    `raw_data` will be implicitly provided in the similar format as the 
    return dict from train/valid step, but may have been accumulated across N running step
    """

    imgs = raw_data["img"]
    if "img_target" in raw_data.keys():
        imgs_target = raw_data["img_target"]
    else:
        imgs_target = imgs
    if "img_target_few" in raw_data.keys():
        imgs_target_few = raw_data["img_target_few"]
    else:
        imgs_target_few = imgs
    if "label_target_few" in raw_data.keys():
        label_target_few = raw_data["label_target_few"]
    else:
        label_target_few = raw_data["np"][0]
    
    true_np, pred_np = raw_data["np"]
    true_hv, pred_hv = raw_data["hv"]
    if num_types is not None:
        true_tp, pred_tp = raw_data["tp"]

    aligned_shape = [list(imgs.shape), list(true_np.shape), list(pred_np.shape)]
    aligned_shape = np.min(np.array(aligned_shape), axis=0)[1:3]

    cmap = plt.get_cmap("jet")

    def colorize(ch, vmin, vmax):
        """
        Will clamp value value outside the provided range to vmax and vmin
        """
        ch = np.squeeze(ch.astype("float32")) # 128 * 128
        ch[ch > vmax] = vmax  # clamp value
        ch[ch < vmin] = vmin
        #ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
        # take RGB from RGBA heat map
        ch_cmap = np.zeros((ch.shape[0], ch.shape[1], 3))
        ch_cmap[ch == 1] = [255, 165, 0]
        ch_cmap[ch == 2] = [0, 255, 0]
        ch_cmap[ch == 3] = [255, 0, 0]
        ch_cmap[ch == 4] = [130, 130, 255]
        ch_cmap[ch == 5] = [0, 0, 255]
        ch_cmap[ch == 6] = [255, 255, 0]
        ch_cmap = ch_cmap.astype("uint8")
        #ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
        # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
        return ch_cmap

    viz_list = []
    for idx in range(imgs.shape[0]):
        # img = center_pad_to_shape(imgs[idx], aligned_shape)
        img = cropping_center(imgs[idx], aligned_shape)
        img_target = cropping_center(imgs_target[idx], aligned_shape)
        img_target_few = cropping_center(imgs_target_few[idx], aligned_shape)
        

        true_viz_list = [img, img_target, img_target_few]
        true_viz_list.append(colorize(label_target_few[idx], 0, num_types))
        # cmap may randomly fails if of other types
        true_viz_list.append(colorize(true_np[idx], 0, 1))
        #true_viz_list.append(colorize(true_hv[idx][..., 0], -1, 1))
        #true_viz_list.append(colorize(true_hv[idx][..., 1], -1, 1))
        if num_types is not None:  # TODO: a way to pass through external info
            true_viz_list.append(colorize(true_tp[idx], 0, num_types))
        true_viz_list = np.concatenate(true_viz_list, axis=1)

        pred_viz_list = [img, img_target, img_target_few]
        pred_viz_list.append(colorize(label_target_few[idx], 0, num_types))
        # cmap may randomly fails if of other types
        pred_viz_list.append(colorize(pred_np[idx], 0, 1))
        #pred_viz_list.append(colorize(pred_hv[idx][..., 0], -1, 1))
        #pred_viz_list.append(colorize(pred_hv[idx][..., 1], -1, 1))
        if num_types is not None:
            pred_viz_list.append(colorize(pred_tp[idx], 0, num_types))
        pred_viz_list = np.concatenate(pred_viz_list, axis=1)

        viz_list.append(np.concatenate([true_viz_list, pred_viz_list], axis=0))
    viz_list = np.concatenate(viz_list, axis=0)
    return viz_list


def proc_valid_step_output(raw_data, num_types=None):
    # TODO: add auto populate from main state track list
    track_dict = {"scalar": {}, "image": {}}

    def track_value(name, value, vtype):
        return track_dict[vtype].update({name: value})

    def _dice_info(true, pred, label):
        true = np.array(true == label, np.int32)
        pred = np.array(pred == label, np.int32)
        inter = (pred * true).sum()
        total = (pred + true).sum()
        return inter, total

    over_inter = 0
    over_total = 0
    over_correct = 0
    prob_np = raw_data["prob_np"]
    true_np = raw_data["true_np"]
    for idx in range(len(raw_data["true_np"])):
        patch_prob_np = prob_np[idx]
        patch_true_np = true_np[idx]
        patch_pred_np = np.array(patch_prob_np > 0.5, dtype=np.int32)
        inter, total = _dice_info(patch_true_np, patch_pred_np, 1)
        correct = (patch_pred_np == patch_true_np).sum()
        over_inter += inter
        over_total += total
        over_correct += correct
    nr_pixels = len(true_np) * np.size(true_np[0])
    acc_np = over_correct / nr_pixels
    dice_np = 2 * over_inter / (over_total + 1.0e-8)
    track_value("np_acc", acc_np, "scalar")
    track_value("np_dice", dice_np, "scalar")

    # * TP statistic
    if num_types is not None:
        pred_tp = raw_data["pred_tp"]
        true_tp = raw_data["true_tp"]
        for type_id in range(0, num_types):
            over_inter = 0
            over_total = 0
            for idx in range(len(raw_data["true_np"])):
                patch_pred_tp = pred_tp[idx]
                patch_true_tp = true_tp[idx]
                inter, total = _dice_info(patch_true_tp, patch_pred_tp, type_id)
                over_inter += inter
                over_total += total
            dice_tp = 2 * over_inter / (over_total + 1.0e-8)
            track_value("tp_dice_%d" % type_id, dice_tp, "scalar")

    # * HV regression statistic
    pred_hv = raw_data["pred_hv"]
    true_hv = raw_data["true_hv"]

    over_squared_error = 0
    for idx in range(len(raw_data["true_np"])):
        patch_pred_hv = pred_hv[idx]
        patch_true_hv = true_hv[idx]
        squared_error = patch_pred_hv - patch_true_hv
        squared_error = squared_error * squared_error
        over_squared_error += squared_error.sum()
    mse = over_squared_error / nr_pixels
    track_value("hv_mse", mse, "scalar")

    # *
    imgs = raw_data["imgs"]
    selected_idx = np.random.randint(0, len(imgs), size=(8,)).tolist()
    imgs = np.array([imgs[idx] for idx in selected_idx])
    true_np = np.array([true_np[idx] for idx in selected_idx])
    true_hv = np.array([true_hv[idx] for idx in selected_idx])
    prob_np = np.array([prob_np[idx] for idx in selected_idx])
    pred_hv = np.array([pred_hv[idx] for idx in selected_idx])
    viz_raw_data = {"img": imgs, "np": (true_np, prob_np), "hv": (true_hv, pred_hv)}

    if num_types is not None:
        true_tp = np.array([true_tp[idx] for idx in selected_idx])
        pred_tp = np.array([pred_tp[idx] for idx in selected_idx])
        viz_raw_data["tp"] = (true_tp, pred_tp)
    viz_fig = viz_step_output(viz_raw_data, num_types)
    track_dict["image"]["output"] = viz_fig

    return track_dict
