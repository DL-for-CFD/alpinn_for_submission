# Given input dataset and PINN solver weight, get and save inference results

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from derivatives import dx,dy,dx_left,dy_top,dx_right,dy_bottom,laplace,map_vx2vy_left,map_vy2vx_top,map_vx2vy_right,map_vy2vx_bottom,normal2staggered,toCuda,toCpu,params
from derivatives import vector2HSV,rot_mac, params
from Logger import Logger,t_step
import tqdm
import logging

from pde_cnn import get_Net
from inference_dataloader import get_validation_dataloader

mu = params.mu
rho = params.rho
dt = params.dt

time_frames = 64
batch_size = 4

dataset = "airfoil"
output_dir = "/csproject/t3_rliuak/rliuak/resnet_autodecoder_for_submission/autoencoder/test/" + dataset
output_dir_v = "/csproject/t3_rliuak/rliuak/resnet_autodecoder_for_submission/autoencoder/test_v/" + dataset

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_v):
    os.makedirs(output_dir_v)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create an instance of the netS model
# Load .pth
netS = toCuda(get_Net(params))
netS.load_state_dict(torch.load("/csproject/t3_rliuak/rliuak/resnet_autodecoder/autoencoder/param_search_log/test_3_r/netS99.pth"))
netS.eval()
# Load .state
'''
netS = toCuda(get_Net(params))
state_dict = torch.load('/project/t3_zxiaoal/resnet_autoencoder/autoencoder/baseline_states/99.state')
netS.load_state_dict(state_dict['model0'])
netS.eval()
'''
def loss_function(x):
    return torch.pow(x,2)

def loss_PINN(v_cond, v_old, a_new, p_new, v_new, cond_mask_mac, flow_mask_mac):
    # compute boundary loss
    loss_bound = torch.mean(loss_function(cond_mask_mac*(v_new-v_cond))[:,:,1:-1,1:-1],dim=(1,2,3))
    
    # explicit / implicit / IMEX integration schemes
    if params.integrator == "explicit":
        v = v_old
    if params.integrator == "implicit":
        v = v_new
    if params.integrator == "imex":
        v = (v_new+v_old)/2
        
    # compute loss for momentum equation
    loss_nav =  torch.mean(loss_function(flow_mask_mac[:,1:2]*(rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace(v[:,1:2])))[:,:,1:-1,1:-1],dim=(1,2,3))+\
                torch.mean(loss_function(flow_mask_mac[:,0:1]*(rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace(v[:,0:1])))[:,:,1:-1,1:-1],dim=(1,2,3))
        
    regularize_grad_p = torch.mean((dx_right(p_new)**2+dy_bottom(p_new)**2)[:,:,2:-2,2:-2],dim=(1,2,3))
        
    # optional: additional loss to keep mean of a / p close to 0
    loss_mean_a = torch.mean(a_new,dim=(1,2,3))**2
    loss_mean_p = torch.mean(p_new,dim=(1,2,3))**2
        
    loss = params.loss_bound*loss_bound + params.loss_nav*loss_nav + params.loss_mean_a*loss_mean_a + params.loss_mean_p*loss_mean_p + params.regularize_grad_p*regularize_grad_p
        
    loss = torch.mean(torch.log(1e-20+loss))
    return loss, params.loss_bound*loss_bound, params.loss_nav*loss_nav, params.loss_mean_a*loss_mean_a, params.loss_mean_p*loss_mean_p, params.regularize_grad_p*regularize_grad_p

errG = []

with torch.no_grad():
    for files, file_names in get_validation_dataloader('/project/t3_zxiaoal/Validation_Dataset/validationdataset/' + dataset):
        files = files.to(device)
        cond_mask = files[:,0:1]
        flow_mask = 1-cond_mask
        v_cond = files[:,1:]
        a_old = torch.zeros_like(cond_mask)
        p_old = torch.zeros_like(cond_mask)
        # convert v_cond,cond_mask,flow_mask to MAC grid
        v_cond = torch.cat([v_cond[:,1:],v_cond[:,0:1]], dim=1)
        v_cond = normal2staggered(v_cond)
        cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))>=0.99).float()
        flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()
        outputs_a = []
        outputs_p = []
        outputs_v = []
        loss_list = []
        for i in range(time_frames):
            v_old = rot_mac(a_old)
            
            # predict new fluid state from old fluid state and boundary conditions using the neural fluid model
            a_new, p_new = netS(a_old,p_old,flow_mask,v_cond,cond_mask)

            v_new = rot_mac(a_new)

            p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
            a_new.data = (a_new.data-torch.mean(a_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize a

            if i>= time_frames - 32:
                losss, _, _, _, _, _= loss_PINN(v_cond,v_old,a_new,p_new,v_new, cond_mask_mac, flow_mask_mac)
                loss_list.append(losss)

            # update the previous time step
            a_old = a_new.clone().detach()
            p_old = p_new.clone().detach()
            # print(a_old, p_old)
    
            if i>= time_frames - 32: 
                outputs_a.append(a_new)
                outputs_p.append(p_new)
                outputs_v.append(v_new)
                
        outputs_a_stacked = torch.stack(outputs_a, dim=1)
        outputs_p_stacked = torch.stack(outputs_p, dim=1)
        outputs = torch.cat([outputs_a_stacked, outputs_p_stacked], dim=2)
        outputs = torch.split(outputs, 1)
        
        errG.append(torch.mean(torch.stack(loss_list)))
        
        for j, output in enumerate(outputs):
            # print(j, file_names[j])
            output_file_name = file_names[j].replace(".npy", ".npy")
            output_file_path = os.path.join(output_dir, output_file_name)
            np.save(output_file_path, output.squeeze(0).cpu().numpy())

        outputs_v = torch.stack(outputs_v)
        outputs_v = outputs_v.permute(1, 0, 2, 3, 4)
        outputs_v = torch.split(outputs_v, 1)
        for j, output in enumerate(outputs_v):
            output_file_name = file_names[j].replace(".npy", ".npy")
            output_file_path = os.path.join(output_dir_v, output_file_name)
            np.save(output_file_path, output.squeeze(0).cpu().numpy())
        
print("Loss", torch.mean(torch.stack(errG)))