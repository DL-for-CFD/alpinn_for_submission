import sys
sys.path.append('./generator/models')
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from utils import *
import torchvision.utils as vutils
import networks
from derivatives import rot_mac, dx,dy,dx_left,dy_top,dx_right,dy_bottom,laplace,map_vx2vy_left,map_vy2vx_top,map_vx2vy_right,map_vy2vx_bottom,params
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set parameters
mu = params.mu
rho = params.rho
dt = params.dt

time_frames = params.n_time_frames

num_epochs = 200
lrS = params.lrS
lrG = params.lrG
beta1 = 0.5
last_frames = params.last_frames
n_init_problems = params.n_init_problems
time_frames_threshold = params.time_frames_threshold
epochG = params.epochG
max_obj_thre = params.max_obj_thre
min_obj_thre = params.min_obj_thre
problem_pool_size = params.problem_pool_size
renewal_pool_size = params.renewal_pool_size
output_dir = params.output_dir

def initialize_G():
    netG = networks.define_G(64, 3, 64, 'FullyConvAE', 'batch', False, 'normal', 0.02, gpu_ids=[0])
    netG.module.decoder.load_state_dict(torch.load("./pretrain_weights/circle_line_decoder_300_100_clear.pth"))
    optimizerG = optim.Adam([
    {'params': netG.module.encoder.parameters(), 'lr': lrG, 'betas': (beta1, 0.999)},  # Parameters to optimize with a specific learning rate
    {'params': netG.module.adapter.parameters(), 'lr': lrG, 'betas': (beta1, 0.999)},  
    {'params': netG.module.decoder.parameters(), 'lr': 0, 'betas': (beta1, 0.999)}  # Parameters to freeze (learning rate set to 0.0)
    ])
    # netG.parameters(), lr=lrG, betas=(beta1, 0.999))
    netG.train()
    return netG, optimizerG

problem_num = 0

def initial_problems(netG, writer):
    global problem_num
    problem_set = []
    for _ in range(n_init_problems):
        G_input_dir = random.choice(os.listdir('./pretrain_weights/mu_0.5_rho_1.0'))
        G_input = torch.from_numpy(np.load(os.path.join('./pretrain_weights/mu_0.5_rho_1.0', G_input_dir))).to(device)
        G_input = G_input.permute(1, 0, 2, 3)
        # Initialize an empty list to store the processed channels
        processed_channels = []

        for channel in G_input[0]:
            channel = channel.unsqueeze(0).unsqueeze(0)
            processed_channel = rot_mac(channel)
            processed_channels.append(processed_channel)

        v_input = torch.cat(processed_channels, dim=0)
        v_input = v_input.permute(1, 0, 2, 3)
        p_input = G_input[1].unsqueeze(0)

        G_input = torch.cat([v_input, p_input], dim=0)
        G_input = G_input.permute(1, 0, 2, 3)
        G_output = netG(G_input)
        grid = vutils.make_grid(G_output[:, 0], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 0', grid, problem_num)
        grid = vutils.make_grid(G_output[:, 1], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 1', grid, problem_num)
        grid = vutils.make_grid(G_output[:, 2], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 2', grid, problem_num)
        problem_num+=1
        problem_set.append([G_output, torch.zeros(1,1,100,300).to('cuda'), torch.zeros(1,1,100,300).to('cuda'), 0]) # Problem, a, p
    return problem_set

def ask(problem_set, biased = True):
    if random.random() > 0.1 and biased:
        i = random.randint(0, len(problem_set)-1)
    else:
        i = random.randint(len(problem_set)-n_init_problems, len(problem_set)-1)
    return problem_set[i][0].clone(), problem_set[i][1].clone(), problem_set[i][2].clone(), i
    

def tell(a_new, p_new, problem_set, i):
    problem_set[i][1]=a_new
    problem_set[i][2]=p_new
    problem_set[i][3]+=1
    new_problem = True
    for problem in problem_set[-n_init_problems:]:
        if problem[3] <= time_frames_threshold:
            new_problem = False
    return new_problem

def add_problems(problem_set, netG, G_inputs):
    global problem_num
    for i in range(len(problem_set)):
        problem_set[i][0] = problem_set[i][0].detach()
    for i in range(n_init_problems):
        # GNet output
        G_output = netG(G_inputs[i].detach())
        grid = vutils.make_grid(G_output[:, 0], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 0', grid, problem_num)
        grid = vutils.make_grid(G_output[:, 1], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 1', grid, problem_num)
        grid = vutils.make_grid(G_output[:, 2], nrow=8, normalize=True, scale_each=True)
        writer.add_image('Generated Images 2', grid, problem_num)
        problem_num+=1
        problem_set.append([G_output, torch.zeros(1,1,100,300).to('cuda'), torch.zeros(1,1,100,300).to('cuda'), 0]) # Problem, a, p with shape
    while len(problem_set)>problem_pool_size:
        problem_set.pop(random.randint(0,renewal_pool_size - 1))

def update_problems(netG, G_inputs):
    problems = []
    for i in range(n_init_problems):
        # GNet output
        G_output = netG(G_inputs[i].detach())
        problems.append([G_output, torch.zeros(1,1,100,300).to('cuda'), torch.zeros(1,1,100,300).to('cuda'), 0]) # Problem, a, p with shape
    return problems

def renew_problems(problem_set):
    for problem in problem_set:
        if random.random() < 0.8:
            continue
        problem[1] = torch.zeros(1,1,100,300).to('cuda')
        problem[2] = torch.zeros(1,1,100,300).to('cuda')
        problem[3] = 0

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
