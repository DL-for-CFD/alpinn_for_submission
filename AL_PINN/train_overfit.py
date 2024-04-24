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
from derivatives import dx, dy, dx_left, dy_top, dx_right, dy_bottom, laplace, map_vx2vy_left, map_vy2vx_top, map_vx2vy_right, map_vy2vx_bottom, normal2staggered, toCuda, toCpu, params
from derivatives import rot_mac
import tqdm
import logging
from derivatives import params
import sys
import networks
from pde_cnn import get_Net

sys.path.append('./RL_agent_pix2pix_gan/models')

fixed_seed = 4211
torch.manual_seed(fixed_seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(fixed_seed)
np.random.seed(fixed_seed)
random.seed(fixed_seed)

torch.set_num_threads(4)

mu = params.mu
rho = params.rho
dt = params.dt

time_frames = params.n_time_frames

num_epochs = 5000
lrS = params.lrS
lrG = params.lrG
beta1 = 0.5
last_frames = params.last_frames
n_init_problems = 1
time_frames_threshold = params.time_frames_threshold

# set logging dir
tb_dir = '/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/Logger/tensorboard'
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
os.chdir(
    tb_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='train_cgan.log')
logger = logging.getLogger(__name__)

# Using GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")


def get_name(path: str):
    '''input file path. return (fname, fext).'''
    return os.path.splitext(os.path.basename(os.path.normpath(path)))


# def initialize_G():
#     netG = networks.define_G(64,
#                              3,
#                              64,
#                              'resnet_autodecoder',
#                              'batch',
#                              False,
#                              'normal',
#                              0.02,
#                              gpu_ids=[0])
#     netG.module.decoder.load_state_dict(
#         torch.load(
#             "/csproject/t3_lzengaf/lzengaf/fyp/resnet_autodecoder/autoencoder/circle_line_decoder_300_100_clear.pth"
#         ))
#     optimizerG = optim.Adam([
#         {
#             'params': netG.module.encoder.parameters(),
#             'lr': lrG,
#             'betas': (beta1, 0.999)
#         },  # Parameters to optimize with a specific learning rate
#         {
#             'params': netG.module.adapter.parameters(),
#             'lr': lrG,
#             'betas': (beta1, 0.999)
#         },
#         {
#             'params': netG.module.decoder.parameters(),
#             'lr': 0,
#             'betas': (beta1, 0.999)
#         }  # Parameters to freeze (learning rate set to 0.0)
#     ])
#     # netG.parameters(), lr=lrG, betas=(beta1, 0.999))
#     netG.train()
#     return netG, optimizerG


def renew_problems(problem_set):
    for problem in problem_set:
        problem[1] = torch.zeros(1, 1, 100, 300).to('cuda')
        problem[2] = torch.zeros(1, 1, 100, 300).to('cuda')
        problem[3] = 0


def initial_problems(dir):
    problem_set = []
    G_output = toCuda(torch.from_numpy(np.load(dir)))

    if len(G_output.shape) == 3:
        G_output = G_output.unsqueeze(0)
    grid = vutils.make_grid(G_output[:, 0],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('Generated Images 0', grid)
    grid = vutils.make_grid(G_output[:, 1],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('Generated Images 1', grid)
    grid = vutils.make_grid(G_output[:, 2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('Generated Images 2', grid)
    problem_set.append([
        G_output,
        torch.zeros(1, 1, 100, 300).to('cuda'),
        torch.zeros(1, 1, 100, 300).to('cuda'), 0
    ])  # Problem, a, p
    return problem_set


def ask(problem_set):
    i = random.randint(0, len(problem_set) - 1)
    return problem_set[i][0].clone(), problem_set[i][1].clone(
    ), problem_set[i][2].clone(), i


def tell(a_new, p_new, problem_set, i):
    problem_set[i][1] = a_new
    problem_set[i][2] = p_new
    problem_set[i][3] += 1
    new_problem = True
    for problem in problem_set[-n_init_problems:]:
        if problem[3] <= time_frames_threshold:
            new_problem = False
    return new_problem


#netG, optimizerG = initialize_G()

netS = toCuda(get_Net(params))

# Load the pretrained netG
# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerS = optim.Adam(netS.parameters(), lr=lrS)

# Using tensorboard
from torch.utils.tensorboard import SummaryWriter

fname = get_name(params.qst)[0]
writer = SummaryWriter(log_dir=f'{fname}/runs')

real_label = torch.Tensor([1.]).to(device)
fake_label = torch.Tensor([0.]).to(device)

netS.train()
torch.autograd.set_detect_anomaly(True)



def loss_function(x):
    return torch.pow(x, 2)


def loss_PINN(v_cond, v_old, a_new, p_new, v_new, cond_mask_mac,
              flow_mask_mac):
    # compute boundary loss
    loss_bound = torch.mean(loss_function(cond_mask_mac *
                                          (v_new - v_cond))[:, :, 1:-1, 1:-1],
                            dim=(1, 2, 3))

    # explicit / implicit / IMEX integration schemes
    if params.integrator == "explicit":
        v = v_old
    if params.integrator == "implicit":
        v = v_new
    if params.integrator == "imex":
        v = (v_new + v_old) / 2

    # compute loss for momentum equation
    loss_nav =  torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:,1:2]-v_old[:,1:2])/dt+v[:,1:2]*dx(v[:,1:2])+0.5*(map_vy2vx_top(v[:,0:1])*dy_top(v[:,1:2])+map_vy2vx_bottom(v[:,0:1])*dy_bottom(v[:,1:2])))+dx_left(p_new)-mu*laplace(v[:,1:2])))[:,:,1:-1,1:-1],dim=(1,2,3))+\
                torch.mean(loss_function(flow_mask_mac*(rho*((v_new[:,0:1]-v_old[:,0:1])/dt+v[:,0:1]*dy(v[:,0:1])+0.5*(map_vx2vy_left(v[:,1:2])*dx_left(v[:,0:1])+map_vx2vy_right(v[:,1:2])*dx_right(v[:,0:1])))+dy_top(p_new)-mu*laplace(v[:,0:1])))[:,:,1:-1,1:-1],dim=(1,2,3))

    regularize_grad_p = torch.mean(
        (dx_right(p_new)**2 + dy_bottom(p_new)**2)[:, :, 2:-2, 2:-2],
        dim=(1, 2, 3))

    # optional: additional loss to keep mean of a / p close to 0
    loss_mean_a = torch.mean(a_new, dim=(1, 2, 3))**2
    loss_mean_p = torch.mean(p_new, dim=(1, 2, 3))**2

    loss = params.loss_bound * loss_bound + params.loss_nav * loss_nav + params.loss_mean_a * loss_mean_a + params.loss_mean_p * loss_mean_p + params.regularize_grad_p * regularize_grad_p

    loss = torch.mean(torch.log(loss))
    return loss, params.loss_bound * loss_bound, params.loss_nav * loss_nav, params.loss_mean_a * loss_mean_a, params.loss_mean_p * loss_mean_p, params.regularize_grad_p * regularize_grad_p


num_iter = 0
problem_set = initial_problems(params.qst)

res_dict = {"a":[], "p": [], "v": []}
for epoch in tqdm.tqdm(range(num_epochs)):

    if epoch % 32 == 0:
        renew_problems(problem_set)
    # initialize the problem
    G_output, a_old, p_old, problem_index = ask(problem_set)
    G_output = torchvision.transforms.functional.resize(
        G_output, (100, 100 * 3)).detach()
    cond_mask = G_output[:, 0:1]
    flow_mask = 1 - cond_mask
    v_cond = G_output[:, 1:]
    v_cond = torch.cat([v_cond[:, 1:], v_cond[:, 0:1]],
                        dim=1)  # switch vx,vy to vy,vx, yeah~~~
    v_cond = normal2staggered(v_cond)
    cond_mask_mac = (normal2staggered(cond_mask.repeat(1, 2, 1,
                                                        1)) == 1).float()
    flow_mask_mac = (normal2staggered(flow_mask.repeat(1, 2, 1, 1)) >=
                        0.5).float()
    netS.zero_grad()

    # convert v_cond,cond_mask,flow_mask to MAC grid
    v_old = rot_mac(a_old)

    # predict new fluid state from old fluid state and boundary conditions using the neural fluid model
    a_new, p_new = netS(a_old, p_old, flow_mask, v_cond, cond_mask)

    v_new = rot_mac(a_new)

    loss, bound, nav, mean_a, mean_p, grad_p = loss_PINN(
        v_cond, v_old, a_new, p_new, v_new, cond_mask_mac, flow_mask_mac)
    num_iter += 1

    writer.add_scalar('Loss/train netS', loss.item(), num_iter)
    writer.add_scalar('bound', bound, num_iter)
    writer.add_scalar('nav', nav, num_iter)
    writer.add_scalar('v_cond', v_cond[0, 1, 50, 0], num_iter)

    # compute gradients
    loss = loss * params.loss_multiplier  # ignore the loss_multiplier (could be used to scale gradients)
    loss.backward()

    # optional: clip gradients
    if params.clip_grad_value is not None:
        torch.nn.utils.clip_grad_value_(netS.parameters(),
                                        params.clip_grad_value)
    if params.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(netS.parameters(),
                                        params.clip_grad_norm)

    # perform optimization step
    optimizerS.step()
    # SNet output
    grid = vutils.make_grid(a_new[:, 0],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('a output 0', grid, num_iter)
    grid = vutils.make_grid(v_new[:, 0, :, :-2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('v output 0', grid, num_iter)
    grid = vutils.make_grid(v_new[:, 1, :-2, :],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('v output 1', grid, num_iter)
    grid = vutils.make_grid(p_new[:, 0],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('p output 0', grid, num_iter)

    grid = vutils.make_grid(v_cond[:, 0, :, :-2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('vcond output 0', grid, num_iter)
    grid = vutils.make_grid(v_cond[:, 1, :-2, :],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('vcond output 1', grid, num_iter)
    grid = vutils.make_grid(v_old[:, 0, :, :-2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('vold output 0', grid, num_iter)
    grid = vutils.make_grid(v_old[:, 1, :-2, :],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('vold output 1', grid, num_iter)
    grid = vutils.make_grid(cond_mask_mac[:, 0, :, :-2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('cond_mask_mac output 0', grid, num_iter)
    grid = vutils.make_grid(cond_mask_mac[:, 1, :-2, :],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('cond_mask_mac output 1', grid, num_iter)
    grid = vutils.make_grid(flow_mask_mac[:, 0, :, :-2],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('flow_mask_mac output 0', grid, num_iter)
    grid = vutils.make_grid(flow_mask_mac[:, 1, :-2, :],
                            nrow=8,
                            normalize=True,
                            scale_each=True)
    writer.add_image('flow_mask_mac output 1', grid, num_iter)
    p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
    a_new.data = (a_new.data-torch.mean(a_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize a
    _ = tell(a_new.clone().detach(),
                p_new.clone().detach(), problem_set, problem_index)
    
    if epoch >= num_epochs-32:
        res_dict["a"].append(a_new)
        res_dict["p"].append(p_new)
        res_dict["v"].append(v_new)
    


# if not os.path.exists('../G_output_log'):
#     os.makedirs('../G_output_log')
# if not os.path.exists('../a_log'):
#     os.makedirs('../a_log')
# if not os.path.exists('../v_log'):
#     os.makedirs('../v_log')
# if not os.path.exists('../p_log/'):
#     os.makedirs('../p_log')
if not os.path.exists('../net/'):
    os.makedirs('../net')

# torch.save(G_output, f'../G_output_log/G_output_{fname}_{epoch}.pth')
# torch.save(a_new, f'../a_log/a_new_{fname}_{epoch}.pth')
# torch.save(v_new, f'../v_log/v_new_{fname}_{epoch}.pth')
# torch.save(p_new, f'../p_log/p_new_{fname}_{epoch}.pth')
# Save model
# torch.save(netG.state_dict(), f'../net/netG_{fname}.pth')
torch.save(netS.state_dict(), f'../net/netS_{fname}.pth')
torch.save(res_dict, f'../net/res_dict_{fname}.pth')
