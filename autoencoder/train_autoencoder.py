import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
from derivatives import rot_mac, normal2staggered,toCuda,params
import tqdm
import logging
from derivatives import params
from pde_cnn import get_Net
from utils import *

# set logging dir
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(output_dir, 'train_cgan.log'))
logger = logging.getLogger(__name__)

# Decide which device we want to run on

netG, optimizerG = initialize_G()

netS = toCuda(get_Net(params))
# Load the pretrained netG
# Initialize BCELoss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerS = optim.Adam(netS.parameters(),lr=lrS)

# Using tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(output_dir, 'runs'))

real_label = torch.Tensor([1.]).to(device)
fake_label = torch.Tensor([0.]).to(device)


netS.train()
torch.autograd.set_detect_anomaly(True)

num_iter = 0
problem_set = initial_problems(netG, writer)
for epoch in tqdm.tqdm(range(num_epochs)):
    netG, optimizerG = initialize_G()
    renew_problems(problem_set)
    for i in tqdm.tqdm(range(params.n_batches_per_epoch)):
        # Update S network:
        # Train with all-real batch
    # initialize the problem
    ## NetG input shape?
        G_output, a_old, p_old, problem_index = ask(problem_set, epoch<190)
        G_output = torchvision.transforms.functional.resize(G_output, (100, 100*3)).detach()
        cond_mask = G_output[:,0:1]
        flow_mask = 1-cond_mask
        v_cond = G_output[:,1:]
        v_cond = torch.cat([v_cond[:,1:],v_cond[:,0:1]], dim=1) # switch vx,vy to vy,vx, yeah~~~
        v_cond = normal2staggered(v_cond)
        cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))>=0.99).float()
        flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()
        netS.zero_grad()
        
        # convert v_cond,cond_mask,flow_mask to MAC grid
        v_old = rot_mac(a_old)
        
        # predict new fluid state from old fluid state and boundary conditions using the neural fluid model
        a_new,p_new = netS(a_old,p_old,flow_mask,v_cond,cond_mask)

        v_new = rot_mac(a_new)
        loss, bound, nav, mean_a, mean_p, grad_p = loss_PINN(v_cond,v_old,a_new,p_new,v_new, cond_mask_mac, flow_mask_mac)
        num_iter += 1

        writer.add_scalar('Loss/train netS', loss.item(), num_iter)
        writer.add_scalar('bound', bound.item(), num_iter)
        writer.add_scalar('nav', nav.item(), num_iter)
        writer.add_scalar('v_cond', v_cond[0, 1, 50, 0].item(), num_iter)
        
        # compute gradients
        
        loss = loss*params.loss_multiplier # ignore the loss_multiplier (could be used to scale gradients)
        loss.backward()
        
        # optional: clip gradients
        if params.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(netS.parameters(),params.clip_grad_value)
        if params.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(netS.parameters(),params.clip_grad_norm)
        
        # perform optimization step
        optimizerS.step()
        # SNet output
        p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
        a_new.data = (a_new.data-torch.mean(a_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize a
        update_G = tell(a_new.clone().detach(), p_new.clone().detach(), problem_set, problem_index)
        
        if update_G and epoch < 190:
            problems = problem_set[-n_init_problems:]
            for _ in tqdm.tqdm(range(epochG)):
                netS.zero_grad()
                netG.zero_grad()

                total_loss = 0
                G_inputs = []
                for problem in problems:
                    p_list = []
                    v_list = []
                    loss_list = []
                    a_old = torch.zeros(1,1,100,300).to('cuda')
                    p_old = torch.zeros(1,1,100,300).to('cuda')

                    G_output = problem[0]
                    G_output = torchvision.transforms.functional.resize(G_output, (100, 100*3))
                    cond_mask = G_output[:,0:1]
                    flow_mask = 1-cond_mask
                    v_cond = G_output[:,1:]
                    v_cond = torch.cat([v_cond[:,1:],v_cond[:,0:1]], dim=1) # switch vx,vy to vy,vx, yeah~~~
                    v_cond = normal2staggered(v_cond)
                    cond_mask_mac = (normal2staggered(cond_mask.repeat(1,2,1,1))>=0.99).float()
                    flow_mask_mac = (normal2staggered(flow_mask.repeat(1,2,1,1))>=0.5).float()
                    for j in range(time_frames):

                        v_old = rot_mac(a_old)
                        a_new, p_new = netS(a_old,p_old,flow_mask,v_cond,cond_mask) # Shape: (1,1,100,300)
                        v_new = rot_mac(a_new)
                        # store the time into a list
                        if j >= time_frames - last_frames:
                            losss, bound, nav, mean_a, mean_p, grad_p = loss_PINN(v_cond,v_old,a_new,p_new,v_new, cond_mask_mac, flow_mask_mac)
                            bound, nav, mean_a, mean_p, grad_p = 0,0,0,0,0
                            loss_list.append(losss)
                            v_list.append(v_new.clone().detach())
                            p_list.append(p_new.clone().detach())
                            # update a_old and p_old
                        p_new.data = (p_new.data-torch.mean(p_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize pressure
                        a_new.data = (a_new.data-torch.mean(a_new.data,dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3))#normalize a
                        a_old = a_new.detach()
                        p_old = p_new.detach()
                    G_input_v = v_list
                    G_input_p = p_list
                    G_input = torch.cat([torch.cat(G_input_v, dim=0), torch.cat(G_input_p, dim=0)], dim=1).detach()
                    G_inputs.append(G_input)
                    errG = torch.mean(torch.stack(loss_list))

                    x = torch.sum(cond_mask[:,:,3:-3,5:-5]) / ((100-6)*(300-10))

                    area_loss = torch.relu(-100*(x-min_obj_thre))+torch.relu(-10*(max_obj_thre-x))
                    total_loss += area_loss
                    total_loss += - errG
                # Calculate gradients for G
                total_loss /= n_init_problems
                total_loss.backward()
                writer.add_scalar('Loss/train netG', total_loss.item(), epoch * params.n_batches_per_epoch + i)
                grad_norm = torch.nn.utils.clip_grad_norm_(netG.parameters(),params.clip_grad_norm)
                # Update G
                optimizerG.step()
                problems = update_problems(netG, G_inputs)
            add_problems(problem_set, netG, G_inputs)
                # Calculate G's loss based on this output
            
    # Save model
    torch.save(netS.state_dict(), os.path.join(output_dir, 'netS%d.pth'%epoch))