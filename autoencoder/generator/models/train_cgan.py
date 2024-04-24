import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import tqdm
import logging
from torch.optim.lr_scheduler import MultiStepLR

torch.manual_seed(58)
torch.backends.cudnn.deterministic=True
torch.cuda.manual_seed(58)
np.random.seed(58)
random.seed(58)


# set logging dir
training_dir = '/localdata/rzhangbq/RL_agent_pix2pix_gan/training_data' #'/localdata/rzhangbq/RL_agent_pix2pix_gan/training_data/pix2pix_gan_seed_e120_58'
os.makedirs(training_dir, exist_ok=True)
os.chdir(training_dir)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(training_dir, 'train_cgan.log'))
logger = logging.getLogger(__name__)


from dataset import get_netG_dataloader, get_netD_dataloader
import networks

# Using GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
         
netD = networks.define_D(3, 64, 'basic', 3, 'batch', 'normal', 0.02, gpu_ids=[0])
netG = networks.define_G(64, 3, 64, 'unet_64', 'batch', False, 'normal', 0.02, gpu_ids=[0])

logger.info(netD)
logger.info(netG)

# Get dataloader
netG_dataloader = get_netG_dataloader()
netD_dataloader = get_netD_dataloader()

# Initialize BCELoss function
criterion = nn.BCELoss()

num_epochs = 120
lrD = 0.00002
lrG = 0.0002
beta1 = 0.5

logger.info('num_epochs: %d\tlrD: %f\tlrG: %f\tbeta1: %f'%(num_epochs, lrD, lrG, beta1))
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
schedulerD = MultiStepLR(optimizerD, milestones=[60, 80, 100], gamma=0.2)
schedulerG = MultiStepLR(optimizerG, milestones=[60, 80, 100], gamma=0.2)
logger.info('milestones: %s\tgamma: %f'%(str(schedulerD.milestones), schedulerD.gamma))


# Using tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs')

real_label = torch.Tensor([1.]).to(device)
fake_label = torch.Tensor([0.]).to(device)

netG.train()
netD.train()
torch.autograd.set_detect_anomaly(True)
for epoch in tqdm.tqdm(range(num_epochs)):
    for i, dataD in tqdm.tqdm(enumerate(netD_dataloader, 0), total=len(netD_dataloader)):
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = dataD.to(device)
        b_size_D = real_cpu.size(0)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1).sigmoid()
        # Calculate loss on all-real batch
        label_D = real_label.expand_as(output)
        errD_real = criterion(output, label_D)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        dataG = next(iter(netG_dataloader))
        noise = dataG.to(device)
        b_size_G = noise.size(0)
        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1).sigmoid()
        # Calculate D's loss on the all-fake batch
        label_G = fake_label.expand_as(output)
        errD_fake = criterion(output, label_G)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # Update G network: maximize log(D(G(z)))
        netG.zero_grad()
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1).sigmoid()
        label_G = real_label.expand_as(output)  # fake labels are real for generator cost
        # Calculate G's loss based on this output
        errG = criterion(output, label_G)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()


        # Output training stats
        if i % 5 == 0:
            logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z1)): %.4f\tD(G(z2)): %.4f'
                % (epoch, num_epochs, i, len(netD_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
        # Save Losses for plotting later
        writer.add_scalar('Loss_D', errD.item(), epoch*len(netD_dataloader)+i)
        writer.add_scalar('Loss_G', errG.item(), epoch*len(netD_dataloader)+i)
        writer.add_scalar('D(x)', D_x, epoch*len(netD_dataloader)+i)
        writer.add_scalar('D(G(z1))', D_G_z1, epoch*len(netD_dataloader)+i)
        writer.add_scalar('D(G(z2))', D_G_z2, epoch*len(netD_dataloader)+i)
        writer.add_scalar('lrD', schedulerD.get_lr()[0], epoch*len(netD_dataloader)+i)
        writer.add_scalar('lrG', schedulerG.get_lr()[0], epoch*len(netD_dataloader)+i)

        
        if i % 5 == 0:
            # Check how the generator is doing by saving G's output on fixed_noise
            # Save to tensorboard
            with torch.no_grad():
                fixed_noise = dataG.to(device)
                fake = netG(fixed_noise).detach().cpu()
                real = dataD.detach().cpu()
                origin = dataG.detach().cpu()
            fake_img_grid = vutils.make_grid(fake, padding=2, normalize=True)
            real_img_grid = vutils.make_grid(real, padding=2, normalize=True)
            origin_img_grid = vutils.make_grid(origin, padding=2, normalize=True)
            writer.add_image('fake_images_0', fake_img_grid[0].unsqueeze(0), epoch*len(netD_dataloader)+i)
            writer.add_image('fake_images_1', fake_img_grid[1].unsqueeze(0), epoch*len(netD_dataloader)+i)
            writer.add_image('fake_images_2', fake_img_grid[2].unsqueeze(0), epoch*len(netD_dataloader)+i)
            writer.add_image('real_images_0', real_img_grid[0].unsqueeze(0), epoch*len(netD_dataloader)+i)
            writer.add_image('real_images_1', real_img_grid[1].unsqueeze(0), epoch*len(netD_dataloader)+i)
            writer.add_image('real_images_2', real_img_grid[2].unsqueeze(0), epoch*len(netD_dataloader)+i)

    schedulerD.step()
    schedulerG.step()

    # Save model
    if epoch % 10 == 9:
        torch.save(netG.state_dict(), 'netG%d.pth'%epoch)
        torch.save(netD.state_dict(), 'netD%d.pth'%epoch)
