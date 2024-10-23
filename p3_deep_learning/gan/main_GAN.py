import os
import sys

import torch
import torchvision
from matplotlib import pyplot as plt
from qqdm import qqdm
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import Generator
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from seeds import same_seeds
from dataset import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
same_seeds(2021)
workspace_dir=''
dataset = get_dataset(os.path.join(workspace_dir, 'faces'))

# netG=Generator(100)
# print(netG)
# netD=Discriminator(3)
# print(netD)

batch_size =64
z_dim = 100
z_sample = Variable(torch.randn(100, z_dim)).to(device)
lr = 1e-4


n_epoch = 500 # 50
n_critic = 10 # 5
# clip_value = 0.01

log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
G = Generator(in_dim=z_dim).to(device)
D = Discriminator(3).to(device)
G.train()
D.train()

# Loss
criterion = nn.BCELoss()


# Optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)


# DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def main(argv=None):
    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = qqdm(dataloader)#进度条
        for i, data in enumerate(progress_bar):
            imgs = data
            imgs = imgs.to(device)

            bs = imgs.size(0) # 64样本大小

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).to(device)
            r_imgs = Variable(imgs).to(device)
            f_imgs = G(z) #大小为64*3*64*64

            #""" Medium: Use WGAN Loss. """
            # Label
            r_label = torch.ones((bs)).to(device)
            f_label = torch.zeros((bs)).to(device)

            # Model forwarding
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # Compute the loss for the discriminator.
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # WGAN Loss
            # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

            # Model backwarding
            D.zero_grad()
            loss_D.backward()

            # Update the discriminator.
            opt_D.step()

            #""" Medium: Clip weights of discriminator. """
            # for p in D.parameters():
            #    p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).to(device)
                f_imgs = G(z)

                # Model forwarding
                f_logit = D(f_imgs)

                #""" Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                loss_G = criterion(f_logit, r_label)
                # WGAN Loss
                # loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()

                # Update the generator.
                opt_G.step()

            steps += 1

            # Set the info of the progress bar
            #   Note that the value of the GAN loss is not directly related to
            #   the quality of the generated images.
            progress_bar.set_infos({
                'Loss_D': round(loss_D.item(), 4),
                'Loss_G': round(loss_G.item(), 4),
                'Epoch': e + 1,
                'Step': steps,
            })

        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(log_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')

        # Show generated images in the jupyter notebook.
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()

        if (e + 1) % 5 == 0 or e == 0:
            # Save the checkpoints.
            torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))

if __name__=='__main__':
    sys.exit(main())