import os
import yaml
import torch
import argparse
import functools
import numpy as np
import csquaregan.logger as logger
from csquaregan.discriminator import calculate_gradient_penalty, extract_patches_2d, Discriminator as PatchDiscriminator
from csquaregan.market_loader import Dataset
from torchvision import transforms
from csquaregan.unet import UNet as Generator
from torch.utils import data
import torch.optim as optim
import matplotlib.pyplot as plt

def load_config(file):
    with open(file, 'r') as f:
        y = yaml.load(f, Loader=yaml.SafeLoader)
    return argparse.Namespace(**y)

parser = argparse.ArgumentParser(description='Cycle In Cycle GAN training')
parser.add_argument('config', type=str)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

# configurations
config = load_config(args.config)

logger.info('Config loaded', config)

logger.DEBUG = args.debug
if args.debug:
    logger.info('Debug mode enabled')

# device definition
if torch.cuda.is_available():
    if config.cuda:
        device = torch.device('cuda')
        logger.success('Current device is', device)
    else:
        device = torch.device('cpu')
        logger.warning('Current device is', device, '(but cuda is available, set "cuda: True" in config file)')
else:
    device = torch.device('cpu')
    if config.cuda:
        logger.fail('Current device is', device, '(but cuda is NOT available)')
    else:
        logger.success('Current device is', device)

# network definition
ik2i = Generator(in_channels=81, out_channels=3).to(device).train()
i2k = Generator(in_channels=3, out_channels=78).to(device).train()

i_disc = PatchDiscriminator(channels=84).to(device).train()
k_disc = PatchDiscriminator(channels=81).to(device).train()

logger.info(
    '<image, kp> -> <image> network has',
    sum([
        functools.reduce(lambda v, a: a*v, p.size(), 1)
        for p in ik2i.parameters()
    ]),
    'parameters'
)
logger.info(
    '<image> -> <kp> network has',
    sum([
        functools.reduce(lambda v, a: a*v, p.size(), 1)
        for p in i2k.parameters()
    ]),
    'parameters'
)

# data loading
train_dataset = Dataset(
    config.image_dir,
    config.image_format,
    transforms.Compose([
        transforms.ToTensor()
    ])
)

test_dataset = Dataset(
    config.test_image_dir,
    config.image_format,
    transforms.Compose([
        transforms.ToTensor()
    ])
)

logger.info('Data loaded', train_dataset)

train_generator = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=5)
test_generator = data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

test_img1, test_img2, test_kp1, test_kp2 = [i.to(device) for i in next(iter(test_generator))]

os.makedirs(config.session_name, exist_ok=True)

# optimizers
i_disc_optimizer = optim.Adam(i_disc.parameters(), lr=config.i_lr, betas=(config.i_adam_b1, config.i_adam_b2))
k_disc_optimizer = optim.Adam(k_disc.parameters(), lr=config.i_lr, betas=(config.i_adam_b1, config.i_adam_b2))
ik2i_optimizer = optim.Adam(list(ik2i.parameters()) + list(i2k.parameters()), lr=config.i_lr, betas=(config.i_adam_b1, config.i_adam_b2))
i2k_optimizer = optim.Adam(list(ik2i.parameters()) + list(i2k.parameters()), lr=config.i_lr, betas=(config.i_adam_b1, config.i_adam_b2))

# misc
one = torch.FloatTensor([1]).to(device)
mone = one*-1

cycle_loss = torch.nn.L1Loss()
# training
for epoch in range(1, config.epochs + 1):
    batch_idx = 1
    logger.info('[epoch {0:2d}]'.format(epoch))
    for img1, img2, kp1, kp2 in train_generator:
        # batch preparation
        img1, img2, kp1, kp2 = img1.to(device), img2.to(device), kp1.to(device), kp2.to(device)
        i2k.zero_grad()
        ik2i.zero_grad()
        i_disc.zero_grad()
        k_disc.zero_grad()

        '''
        # I2I2I cycle
        g_img2 = ik2i(torch.cat([img1, kp2], 1))
        g_img1 = ik2i(torch.cat([g_img2, kp1], 1))

        # I2I2K
        g_kp2 = i2k(g_img2)

        # I2K2I
        g_kp1 = i2k(g_img1)
        '''
        # ++++++++++++++++ patches from discriminator's input ++++++++++++++++
        for param in ik2i.parameters():
            param.requires_grad = False
        for param in i2k.parameters():
            param.requires_grad = False
        for param in i_disc.parameters():
            param.requires_grad = True
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_kp2 = i2k(g_img2)                         # img2* -> kp2*
        i_dx_real = extract_patches_2d(
            torch.cat([img2, img1, kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3+3, config.patch_h, config.patch_w))
        i_dx_fake = extract_patches_2d(
            torch.cat([g_img2, img1, kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3+3, config.patch_h, config.patch_w))
        
        #import ipdb; ipdb.set_trace()
        # discriminate image triplets
        i_disc_loss_real = i_disc(i_dx_real).mean()*0.5
        i_disc_loss_real.backward(mone, retain_graph=True)
        i_disc_loss_fake = i_disc(i_dx_fake).mean()*0.5
        i_disc_loss_fake.backward(one, retain_graph=True)

        # gradient penalty
        i_disc_gradient_penalty = calculate_gradient_penalty(i_dx_real.shape[0], device, i_disc, i_dx_real.data, i_dx_fake.data, config.gp_lambda)
        i_disc_gradient_penalty.backward(retain_graph=True)

        i_disc_loss = i_disc_loss_real - i_disc_loss_fake + i_disc_gradient_penalty
        i_disc_w_dist = i_disc_loss_real - i_disc_loss_fake
        i_disc_optimizer.step()

        # generator loss
        for param in ik2i.parameters():
            param.requires_grad = True
        for param in i2k.parameters():
            param.requires_grad = True
        for param in i_disc.parameters():
            param.requires_grad = False
        
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_kp2 = i2k(g_img2)                         # img2* -> kp2*
        
        i_dx_fake = extract_patches_2d(
            torch.cat([g_img2, img1, kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3+3, config.patch_h, config.patch_w))
        
        i_ik2i_loss = i_disc(i_dx_fake).mean()
        i_ik2i_loss.backward(mone, retain_graph=True)
        ik2i_optimizer.step()

        # ++++++++++++++++ I2I2I cycle ++++++++++++++++
        for param in ik2i.parameters():
            param.requires_grad = True
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_img1 = ik2i(torch.cat([g_img2, kp1], 1))    # <img2*, kp1> -> img1*

        img1_idt_loss = cycle_loss(img1, g_img1)*config.lambda_idt_i
        img1_idt_loss.backward()
        ik2i_optimizer.step()

        # ++++++++++++++++ discriminate <image, kp> pairs ++++++++++++++++
        for param in ik2i.parameters():
            param.requires_grad = False
        for param in i2k.parameters():
            param.requires_grad = False
        for param in i_disc.parameters():
            param.requires_grad = False
        for param in k_disc.parameters():
            param.requires_grad = True
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_kp2 = i2k(g_img2)                         # img2* -> kp2*
        k_dx_real = extract_patches_2d(
            torch.cat([g_img2, kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3, config.patch_h, config.patch_w))
        k_dx_fake = extract_patches_2d(
            torch.cat([g_img2, g_kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3, config.patch_h, config.patch_w))
        
        # discriminate image triplets
        k_disc_loss_real = k_disc(k_dx_real).mean()*0.5
        k_disc_loss_real.backward(mone, retain_graph=True)
        k_disc_loss_fake = k_disc(k_dx_fake).mean()*0.5
        k_disc_loss_fake.backward(one, retain_graph=True)

        # gradient penalty
        k_disc_gradient_penalty = calculate_gradient_penalty(k_dx_real.shape[0], device, k_disc, k_dx_real.data, k_dx_fake.data, config.gp_lambda)
        k_disc_gradient_penalty.backward(retain_graph=True)

        k_disc_loss = k_disc_loss_real - k_disc_loss_fake + k_disc_gradient_penalty
        k_disc_w_dist = k_disc_loss_real - k_disc_loss_fake
        k_disc_optimizer.step()

        # generator loss
        for param in ik2i.parameters():
            param.requires_grad = True
        for param in i2k.parameters():
            param.requires_grad = True
        for param in i_disc.parameters():
            param.requires_grad = False
        for param in i_disc.parameters():
            param.requires_grad = False
        
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_kp2 = i2k(g_img2)                         # img2* -> kp2*
        
        k_dx_fake = extract_patches_2d(
            torch.cat([g_img2, g_kp2], 1),
            patch_shape=(config.patch_h, config.patch_w),
            step=[config.patch_h, config.patch_w], batch_first=True).reshape((-1, 78+3, config.patch_h, config.patch_w))
        
        k_i2k_loss = k_disc(k_dx_fake).mean()
        k_i2k_loss.backward(mone, retain_graph=True)
        i2k_optimizer.step()

        # ++++++++++++++++ K2I2K and I2K cycle ++++++++++++++++
        for param in i2k.parameters():
            param.requires_grad = True
        for param in ik2i.parameters():
            param.requires_grad = True
        g_img2 = ik2i(torch.cat([img1, kp2], 1))    # <img1, kp2> -> img2*
        g_img1 = ik2i(torch.cat([g_img2, kp1], 1))    # <img2*, kp1> -> img1*
        g_kp2 = i2k(g_img2)
        g_kp1 = i2k(g_img1)

        kp_idt_loss = cycle_loss(g_kp2, kp2)*config.lambda_idt_k + cycle_loss(g_kp1, kp1)*config.lambda_idt_k
        kp_idt_loss.backward()
        i2k_optimizer.step()
        
        if args.debug:
            logger.debug('[epoch {0:2d} batch {1:3d}] i-disc-loss={2:8.3f}, i-disc-w-dist={3:8.3f}, i-ik2i-loss={4:8.3f}, img1_idt_loss={5:8.3f}'.format(
                epoch, batch_idx, i_disc_loss.item(), i_disc_w_dist.item(), -i_ik2i_loss.item(), img1_idt_loss.item()
            ))
        batch_idx += 1
    # visualizing
    test_g_img2 = ik2i(torch.cat([test_img1, test_kp2], 1))    # <img1, kp2> -> img2*
    test_g_img1 = ik2i(torch.cat([test_g_img2, test_kp1], 1))    # <img2*, kp1> -> img1*
    test_g_kp2 = i2k(test_g_img2)
    test_g_kp1 = i2k(test_g_img1)
    os.makedirs(os.path.join(config.session_name, '{0:03d}'.format(epoch)), exist_ok=True)
    for i in range(test_g_img2.shape[0]):
        test_g_img2_np = test_g_img2[i].detach().cpu().numpy().transpose(1, 2, 0)
        test_g_img1_np = test_g_img1[i].detach().cpu().numpy().transpose(1, 2, 0)
        #test_g_kp2_np = np.repeat(test_g_kp2[i].detach().cpu().numpy().transpose(1, 2, 0), 3, 2)
        test_g_kp2_np = np.repeat(test_g_kp2[i].detach().cpu().numpy().transpose(1, 2, 0).mean(2)[:, :, np.newaxis], 3, 2)
        test_g_kp1_np = np.repeat(test_g_kp1[i].detach().cpu().numpy().transpose(1, 2, 0).mean(2)[:, :, np.newaxis], 3, 2)

        test_source = np.concatenate([
            test_img1[i].detach().cpu().numpy().transpose(1, 2, 0),
            np.repeat(test_kp1[i].detach().cpu().numpy().transpose(1, 2, 0).mean(2)[:, :, np.newaxis], 3, 2),
            test_img2[i].detach().cpu().numpy().transpose(1, 2, 0),
            np.repeat(test_kp2[i].detach().cpu().numpy().transpose(1, 2, 0).mean(2)[:, :, np.newaxis], 3, 2)
        ], 1)
        test_generated = np.concatenate([test_g_img1_np, test_g_kp1_np, test_g_img2_np, test_g_kp2_np], 1)
        test_vis = np.concatenate([test_source, test_generated], 0)

        plt.imsave(os.path.join(config.session_name, '{0:03d}'.format(epoch), '{0:03d}.png'.format(i)), test_vis)
    # visualizing
torch.save(ik2i.state_dict(), 'ik2i.pth')
torch.save(i2k.state_dict(), 'i2k.pth')
torch.save(k_disc.state_dict(), 'k_disc.pth')
torch.save(i_disc.state_dict(), 'i_disc.pth')
#import ipdb; ipdb.set_trace()
