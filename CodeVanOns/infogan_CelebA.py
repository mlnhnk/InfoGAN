# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:26:14 2019

@author: User
"""

# CSC 321, Assignment 4
#
# This is the main training file for the vanilla GAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to checkpoints_vanilla/ and samples_vanilla/):
#       python vanilla_gan.py

import os
import pdb
import pickle
import argparse
import time

import warnings
warnings.filterwarnings("ignore")

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Local imports
import utils
from data_loader import get_celeba_loader
from models_CelebA import Generator, Discriminator, Recognition, SharedPartDQ


#SEED = 11
#
## Set the random seed manually for reproducibility.
#np.random.seed(SEED)
#torch.manual_seed(SEED)
#if torch.cuda.is_available():
#    torch.cuda.manual_seed(SEED)


class log_gaussian:

  def __call__(self, x, mu, var):
      logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
      (x-mu).pow(2).div(var.mul(2.0)+1e-6)
                
      return logli.sum(1).mean().mul(-1)

def print_models(DQ, D, Q, G):
    """Prints model information for the generators and discriminators.
    """
    print("                    DQ                  ")
    print("---------------------------------------")
    print(DQ)
    print("---------------------------------------")

    print("                    D                  ")
    print("---------------------------------------")
    print(D)
    print("---------------------------------------")
    
    print("                    Q                  ")
    print("---------------------------------------")
    print(Q)
    print("---------------------------------------")

    print("                    G                  ")
    print("---------------------------------------")
    print(G)
    print("---------------------------------------")
    

def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = Generator(noise_size=opts.noise_size, conv_dim=opts.conv_dim)
    D = Discriminator()
    Q = Recognition(categorical_dims=opts.cat_dim_size, continuous_dims=opts.cont_dim_size)
    DQ = SharedPartDQ()
    
    if opts.display_debug:
        print_models(DQ, D, Q, G)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        Q.cuda()
        DQ.cuda()
        if opts.display_debug:
            print('Models moved to GPU.')

    return G, D, Q, DQ


def checkpoint(iteration, G, D, Q, DQ, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G.pkl')
    D_path = os.path.join(opts.checkpoint_dir, 'D.pkl')
    Q_path = os.path.join(opts.checkpoint_dir, 'Q.pkl')
    DQ_path = os.path.join(opts.checkpoint_dir, 'DQ.pkl')
    Opts_path = os.path.join(opts.checkpoint_dir, 'opts.pkl')
    
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)
    torch.save(Q.state_dict(), Q_path)
    torch.save(DQ.state_dict(), DQ_path)
    pickle.dump( opts, open( Opts_path, "wb" ) )
    
def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    if opts.load == None:
        print("None selected, thus we assume we load from checkpoint_dir.")
        load_path = opts.checkpoint_dir
    else:
        print("Use opts.load")
        load_path = opts.load
    
    G_path = os.path.join(load_path, 'G.pkl')
    D_path = os.path.join(load_path, 'D.pkl')
    Q_path = os.path.join(load_path, 'Q.pkl')
    DQ_path = os.path.join(load_path, 'DQ.pkl')

    G, D, Q, DQ = create_model(opts)

    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    Q.load_state_dict(torch.load(Q_path, map_location=lambda storage, loc: storage))
    DQ.load_state_dict(torch.load(DQ_path, map_location=lambda storage, loc: storage))

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        Q.cuda()
        DQ.cuda()
        if opts.display_debug:
            print('Models moved to GPU.')

    return G, D, Q, DQ


def create_image_grid(array, ncols=None):
    """
    """
    num_images, channels, cell_h, cell_w = array.shape

    if not ncols:
        ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros((cell_h*nrows, cell_w*ncols, channels), dtype=array.dtype)
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w, :] = array[i*ncols+j].transpose(1, 2, 0)

    if channels == 1:
        result = result.squeeze()
    return result


def save_samples(G, fixed_noise, iteration, opts, extra_name):
    generated_images = G(fixed_noise)
    generated_images = utils.to_data(generated_images)
    
    grid = create_image_grid(generated_images, ncols=opts.interp_size)

    # merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'c{}_sample-{:06d}.png'.format(extra_name, iteration))
    scipy.misc.imsave(path, grid)
    print('Saved {}'.format(path))


def sample_noise(opts):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    batch_noise = utils.to_var(torch.rand(batch_size, opts.noise_size) * 2 - 1)
    
    random_categories = np.random.randint(low=0, high=opts.cat_dim_size, size=batch_size)
    onehot_categories = np.eye(opts.cat_dim_size)[random_categories]
    
    batch_noise[:, :opts.cat_dim_size] = torch.tensor(onehot_categories)
    
    cont_latent_variables = torch.zeros([batch_size, opts.cont_dim_size]).uniform_() * 2 - 1
    batch_noise[:, opts.cat_dim_size:opts.cat_dim_size + opts.cont_dim_size] = cont_latent_variables
    return batch_noise, torch.LongTensor(random_categories).cuda(), cont_latent_variables.cuda()

def get_fixed_noise(opts, var=0):
    """
    Generate a PyTorch Variable of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Variable of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    batch_noise = utils.to_var(torch.rand(10 * opts.interp_size, opts.noise_size) * 2 - 1)
    
    onehot_categories = np.eye(10)
    
    for ind in range(10):
        batch_noise[ind*10:(ind+1)*10, :10] = torch.tensor(onehot_categories[ind])
        
    continous_spectrum = torch.linspace(-2, 2, steps=opts.interp_size).repeat(10)
    
    if var == 0:
        batch_noise[:,10] = continous_spectrum
        batch_noise[:,11] = 0
    if var == 1:
        batch_noise[:,10] = 0
        batch_noise[:,11] = continous_spectrum
    
#    cont_latent_variables = torch.zeros([batch_size, opts.cont_dim_size]).uniform_() * 2 - 1
#    batch_noise[:, opts.cat_dim_size:opts.cat_dim_size + opts.cont_dim_size] = cont_latent_variables
    return batch_noise.cuda()

def training_loop(train_dataloader, opts):
    """Runs the training loop.
        * Saves checkpoints every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D, Q, DQ = create_model(opts)

    # Create optimizers for the generators and discriminators
    d_optimizer = optim.Adam([{'params':DQ.parameters()}, {'params':D.parameters()}], 2e-4, [opts.beta1, opts.beta2])
    g_optimizer = optim.Adam([{'params':G.parameters()}, {'params':Q.parameters()}], 1e-3, [opts.beta1, opts.beta2])

    # Generate fixed noise for sampling from the generator
#    fixed_noise, random_categories, cont_latent_variables = sample_noise(opts)
    
    fixed_noise = []
    for i in range(opts.cont_dim_size):
        fixed_noise.append(get_fixed_noise(opts, var=i))
        
    iteration = 1

    total_train_iters = opts.num_epochs * len(train_dataloader)
    
    # Loss function:
    loss_criterion = torch.nn.BCELoss(reduction='elementwise_mean')
    if torch.cuda.is_available():
        loss_criterion.cuda()
        zeros_label = torch.autograd.Variable(torch.zeros(batch_size).float().cuda())
        ones_label = torch.autograd.Variable(torch.ones(batch_size).float().cuda())
        if opts.display_debug:
            print('MSE loss moved to GPU.')
    else:
        zeros_label = torch.autograd.Variable(torch.zeros(batch_size).float())
        ones_label = torch.autograd.Variable(torch.ones(batch_size).float())
        
    criterion_Q_dis = nn.CrossEntropyLoss().cuda()
    criterion_Q_con = log_gaussian()
    
    for epoch in range(opts.num_epochs):

        for real_images, real_labels in train_dataloader:

            #real_images, labels = utils.to_var(real_images), utils.to_var(real_labels).long().squeeze()
            real_images = utils.to_var(real_images)

            ################################################
            ###         TRAIN THE DISCRIMINATOR         ####
            ################################################
            d_optimizer.zero_grad()
            
            # First do shared part, then apply discriminator
            D_real_images = D(DQ(real_images))
                        
            D_real_loss = loss_criterion(D_real_images, ones_label[:real_images.size()[0]])

            batch_noise, _, _= sample_noise(opts)
            fake_images = G(batch_noise)
            
            D_fake_images = D(DQ(fake_images))
            D_fake_loss = loss_criterion(D_fake_images, zeros_label[:fake_images.size()[0]])
            
            D_total_loss = (D_real_loss + D_fake_loss)
            
            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################
            g_optimizer.zero_grad()
            
            batch_noise, category_target, continous_target = sample_noise(opts)
            fake_images = G(batch_noise)
            
            DQ_fake_images = DQ(fake_images)
            
            G_loss_fake = loss_criterion(D(DQ_fake_images), ones_label[:fake_images.size()[0]])
            
#            print(D(DQ_fake_images))
            
            cat, cont_mu, cont_sigma = Q(DQ_fake_images)
            
            dis_loss = criterion_Q_dis(cat, category_target)
            
            con_loss = criterion_Q_con(continous_target, cont_mu, cont_sigma)*0.1
 
            G_loss_total = G_loss_fake + dis_loss + con_loss
#            print([G_loss_fake.data[0], dis_loss.data[0], con_loss.data[0]])

            G_loss_total.backward()
            g_optimizer.step()


            # Print the log info
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}'.format(
                       iteration, total_train_iters, D_real_loss.item(), D_fake_loss.item(), G_loss_total.item()))

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                for i in range(opts.cont_dim_size):
                    save_samples(G, fixed_noise[i], iteration, opts, i)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, Q, DQ, opts)

            iteration += 1


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    train_dataloader = get_celeba_loader(opts)
    
    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)
#
    startTime = time.time()
    training_loop(train_dataloader, opts)
    print("Training {} epochs took {:0.2f}s.".format(opts.num_epochs, time.time()-startTime))
    print("That is {:0.2f}s per epoch.".format((time.time()-startTime)/float(opts.num_epochs)))


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--noise_size', type=int, default=228)
    parser.add_argument('--cont_dim_size', type=int, default=4)
    parser.add_argument('--cat_dim_size', type=int, default=10)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64, help='The number of images in a batch.')
    parser.add_argument('--interp_size', type=int, default=10, help='The number of interpolation for continuous variables images displayed.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)

    # Directories and checkpoint/sample iterations
    parser.add_argument('--display_debug', type=str, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default='./five_latent_variables')
    parser.add_argument('--sample_dir', type=str, default='./five_latent_variables_sample')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=500)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    batch_size = opts.batch_size

    print(opts)
    main(opts)
