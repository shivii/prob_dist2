# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math 
import sys
from types import SimpleNamespace
import numpy
#from options.train_options import TrainOptions
from datetime import datetime

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def get_JSDiv(self, real_tensor, fake_tensor, pdf=1):
        # for any pdf A, adding epsilon avoids 0's in the tensor A
        # to counter the addition of epsilon,
        # (A + epsilon )* 1/ 1+n*epsilon 
        epsilon = 1e-20

        # step4 joint distribution of 2 tensors
        m = (real_tensor + fake_tensor) * 0.5

        # compute JS divergence = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
        kl_real_fake = (real_tensor) * ((real_tensor)/(m)).log()
        kl_fake_real = (fake_tensor) * ((fake_tensor)/(m)).log()

        if pdf == 4:
            js_div = kl_real_fake.sum() * 0.5 + kl_fake_real.sum() * 0.5
        else:
            js_div = kl_real_fake.sum(dim=1) * 0.5 + kl_fake_real.sum(dim=1) * 0.5

        return js_div

    """
    k is the kernel size
    p is the padding
    """
    def get_neigh(self, image, k, p):
        #print("calculating neighbours after: image shape", image.shape)
        unfold = nn.Unfold(kernel_size=(k,k), padding=p)
        output = unfold(image)
        swapped_ouput = torch.swapaxes(output, 1, 2)
        
        return swapped_ouput


    """
    Computes pdf of a gaussian distribution with the formula exp^(-|x_i - x_j|^2)/2*sigma*sigma
    return a gaussian distributoin
    takes in a histogram values
    """
    
    def calculate_pdf_gaussian(self, image, kernel=3, sigma=1):
        no_of_neigh = kernel*kernel
        padding = math.floor(kernel/2)

        # step2 get the nieghbours for each channel
        neigh = self.get_neigh(image, kernel, padding)

        #print("in gauss: neigh r shape:", neigh_r.shape)
        # step3 build flattened repeated rgb tensor
        repeated = self.get_flattened_and_repeated(image,no_of_neigh)

        # step4: Xi - Xj i.e : subtract repeated_r and neigh_r
        diff = neigh - repeated
        diff_squared = torch.pow(diff, 2) 

        #normalised_diff_squared_sum = torch.nn.functional.normalize(diff_squared_sum, p=2.0, dim=1)
        sum_normalised_with_sigma = diff_squared/(2*sigma*sigma)
        gaussian_space = torch.exp(-sum_normalised_with_sigma)
        gaussian_nieghbourhood_sums = gaussian_space.sum(dim=2)
        gaussian_nieghbourhood_sums_repeated = gaussian_nieghbourhood_sums.unsqueeze(2)
        gaussian_distribution = gaussian_space/gaussian_nieghbourhood_sums_repeated

        return gaussian_distribution.squeeze(0)

    """
    n is the number of repetitions
    """
    def get_flattened_and_repeated(self, t, n):
        #print("genrating a flattened and repeated tensor")
        return torch.flatten(t).unsqueeze(1).repeat(1,1,n)
    
    def adv_loss(self, pred_real, pred_fake):
        """
        L_D = - log(4) + 2 * log(D(x)) + log(1 - D(G(z)))
            = -log(2) + KL(P_r || M) + KL(P_g || M)
        """
        #print("from adv_disc")
        prob1 = self.calculate_pdf_gaussian(pred_real)
        prob2 = self.calculate_pdf_gaussian(pred_fake)
        eps = 1e-12

        # KL(P_r || M) + KL(P_g || M)
        div = self.get_JSDiv(prob1, prob2)

        adversarial_loss = 2 * div.sum()
        #print(div.shape)
        return adversarial_loss
    
    def adv_loss_gen(self, prediction, target_is_real):
        """
        L_G = - log(4) + log(D(G(z)))
            = - log(2) + KL(P_g || M)
        """
        #print("from adv_gen")
        prob = self.calculate_pdf_gaussian(prediction)
        eps = 1e-12
        if target_is_real:
            target = torch.ones_like(prob)
        else:
            target = torch.zeros_like(prob)


        p, q = prob.view(-1, prob.size(-1)), target.view(-1, target.size(-1))
        
        m = (0.5 * (p + q)).log()

        # KL(P_g || M)
        div = 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log())) 

        #adversarial_loss = -torch.log(div + eps) 
        return div

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #opt = TrainOptions().parse() 

    print(torch.cuda.get_device_properties(0))
    torch.set_printoptions(threshold=100000)

    input_image1 = Image.open("/home/apoorvkumar/shivi/Phd/Project/data/struck_padded/trainA/2.png")
    input_image2 = Image.open("/home/apoorvkumar/shivi/Phd/Project/data/struck_padded/trainA/5.png")

    r1 = -2 
    r2 = 2
    a = 30
    b = 30
    img1 = (r1 - r2) * torch.rand(a,b) + r2
    img1 = img1.unsqueeze(0)
    img2 = (r1 - r2) * torch.rand(a,b) + r2
    img2 = img2.unsqueeze(0)
    print("img shape",img1.shape, img2.shape)


    calc_js = JSD().to(device)

    """
    Normalise image
    """
    trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                ])
    image1 = trans(img1).to(device)
    image2 = trans(img2).to(device)

    loss_D = calc_js.adv_loss(img1.unsqueeze(0).to(device), img2.unsqueeze(0).to(device))
    #loss_G = calc_js.adv_loss_gen(img1.unsqueeze(0).to(device), True)



    print("adv loss: ",loss_D)

    