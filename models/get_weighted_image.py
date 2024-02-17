
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

class WImage(nn.Module):
    def __init__(self):
        super(WImage, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    
    """
    k is the kernel size
    p is the padding
    """
    def get_neigh(self, image, k, p):
        #print("calculating neighbours after: image shape", image.shape)
        unfold = nn.Unfold(kernel_size=(k,k), padding=p)
        output = unfold(image)
        #print("neigh output", output.shape)
        swapped_ouput = torch.swapaxes(output, 1, 2)
        
        return swapped_ouput
    
    def get_rgb(self, image):
        #print("seperating out into channels")
        image_r = image[0, 0, :, :]
        image_g = image[0, 1, :, :]
        image_b = image[0, 2, :, :]
        image_r = image_r.unsqueeze(0).unsqueeze(0)
        image_g = image_g.unsqueeze(0).unsqueeze(0)
        image_b = image_b.unsqueeze(0).unsqueeze(0)
        return image_r, image_g, image_b
    

    """
    Computes pdf of a gaussian distribution with the formula exp^(-|x_i - x_j|^2)/2*sigma*sigma
    return a gaussian distributoin takes in a histogram values
    """
    
    def denorm(self, image):
        # getting the values into range [0,1] from [-1, 1] : denormazlizing
        image = image.squeeze(0) * 0.5 + 0.5

        # converting toTensor to toPIL image
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.PILToTensor()
            ])
        image = transform(image)

        return image.unsqueeze(0)

    """
    Calculate PDF of gaussian distributions of every pixel with its neighbors
    """
    def calculate_pdf_gaussian(self, r,g,b, kernel, sigma):
        #print("gaussian begin")
        no_of_neigh = kernel*kernel
        padding = math.floor(kernel/2)

        # step2 get the nieghbours for each channel
        neigh_r = self.get_neigh(r, kernel, padding)
        neigh_g = self.get_neigh(g, kernel, padding)
        neigh_b = self.get_neigh(b, kernel, padding)
        
        #print("in gauss: neigh r shape:", neigh_r.shape)
        # step3 build flattened repeated rgb tensor
        repeated_r = self.get_flattened_and_repeated(r,no_of_neigh)
        repeated_g = self.get_flattened_and_repeated(g,no_of_neigh)
        repeated_b = self.get_flattened_and_repeated(b,no_of_neigh)
        #print("in gauss: repeated neigh r shape:", repeated_r.shape)
        # step4: Xi - Xj i.e : subtract repeated_r and neigh_r

        diff_r = neigh_r - repeated_r
        diff_g = neigh_g - repeated_g
        diff_b = neigh_b - repeated_b

        # step5: (Xi - Xj)^2 i.e : square
        diff_squared_r = torch.pow(diff_r, 2) 
        diff_squared_g = torch.pow(diff_g, 2)
        diff_squared_b = torch.pow(diff_b, 2)

        # step6: (Xi - Xj)^2/2*sigma*sigma
        sum_normalised_with_sigma_r = diff_squared_r/(2*sigma*sigma)
        sum_normalised_with_sigma_g = diff_squared_g/(2*sigma*sigma)
        sum_normalised_with_sigma_b = diff_squared_b/(2*sigma*sigma)

        #step6: exp^((Xi - Xj)^2/2*sigma*sigma)
        gaussian_space_r = torch.exp(-sum_normalised_with_sigma_r)
        gaussian_space_g = torch.exp(-sum_normalised_with_sigma_g)
        gaussian_space_b = torch.exp(-sum_normalised_with_sigma_b)

        # convcerting from shape [65536,9] to [65536]
        gaussian_distribution_r = gaussian_space_r.sum(dim=2)-1
        gaussian_distribution_g = gaussian_space_g.sum(dim=2)-1
        gaussian_distribution_b = gaussian_space_b.sum(dim=2)-1

        #print("gaussian,shape", gaussian_space_b.min(), gaussian_space_b.max())

        return gaussian_distribution_r.squeeze(0), gaussian_distribution_g.squeeze(0), gaussian_distribution_b.squeeze(0)

    """
    n is the number of repetitions
    """
    def get_flattened_and_repeated(self, t, n):
        #print("genrating a flattened and repeated tensor")
        return torch.flatten(t).unsqueeze(1).repeat(1,1,n)
    
    def wt_image(self, image, kernel=3, sigma=0.5, alpha=0.05):

        # get r,g,b components
        r, g, b = self.get_rgb(image)

        w_image_r, w_image_g, w_image_b = self.calculate_pdf_gaussian(r, g, b, kernel, sigma)

        w_image_r = w_image_r.resize(256,256).unsqueeze(0).unsqueeze(0)
        w_image_g = w_image_g.resize(256,256).unsqueeze(0).unsqueeze(0)
        w_image_b = w_image_b.resize(256,256).unsqueeze(0).unsqueeze(0)

        w_image = torch.cat([w_image_r, w_image_g, w_image_b], dim=1)

        # new_image = x_i + alpha * sum neighb
        no_of_neigh = kernel*kernel - 1
        new_image = image + alpha * w_image

        """
        # scale the new image [-1,1]
        # calculating min and max
        min : -1 + 8 * alpha (e^ ((2 * 2)/2*sigma*sigma ))
        max = 1 +   8 * alpha (e^ ((0)/2*sigma*sigma )) = 1 +   8 * alpha
        new_image = (new_image * 2) -1
        """
        min = -1 + (no_of_neigh * alpha * (math.exp(-4/ (2*sigma*sigma))))
        max = 1.0 + no_of_neigh * alpha

        new_image = (new_image - min) / (max-min)
        new_image = new_image * 2 - 1 

        return new_image
    
    
    
if __name__ == '__main__':
    print(torch.cuda.get_device_properties(0))
    torch.set_printoptions(threshold=100000)

    input_image1 = Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_real_B.png")
    input_image2 = Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_rec_B.png")

    """
    Normalise image
    """
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    image1 = trans(input_image1)
    image2 = trans(input_image2)
    print("image shape", image1.shape, image2.shape)

    wt = WImage()
    wt_image1 = wt.wt_image(image1.unsqueeze(0), kernel=5)
    wt_image2 = wt.wt_image(image2.unsqueeze(0), kernel=5)



    print("wt_image", wt_image1.shape,wt_image1.min(), wt_image1.max(), wt_image1.sum())
