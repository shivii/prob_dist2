# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math 
import sys
import numpy

from datetime import datetime

def print_with_time(*argument):
    #start_time = time.localtime( time.time() )  
    start_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    print(start_time, argument)


"""
k is the kernel size
p is the padding
"""
def get_neigh(image, k, p):
    #print("calculating neighbours")
    unfold = nn.Unfold(kernel_size=(k,k), padding=p)
    output = unfold(image)
    swapped_ouput = torch.swapaxes(output, 1, 2)
    
    return swapped_ouput

def get_rgb(image):
    #print("seperating out into channels")
    image_r = image[0, 0, :, :]
    image_g = image[0, 1, :, :]
    image_b = image[0, 2, :, :]
    image_r = image_r.unsqueeze(0).unsqueeze(0)
    image_g = image_g.unsqueeze(0).unsqueeze(0)
    image_b = image_b.unsqueeze(0).unsqueeze(0)
    return image_r, image_g, image_b

def get_hist(img, bins):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neigh_hist = torch.zeros(0,bins).to(device)
    img_r, img_c = img.shape
    for i in range(1550,1554):
        hist_r = img[i, :]
        hist = torch.histc(hist_r, bins = bins, min=0, max=255)
        neigh_hist = torch.cat((neigh_hist, hist.unsqueeze(0)), 0)
    return neigh_hist    



def get_pdf(tensor, sigma):
    hist_squared = torch.pow(tensor, 2) 
    normalised_with_sigma = hist_squared/(2*sigma*sigma)
    gaussian_space = torch.exp(-normalised_with_sigma).unsqueeze(0)
    #print_with_time("gaussian space", gaussian_space.shape)
    
    gaussian_nieghbourhood_sums = gaussian_space.sum(dim=2)
    gaussian_nieghbourhood_sums_repeated = gaussian_nieghbourhood_sums.unsqueeze(2)
    gaussian_distribution = gaussian_space/gaussian_nieghbourhood_sums_repeated

    print(gaussian_distribution.shape)

    return gaussian_distribution


def hist_divergence(image1, image2, sigma, kernel, bins):
    #step1 get images as floats
    image_p = image1.to(torch.float32)
    image_q = image2.to(torch.float32)

    calculate_probability_distribution_histogram(image_p, sigma, kernel, bins) 

    #step2 get pdf of histograms of neighbours
    #hist_p = calculate_probability_distribution_histogram(image_p, sigma, kernel, bins)   
    #hist_q = calculate_probability_distribution_histogram(image_q, sigma, kernel, bins)
    #print("hist shape:", hist_p.shape, hist_q.shape)

    #step3 get JS Divergence
    #div = get_JSDiv(hist_p, hist_q, bins)

    #return div

def calculate_probability_distribution_histogram(image, sigma, kernel, bins):
    #print("gaussian begin")
    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # step1 get the nieghbours for each channel
    neigh = get_neigh(image, kernel, padding).squeeze(0)
    #print_with_time("neighbour shapes", neigh[1000])

    # step2 get histogran of neighbours
    hist = batch_histogram(neigh.long())    
    hist_avg = hist.sum(1)/hist.count_nonzero(dim=1)
    print("sum", hist_avg.shape)
    repeat_hist_avg = hist_avg.unsqueeze(1).repeat(1,256)
    diff_avg = (hist - repeat_hist_avg)/repeat_hist_avg
    diff_avg[diff_avg==-1] = 0
    diff_avg = diff_avg * repeat_hist_avg
    print("diff avg:", diff_avg.shape)
    hist_gaussian = torch.exp(-torch.pow(diff_avg, 2)/ (2 * sigma * sigma))
    """remove 1's from hist_gaussian as these represent 0's which are not part of sample space"""
    hist_gaussian[hist_gaussian==1] = 0
    print("hist_gaussian", hist_gaussian[0])
    gaussian_hist_sums = hist_gaussian.sum(dim=1)
    gaussian_hist_sums_repeated = gaussian_hist_sums.unsqueeze(1)
    gaussian_distribution = hist_gaussian/gaussian_hist_sums_repeated

    print(gaussian_distribution[0])


    # step3 get PDF
    #pdf = get_pdf(hist, sigma)

    #step4
    #return pdf

def batch_histogram(data_tensor, num_classes=-1):
    """
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    batch_hist = torch.nn.functional.one_hot(data_tensor, num_classes).sum(dim=-2)
    min = data_tensor.min()
    max = data_tensor.max()
    if min == 0 and max == 255:
        return batch_hist
    else:
        padding_left = min - 0
        padding_right = 255 - max
        batch_hist = F.pad(input=batch_hist, pad=(padding_left, padding_right), mode='constant', value=0) 
        return batch_hist
    


def get_JSDiv(image1, image2):
    # for any pdf A, adding epsilon avoids 0's in the tensor A
    # to counter the addition of epsilon,
    # (A + epsilon )* 1/ 1+n*epsilon 
    epsilon = 1e-20
    real_tensor = (image1 + epsilon) * (1/(1 + 255 * epsilon))
    fake_tensor = (image2 + epsilon) * (1/(1 + 255 * epsilon))

    # step4 joint distribution of 2 tensors
    m = (real_tensor + fake_tensor)/2

    # step5 compute JS divergence = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
    kl_real_fake = (real_tensor) * ((real_tensor)/(m)).log()
    kl_fake_real = (fake_tensor) * ((fake_tensor)/(m)).log()

    #get_details(kl_real_fake,"kl_divergence_elements")

    kl_per_pixel_real_fake = kl_real_fake.sum(dim=1)
    kl_per_pixel_real_fake[kl_per_pixel_real_fake < 0] = 0
    #get_details(kl_per_pixel_real_fake,"kl_per_pixel_real_fake")

    kl_per_pixel_fake_real = kl_fake_real.sum(dim=1)
    kl_per_pixel_fake_real[kl_per_pixel_fake_real < 0] = 0
    #get_details(kl_per_pixel_fake_real,"kl_per_pixel_fake_real")


    js_div = 0.5 * kl_per_pixel_real_fake + 0.5 * kl_per_pixel_fake_real

    return js_div

def calculate_probability_distribution_simple(image, sigma, kernel):
    # get r,g,b components
    r, g, b = get_rgb(image)

    # get kernel size and padding
    padding = math.floor(kernel/2)

    # get the nieghbours for each channel
    neigh_r = get_neigh(r, kernel, padding).squeeze(0)
    neigh_g = get_neigh(g, kernel, padding).squeeze(0)
    neigh_b = get_neigh(b, kernel, padding).squeeze(0)

    # get histogram of neighbours
    hist_r = batch_histogram(neigh_r.long())
    hist_g = batch_histogram(neigh_g.long()) 
    hist_b = batch_histogram(neigh_b.long()) 

    #getting sum along dim 2
    neigh_r_sum = hist_r.sum(1)
    neigh_g_sum = hist_g.sum(1)
    neigh_b_sum = hist_b.sum(1)

    #repeat the sum values along every dim
    neigh_r_repeat = neigh_r_sum.unsqueeze(1).repeat(1, 256)
    neigh_g_repeat = neigh_g_sum.unsqueeze(1).repeat(1, 256)
    neigh_b_repeat = neigh_b_sum.unsqueeze(1).repeat(1, 256)

    # probability = current value/sum
    prob_r = hist_r / neigh_r_repeat
    prob_g = hist_g / neigh_g_repeat
    prob_b = hist_b / neigh_b_repeat

    return prob_r, prob_g, prob_b

def get_KLDiv(image1, image2):
    # for any pdf A, adding epsilon avoids 0's in the tensor A
    # to counter the addition of epsilon,
    # (A + epsilon )* 1/ 1+n*epsilon 
    epsilon = 1e-20
    real_tensor = image1 + epsilon
    fake_tensor = image2 + epsilon


    # step5 compute JS divergence = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
    kl_real_fake = (real_tensor) * ((real_tensor)/(fake_tensor)).log()
    kl_fake_real = (fake_tensor) * ((fake_tensor)/(real_tensor)).log()

    print("KL real-> fake", kl_real_fake.mean())
    print("KL fake-> real", kl_fake_real.mean())

    return kl_real_fake
    
def pdf_divergence(image1, image2, sigma, kernel):
    """
    Denormalize image
    """
    im1_r, im1_g, im1_b = scale_image_0_1(image1)
    im2_r, im2_g, im2_b = scale_image_0_1(image2)

    #get JS Divergence
    div_r = get_KLDiv(im1_r, im2_r)
    div_g = get_KLDiv(im1_g, im2_g)
    div_b = get_KLDiv(im1_b, im2_b)

    div = div_r.mean() + div_g.mean() + div_b.mean()
    div = div * 1e+06
    return div

    

def scale_image_0_1(tensor_image):
    # get r,g,b components
    tensor_image_r, tensor_image_g, tensor_image_b = get_rgb(tensor_image)

    epsilon = 1e-20
    # step 1: convert it to [0 ,2]
    tensor_image_r = tensor_image_r +1
    tensor_image_g = tensor_image_g +1
    tensor_image_b = tensor_image_b +1

    # step 2: convert it to [0 ,1]
    tensor_image_r = (tensor_image_r / 2 + epsilon) / (1/(1+256*256*epsilon))
    tensor_image_g = (tensor_image_g / 2 + epsilon) / (1/(1+256*256*epsilon))
    tensor_image_b = (tensor_image_b / 2 + epsilon) / (1/(1+256*256*epsilon))

    #tensor_image_0_1r = tensor_image_r / (tensor_image_r.max() - tensor_image_r.min())
    #tensor_image_0_1g = tensor_image_g / (tensor_image_g.max() - tensor_image_g.min())
    #tensor_image_0_1b = tensor_image_b / (tensor_image_b.max() - tensor_image_b.min())

    tensor_image_sumr = tensor_image_r.sum()
    tensor_image_sumg = tensor_image_g.sum()
    tensor_image_sumb = tensor_image_b.sum()

    tensor_norm_r = tensor_image_r /tensor_image_sumr
    tensor_norm_g = tensor_image_g /tensor_image_sumg
    tensor_norm_b = tensor_image_b /tensor_image_sumb
    
    #print("sum:", tensor_image_sumr, tensor_image_sumg, tensor_image_sumb)
    #print("Min, Max R:", tensor_norm_r.unique(), tensor_norm_r.max(), tensor_norm_r.sum())
    #print("Min, Max G:", tensor_norm_g.min(), tensor_norm_g.max(), tensor_norm_g.sum())
    #print("Min, Max B:", tensor_norm_b.min(), tensor_norm_b.max(), tensor_norm_b.sum())

    return tensor_norm_r, tensor_norm_g, tensor_norm_b

    
"""
n is the number of repetitions
"""
def get_flattened_and_repeated(t, n):
    #print("genrating a flattened and repeated tensor")
    return torch.flatten(t).unsqueeze(1).repeat(1,1,n)

def get_details(t, label):
    print(label, " shape ", t.shape)
    print(label, " minimum ", t.min())
    print(label, " maximum ", t.max())

def denorm(image):
    device = image.get_device()
    # getting the values into range [0,1] from [-1, 1] : denormazlizing
    image = image.squeeze(0) * 0.5 + 0.5

    # converting toTensor to toPIL image
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.PILToTensor()
        ])
    image = transform(image).to(device)

    return image


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    image1 = trans(input_image1).to(device)
    image2 = trans(input_image2).to(device)
    div = pdf_divergence(image1.unsqueeze(0),image2.unsqueeze(0), sigma=1, kernel=5)
    print("div:", div * 1e+06)

    

 
