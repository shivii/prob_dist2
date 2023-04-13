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

    print(gaussian_distribution[0])F.pad(input=batch_hist, pad=(padding_left, padding_right), mode='constant', value=0) 


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
    print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("data tensor  shape:", data_tensor.shape)
    print("going to use one_hot:::::")
    print("min is ", data_tensor.min())
    print("max is ", data_tensor.max())
    batch_hist = torch.nn.functional.one_hot(data_tensor, num_classes).sum(dim=-2)
    print("batch_hist:", batch_hist.shape)
    min = data_tensor.min()
    max = data_tensor.max()
    print("min max", min, max)
    if min == 0 and max == 255:
        print(batch_hist.shape)
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
        return batch_hist
    else:
        padding_left = min - 0
        padding_right = 255 - max
        batch_hist = F.pad(input=batch_hist, pad=(padding_left, padding_right), mode='constant', value=0) 
        print(batch_hist.shape)
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------")
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

    print("js_div", js_div.mean())
    return js_div

def print_memory_usage():    
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    print ("memory =", r-a)


def calculate_probability_distribution_simple(image, sigma, kernel):
    # get r,g,b components
    r, g, b = get_rgb(image)

    # get kernel size and padding
    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # get the nieghbours for each channel
    neigh_r = get_neigh(r, kernel, padding).squeeze(0)
    neigh_g = get_neigh(g, kernel, padding).squeeze(0)
    neigh_b = get_neigh(b, kernel, padding).squeeze(0)
    print_with_time("neighbour shapes", neigh_r.shape)
    print_memory_usage()

    # get histogram of neighbours
    hist_r = batch_histogram(neigh_r.long()) 
    hist_g = batch_histogram(neigh_g.long()) 
    hist_b = batch_histogram(neigh_b.long()) 
    print("histogram shape:", hist_r.shape)
    print_memory_usage()

    #getting sum along dim 2
    neigh_r_sum = hist_r.sum(1)
    neigh_g_sum = hist_g.sum(1)
    neigh_b_sum = hist_b.sum(1)
    print("sum shape:", neigh_r_sum.shape)

    #repeat the sum values along every dim
    neigh_r_repeat = neigh_r_sum.unsqueeze(1).repeat(1, 256)
    neigh_g_repeat = neigh_g_sum.unsqueeze(1).repeat(1, 256)
    neigh_b_repeat = neigh_b_sum.unsqueeze(1).repeat(1, 256)
    print("neigh sum size:", neigh_r_repeat.shape)
    print_memory_usage()

    # probability = current value/sum
    prob_r = hist_r / neigh_r_repeat
    prob_g = hist_g / neigh_g_repeat
    prob_b = hist_b / neigh_b_repeat
    print("prob size:", prob_r.shape)

    return prob_r, prob_g, prob_b
    
def pdf_divergence(image1, image2, sigma, kernel):
    #step1 get images as floats
    image_p = image1.to(torch.float32)
    image_q = image2.to(torch.float32)
    print("image shape:", image_p.shape)

    #get probabilities of images for different channels
    prob1_r, prob1_g, prob1_b = calculate_probability_distribution_simple(image_p, sigma, kernel) 
    prob2_r, prob2_g, prob2_b = calculate_probability_distribution_simple(image_q, sigma, kernel) 
    print("prob shape:",prob1_r.shape, prob1_g.shape, prob1_b.shape)

    #get JS Divergence
    div_r = get_JSDiv(prob1_r, prob2_r)
    div_g = get_JSDiv(prob1_g, prob2_g)
    div_b = get_JSDiv(prob1_b, prob2_b)

    div = div_r.mean() + div_g.mean() + div_b.mean()

    return div
    
    
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



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print(torch.cuda.get_device_properties(0))
    print_memory_usage()
    torch.set_printoptions(threshold=100000)

    transform = transforms.Compose([
    transforms.PILToTensor()
    ])

    input_image1 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_real_B.png")).to(device).unsqueeze(0).to(torch.float32)
    input_image2 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_rec_B.png")).to(device).unsqueeze(0).to(torch.float32)

    input_image3 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1920_real_A.png")).to(device).unsqueeze(0).to(torch.float32)
    input_image4 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1920_rec_A.png")).to(device).unsqueeze(0).to(torch.float32)

    input_image5 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/n02381460_489.jpg")).to(device).unsqueeze(0).to(torch.float32)
    input_image6 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/n02391049_87.jpg")).to(device).unsqueeze(0).to(torch.float32)

    print_memory_usage()

    print(pdf_divergence(input_image3,input_image4, sigma=1, kernel=5))

    

 