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

# image should be a 4d tensor
def get_rgb(image):
    #print("seperating out into channels")
    image_r = image[0, 0, :, :].to(float)
    image_g = image[0, 1, :, :].to(float)
    image_b = image[0, 2, :, :].to(float)
    image_r = image_r.unsqueeze(0).unsqueeze(0)
    image_g = image_g.unsqueeze(0).unsqueeze(0)
    image_b = image_b.unsqueeze(0).unsqueeze(0)
    return image_r, image_g, image_b


"""
Computes pdf of a gaussian distribution with the formula exp^(-|x_i - x_j|^2)/2*sigma*sigma

return a gaussian distributoin

takes in a histogram values
"""
def get_pdf(tensor, sigma):
    hist_squared = torch.pow(tensor, 2) 
    normalised_with_sigma = hist_squared/(2*sigma*sigma)
    gaussian_space = torch.exp(-normalised_with_sigma).unsqueeze(0)
    #print_with_time("gaussian space", gaussian_space.shape)
    
    gaussian_nieghbourhood_sums = gaussian_space.sum (dim=2)
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
        batch_hist = F.pad(input=batch_hist, pad=(0, padding_right), mode='constant', value=0) 
        return batch_hist
    


def get_JSDiv(image1, image2, pdf):
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

    if pdf == 4:
        js_div = kl_real_fake.sum() * 0.5 + kl_fake_real.sum() * 0.5
    else:
        js_div = kl_real_fake.sum(dim=1) * 0.5 + kl_fake_real.sum(dim=1) * 0.5

    return js_div

def get_KLDiv(image1, image2, pdf):
    # for any pdf A, adding epsilon avoids 0's in the tensor A
    # to counter the addition of epsilon,
    # (A + epsilon )* 1/ 1+n*epsilon 
    epsilon = 1e-20
    real_tensor = image1 + epsilon
    fake_tensor = image2 + epsilon

    # step5 compute JS divergence = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
    kl_real_fake = (real_tensor) * ((real_tensor)/(fake_tensor)).log()

    if pdf == 4:
        kl_real_fake = kl_real_fake.sum()
    else:
        kl_real_fake = kl_real_fake.sum(dim=1)
    #print("kldiv shape:",kl_real_fake.shape)

    #print("KL real-> fake", kl_real_fake.mean())
    #print("KL fake-> real", kl_fake_real.mean())

    return kl_real_fake

def get_patches(image, k):
    kernel = k
    stride = k
    #print("calculating neighbours")
    unfold = nn.Unfold(kernel_size=(k,k), padding=0, stride=stride)
    output = unfold(image)
    swapped_ouput = torch.swapaxes(output, 1, 2)
    #print("unfold outout size:", swapped_ouput.shape)
    return swapped_ouput

def scale_image_0_1(image):
    # get r,g,b components
    tensor_image_r, tensor_image_g, tensor_image_b = get_rgb(image)

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
    
    #print("scale shape", tensor_norm_r.squeeze(0).squeeze(0).shape)
    #print("Min, Max R:", tensor_norm_r.unique(), tensor_norm_r.max(), tensor_norm_r.sum())
    #print("Min, Max G:", tensor_norm_g.min(), tensor_norm_g.max(), tensor_norm_g.sum())
    #print("Min, Max B:", tensor_norm_b.min(), tensor_norm_b.max(), tensor_norm_b.sum())

    return tensor_norm_r.squeeze(0).squeeze(0), tensor_norm_g.squeeze(0).squeeze(0), tensor_norm_b.squeeze(0).squeeze(0)

    
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

    return image.unsqueeze(0)

"""
Utility functions
"""

"""
Calculate PDF of gaussian distributions of every pixel with its neighbors
"""
def calculate_pdf_gaussian(r,g,b, kernel, sigma):
    #print("gaussian begin")
    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # step2 get the nieghbours for each channel
    neigh_r = get_neigh(r, kernel, padding)
    neigh_g = get_neigh(g, kernel, padding)
    neigh_b = get_neigh(b, kernel, padding)
    #print("in gauss: neigh r shape:", neigh_r.shape)
    # step3 build flattened repeated rgb tensor
    repeated_r = get_flattened_and_repeated(r,no_of_neigh)
    repeated_g = get_flattened_and_repeated(g,no_of_neigh)
    repeated_b = get_flattened_and_repeated(b,no_of_neigh)
    #print("in gauss: repeated neigh r shape:", repeated_r.shape)
    # step4: Xi - Xj i.e : subtract repeated_r and neigh_r

    diff_r = neigh_r - repeated_r
    diff_g = neigh_g - repeated_g
    diff_b = neigh_b - repeated_b

    diff_squared_r = torch.pow(diff_r, 2) 
    diff_squared_g = torch.pow(diff_g, 2)
    diff_squared_b = torch.pow(diff_b, 2)
    #normalised_diff_squared_sum = torch.nn.functional.normalize(diff_squared_sum, p=2.0, dim=1)
    
    sum_normalised_with_sigma_r = diff_squared_r/(2*sigma*sigma)
    sum_normalised_with_sigma_g = diff_squared_g/(2*sigma*sigma)
    sum_normalised_with_sigma_b = diff_squared_b/(2*sigma*sigma)

    gaussian_space_r = torch.exp(-sum_normalised_with_sigma_r)
    gaussian_space_g = torch.exp(-sum_normalised_with_sigma_g)
    gaussian_space_b = torch.exp(-sum_normalised_with_sigma_b)

    gaussian_nieghbourhood_sums_r = gaussian_space_r.sum(dim=2)
    gaussian_nieghbourhood_sums_g = gaussian_space_g.sum(dim=2)
    gaussian_nieghbourhood_sums_b = gaussian_space_b.sum(dim=2)

    gaussian_nieghbourhood_sums_repeated_r = gaussian_nieghbourhood_sums_r.unsqueeze(2)
    gaussian_nieghbourhood_sums_repeated_g = gaussian_nieghbourhood_sums_g.unsqueeze(2)
    gaussian_nieghbourhood_sums_repeated_b = gaussian_nieghbourhood_sums_b.unsqueeze(2)

    gaussian_distribution_r = gaussian_space_r/gaussian_nieghbourhood_sums_repeated_r
    gaussian_distribution_g = gaussian_space_g/gaussian_nieghbourhood_sums_repeated_g
    gaussian_distribution_b = gaussian_space_b/gaussian_nieghbourhood_sums_repeated_b

    return gaussian_distribution_r.squeeze(0), gaussian_distribution_g.squeeze(0), gaussian_distribution_b.squeeze(0)

"""
Calculate PDF of histogram of a neighbourhood for every pixel (overlapped neighborhood)
"""
def calculate_pdf_histogram(r, g, b, kernel):
    # get kernel size and padding
    padding = 0

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

"""
Calculate PDF of weighted histogram of a neighbourhood for every pixel (overlapped neighborhood)
weights given to neighbors as per the proximity of them.
current pixel = 1
1 hop neighbors = 1/8
2 hop neighbors = 1/16 
"""

def get_neigh_weights(neigh):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # predefined masks for computing weights of neighbours

    #print("shape of neigh is ::", neigh.shape)
    mask1 = torch.tensor([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
        ]).to(device).flatten()
    mask2 = torch.tensor([
        [-1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1,  1, -1,  1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1]
        ]).to(device).flatten()
    mask3 = torch.tensor([
        [1,  1,  1,  1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1],
        [1,  1,  1,  1, 1]
        ]).to(device).flatten()
    neigh_flatten_1 = (neigh * mask1)
    neigh_flatten_1, indices = neigh_flatten_1.sort(1, descending=True)
    neigh_flatten_1 = neigh_flatten_1[:, :1]

    neigh_flatten_2 = (neigh * mask2)
    neigh_flatten_2, indices = neigh_flatten_2.sort(1, descending=True)
    neigh_flatten_2 = neigh_flatten_2[:, :8]

    neigh_flatten_3 = (neigh * mask3)
    neigh_flatten_3, indices = neigh_flatten_3.sort(1, descending=True)
    neigh_flatten_3 = neigh_flatten_3[:, :16]

    return neigh_flatten_1, neigh_flatten_2, neigh_flatten_3

def batch_histogram_weighted(neigh):
    neigh_wt_1, neigh_wt_2, neigh_wt_3 = get_neigh_weights(neigh)
    hist1 = batch_histogram(neigh_wt_1.long())
    hist2 = batch_histogram(neigh_wt_2.long())
    hist3 = batch_histogram(neigh_wt_3.long())
    weighted_hist = hist1 + hist2/8 + hist3/16
    return weighted_hist


def calculate_pdf_weighted_hist(r, g, b, kernel):
    # get kernel size and padding
    kernel = 5
    padding = math.floor(kernel/2)

    # get the nieghbours for each channel
    neigh_r = get_neigh(r, kernel, padding).squeeze(0)
    neigh_g = get_neigh(g, kernel, padding).squeeze(0)
    neigh_b = get_neigh(b, kernel, padding).squeeze(0)

    # get histogram of neighbours
    hist_r = batch_histogram_weighted(neigh_r)
    hist_g = batch_histogram_weighted(neigh_g) 
    hist_b = batch_histogram_weighted(neigh_b) 

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

"""
Calculate PDF of whole image by simple converting the scale of values from [-1,1] to [1,0]
and dividing by sum each value
"""
def calculate_pdf_image(image):
    """
    Denormalize and scale image
    """
    im_r, im_g, im_b = scale_image_0_1(image)

    return im_r, im_g, im_b

"""
Calculate PDF of histogram of a neighbourhood for every pixel (non-overlapped neighborhood)
"""
def calculate_pdf_image_patch(r, g, b, kernel):  
    # get non overlapping patches
    im_r = get_patches(r, kernel).squeeze(0)
    im_g = get_patches(g, kernel).squeeze(0)
    im_b = get_patches(b, kernel).squeeze(0)
    
    # get histogram of neighbours
    hist_r = batch_histogram(im_r.long())
    hist_g = batch_histogram(im_g.long()) 
    hist_b = batch_histogram(im_b.long()) 

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

def get_divergence(image1, image2, pdf, klloss=2, kernel=3, patch=8, sigma=1):
    """
    Main functions 
    1. Gaussian pdf
    2. Histogram pdf
    3. Weighted histogram pdf
    4. Image pdf
    5. Image pdf for patches

    options:
    input images: input1, input2
    kernel sizes: 3,5 for 1,2,3; 8,16,.. for 4
    kl_or_js: "kl" for KL divergence; "js" for JS divergence
    """
    #get options
    #pdf = 1:gaussian,2:hist,3:wt_hist,4:imagePDF,5:patch_imagePDF,6:combination of 2,4"

    if klloss == 0:
        return 0 # when we donot want to compute divergence

    """
    Denormalize image
    """
    de_image1 = denorm(image1).to(float)
    de_image2 = denorm(image2).to(float)

    # get r,g,b components
    r1, g1, b1 = get_rgb(de_image1)
    r2, g2, b2 = get_rgb(de_image2)

    #print("shape of image received : ", image1.shape)
    #print("shape of dnorm Image : ", de_image1.shape)
    #print("shape of channel is : ", r1.shape)


    #get pdf of images
    if pdf == 1:
        prob1_r, prob1_g, prob1_b = calculate_pdf_gaussian(r1, g1, b1, kernel, sigma)
        prob2_r, prob2_g, prob2_b = calculate_pdf_gaussian(r2, g2, b2, kernel, sigma)
    elif pdf == 2:
        prob1_r, prob1_g, prob1_b = calculate_pdf_histogram(r1, g1, b1, kernel)
        prob2_r, prob2_g, prob2_b = calculate_pdf_histogram(r2, g2, b2, kernel)
    elif pdf == 3:
        prob1_r, prob1_g, prob1_b = calculate_pdf_weighted_hist(r1, g1, b1, kernel)
        prob2_r, prob2_g, prob2_b = calculate_pdf_weighted_hist(r2, g2, b2, kernel)
    elif pdf == 4:
        prob1_r, prob1_g, prob1_b = calculate_pdf_image(image1)
        prob2_r, prob2_g, prob2_b = calculate_pdf_image(image2)
    elif pdf == 5:
        prob1_r, prob1_g, prob1_b = calculate_pdf_image_patch(r1, g1, b1, patch)
        prob2_r, prob2_g, prob2_b = calculate_pdf_image_patch(r2, g2, b2, patch)

    #print("prob_r shape", prob1_r.shape)
    
    #get Divergence
    if klloss == 1:
        div_r = get_KLDiv(prob1_r, prob2_r, pdf)
        div_g = get_KLDiv(prob1_g, prob2_g, pdf)
        div_b = get_KLDiv(prob1_b, prob2_b, pdf)
    elif klloss == 2:
        div_r = get_JSDiv(prob1_r, prob2_r, pdf)
        div_g = get_JSDiv(prob1_g, prob2_g, pdf)
        div_b = get_JSDiv(prob1_b, prob2_b, pdf)

    div = div_r + div_g + div_b
    
    print("div shape", div.shape)
    return div.item()






if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #opt = TrainOptions().parse() 

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

    pdf = 2
    klloss=2
    div = get_divergence(image1.unsqueeze(0),image2.unsqueeze(0), pdf, klloss, opt)

    print("pdf=1 klloss div: ",pdf, klloss, div.mean()) 

 
