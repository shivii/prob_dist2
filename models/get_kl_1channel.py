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
    print("batch hist shape",data_tensor.shape)
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

    epsilon = 1e-20
    # step 1: convert it to [0 ,2]
    tensor_image = image +1

    # step 2: convert it to [0 ,1]
    tensor_image = (tensor_image / 2 + epsilon) / (1/(1+256*256*epsilon))

    #tensor_image_0_1r = tensor_image_r / (tensor_image_r.max() - tensor_image_r.min())
    #tensor_image_0_1g = tensor_image_g / (tensor_image_g.max() - tensor_image_g.min())
    #tensor_image_0_1b = tensor_image_b / (tensor_image_b.max() - tensor_image_b.min())

    tensor_image_sum = tensor_image.sum()
    tensor_norm = tensor_image /tensor_image_sum

    #print("scale shape", tensor_norm_r.squeeze(0).squeeze(0).shape)
    #print("Min, Max R:", tensor_norm_r.unique(), tensor_norm_r.max(), tensor_norm_r.sum())
    #print("Min, Max G:", tensor_norm_g.min(), tensor_norm_g.max(), tensor_norm_g.sum())
    #print("Min, Max B:", tensor_norm_b.min(), tensor_norm_b.max(), tensor_norm_b.sum())

    return tensor_norm.squeeze(0).squeeze(0)

    
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
def calculate_pdf_gaussian(image, kernel, sigma):
    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # step2 get the nieghbours for each channel
    neigh = get_neigh(image, kernel, padding)

    #print("in gauss: neigh r shape:", neigh_r.shape)
    # step3 build flattened repeated rgb tensor
    repeated = get_flattened_and_repeated(image,no_of_neigh)

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
Calculate PDF of histogram of a neighbourhood for every pixel (overlapped neighborhood)
"""
def calculate_pdf_histogram(image, kernel):
    # get kernel size and padding
    padding = 0

    # get the nieghbours for each channel
    neigh = get_neigh(image, kernel, padding).squeeze(0)

    # get histogram of neighbours
    hist = batch_histogram(neigh.long())

    #getting sum along dim 2
    neigh_sum = hist.sum(1)

    #repeat the sum values along every dim
    neigh_repeat = neigh_sum.unsqueeze(1).repeat(1, 256)

    # probability = current value/sum
    prob = hist / neigh_repeat

    return prob

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
    print("neigh shape", neigh.shape)
    neigh_wt_1, neigh_wt_2, neigh_wt_3 = get_neigh_weights(neigh)
    hist1 = batch_histogram(neigh_wt_1.long())
    hist2 = batch_histogram(neigh_wt_2.long())
    hist3 = batch_histogram(neigh_wt_3.long())
    weighted_hist = hist1 + hist2/8 + hist3/16
    return weighted_hist


def calculate_pdf_weighted_hist(image, kernel):
    # get kernel size and padding
    kernel = 5
    padding = math.floor(kernel/2)

    # get the nieghbours for each channel
    neigh = get_neigh(image, kernel, padding).squeeze(0)

    # get histogram of neighbours
    hist = batch_histogram_weighted(neigh)

    #getting sum along dim 2
    neigh_sum = hist.sum(1)

    #repeat the sum values along every dim
    neigh_repeat = neigh_sum.unsqueeze(1).repeat(1, 256)

    # probability = current value/sum
    prob = hist / neigh_repeat

    return prob

"""
Calculate PDF of whole image by simple converting the scale of values from [-1,1] to [1,0]
and dividing by sum each value
"""
def calculate_pdf_image(image):
    """
    Denormalize and scale image
    """
    im = scale_image_0_1(image)

    return im

"""
Calculate PDF of histogram of a neighbourhood for every pixel (non-overlapped neighborhood)
"""
def calculate_pdf_image_patch(image, kernel):  
    # get non overlapping patches
    im = get_patches(image, kernel).squeeze(0)

    # get histogram of neighbours
    hist = batch_histogram(im.long())
        
    #getting sum along dim 2
    neigh_sum = hist.sum(1)  

    #repeat the sum values along every dim
    neigh_repeat = neigh_sum.unsqueeze(1).repeat(1, 256)

    # probability = current value/sum
    prob = hist / neigh_repeat

    return prob

def get_pdf(image, target_is_real, pdf, kernel=3, patch=8, sigma=1):
    """ Initialize the GANLoss class.
    Parameters:
        prediction (tensor) - - tpyically the prediction from a discriminator
        target_is_real (bool) - - if the ground truth label is for real images or fake images
        PDF:
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
    
    #get pdf of image
    if pdf == 1:
        prob = calculate_pdf_gaussian(image, kernel, sigma)
    elif pdf == 2:
        prob = calculate_pdf_histogram(image, kernel)
    elif pdf == 3:
        prob = calculate_pdf_weighted_hist(image, kernel)
    elif pdf == 4:
        prob = calculate_pdf_image(image)        
    elif pdf == 5:
        prob = calculate_pdf_image_patch(image, patch)

   
    return pdf




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
    #print("img shape",img1.shape, img2.shape)



    """
    Normalise image
    """
    trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                ])
    image1 = trans(img1).to(device)
    image2 = trans(img2).to(device)

    pdf = 1
    klloss = 2
    div = get_adversarial_loss(img1.unsqueeze(0).to(device),True, pdf)

    print("pdf=1 klloss div: ",pdf, klloss, div) 
    