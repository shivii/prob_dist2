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
    neigh_hist = torch.zeros(0,bins)
    img_r, img_c = img.shape
    for i in range(img_r):
        hist_r = img[i, :]
        hist = torch.histc(hist_r, bins = bins, min=0, max=255).cpu()
        #print(hist.shape)
        neigh_hist = torch.cat((neigh_hist.cpu(), hist.unsqueeze(0)), 0)
    return neigh_hist    

def calculate_probability_distribution_histogram(image, sigma, kernel, bins):
    #print("gaussian begin")
    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # step1 get the nieghbours for each channel
    neigh = get_neigh(image, kernel, padding).squeeze(0)
    print_with_time("neighbour shapes", neigh.shape)

    # step2 get histogran of neighbours
    hist = get_hist(neigh, bins)    
    print_with_time("histogram shapes", hist.shape)

    # step3 get PDF
    squared_sum = torch.pow(hist, 2) 
    sum_normalised_with_sigma = squared_sum/(2*sigma*sigma)
    gaussian_space = torch.exp(-sum_normalised_with_sigma).unsqueeze(0)
    print_with_time("gaussian space", gaussian_space.shape)
    
    gaussian_nieghbourhood_sums = gaussian_space.sum(dim=2)
    gaussian_nieghbourhood_sums_repeated = gaussian_nieghbourhood_sums.unsqueeze(2)
    gaussian_distribution = gaussian_space/gaussian_nieghbourhood_sums_repeated

    print(gaussian_distribution.shape)
    
    return gaussian_distribution

def hist_divergence(image1, image2, sigma, kernel, bins):
    #step1 get images as floats
    image_p = image1.to(torch.float32)
    image_q = image2.to(torch.float32)

    #step2 get pdf of histograms of neighbours
    hist_p = calculate_probability_distribution_histogram(image_p, sigma, kernel, bins)
    hist_q = calculate_probability_distribution_histogram(image_q, sigma, kernel, bins)
    print("hist shape:", hist_p.shape, hist_q.shape)
    
    # for any pdf A, adding epsilon avoids 0's in the tensor A
    # to counter the addition of epsilon,
    # (A + epsilon )* 1/ 1+n*epsilon 
    epsilon = 1e-20
    real_tensor = (hist_p + epsilon) * (1/(1 + bins * epsilon))
    fake_tensor = (hist_q + epsilon) * (1/(1 + bins * epsilon))

    # step4 joint distribution of 2 tensors
    m = (real_tensor + fake_tensor)/2

    # step5 compute JS divergence = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)
    kl_real_fake = (real_tensor) * ((real_tensor)/(m)).log()
    kl_fake_real = (fake_tensor) * ((fake_tensor)/(m)).log()

    get_details(kl_real_fake,"kl_divergence_elements")

    kl_per_pixel_real_fake = kl_real_fake.sum(dim=1)
    kl_per_pixel_real_fake[kl_per_pixel_real_fake < 0] = 0

    kl_per_pixel_fake_real = kl_fake_real.sum(dim=1)
    kl_per_pixel_fake_real[kl_per_pixel_fake_real < 0] = 0

    js_div = 0.5 * kl_per_pixel_real_fake + 0.5 * kl_per_pixel_fake_real

    print("js_div", js_div.mean())
    #return kl_divergence_per_pixel


"""
n is the number of repetitions
"""
def get_flattened_and_repeated(t, n):
    #print("genrating a flattened and repeated tensor")
    return torch.flatten(t).unsqueeze(1).repeat(1,1,n)


def calculate_gaussian_distribution(image, sigma, kernel):
    #print("gaussian begin")

    no_of_neigh = kernel*kernel
    padding = math.floor(kernel/2)

    # step1 get r g b seperately for the image
    r,g,b = get_rgb(image)

    # step2 get the nieghbours for each channel
    neigh_r = get_neigh(r, kernel, padding)
    neigh_g = get_neigh(g, kernel, padding)
    neigh_b = get_neigh(b, kernel, padding)

    # step3 build flattened repeated rgb tensor
    repeated_r = get_flattened_and_repeated(r,no_of_neigh)
    repeated_g = get_flattened_and_repeated(g,no_of_neigh)
    repeated_b = get_flattened_and_repeated(b,no_of_neigh)

    # step4: Xi - Xj i.e : subtract repeated_r and neigh_r

    diff_r = neigh_r - repeated_r
    diff_g = neigh_g - repeated_g
    diff_b = neigh_b - repeated_b

    diff_squared_sum = torch.pow(diff_r, 2) + torch.pow(diff_g, 2) + torch.pow(diff_b, 2)
    #normalised_diff_squared_sum = torch.nn.functional.normalize(diff_squared_sum, p=2.0, dim=1)
    
    sum_normalised_with_sigma = diff_squared_sum/(2*sigma*sigma)
    gaussian_space = torch.exp(-sum_normalised_with_sigma)
    gaussian_nieghbourhood_sums = gaussian_space.sum(dim=2)
    gaussian_nieghbourhood_sums_repeated = gaussian_nieghbourhood_sums.unsqueeze(2)
    gaussian_distribution = gaussian_space/gaussian_nieghbourhood_sums_repeated
    return gaussian_distribution

"""
Predefined method of Pytorch
"""

def calculate_divergence(real, fake):
    #get_details(real, "real")
    #get_details(fake, "fake")
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    output = kl_loss(fake, real)
    return output

"""
Computing kl divergence of respective neighbours of real and fake images
"""

def calculate_divergence_per_neighbourhood(real, fake):
    epsilon = 1e-20
    real_tensor = real + epsilon
    fake_tensor = fake + epsilon
    #print("epsilon is ", epsilon)
    kl_divergence_elements = (fake_tensor) * ((fake_tensor)/(real_tensor)).log()
    #get_details(kl_divergence_elements,"kl_divergence_elements")
    kl_divergence_per_pixel = kl_divergence_elements.sum(dim=1)
    kl_divergence_per_pixel[kl_divergence_per_pixel < 0] = 0

    return kl_divergence_per_pixel

def get_details(t, label):
    print(label, " shape ", t.shape)
    print(label, " minimum ", t.min())
    print(label, " maximum ", t.max())

def plot(t, label):
    max = 255
    min = 0
    dataset = torch.flatten(t).cpu()
    hist = torch.histc(dataset, bins = 255, min=min, max=max)

    x = range(100)
    plt.bar(x, hist, align='center')
    plt.xlabel(label)
    plt.show()

"""
kl divergence is used for computing the differnce between 2 probability distributions !
A probability distribution should have summation = 1 other wise its not a probability distribution.
so this needs to be neccessarily checked.
"""

def test_is_pdf(t):
    sumt = t.sum(dim=2)
    get_details(sumt, "sumt")
    return torch.all(sumt == 1, dim=1)


def get_divergence(image1, image2, sigma, kernel):
    image1 = image1.to(torch.float32)
    image2 = image2.to(torch.float32)
    gaussian_distribution1 = calculate_gaussian_distribution(image1, sigma, kernel)
    gaussian_distribution2 = calculate_gaussian_distribution(image2, sigma, kernel)

    return calculate_divergence_per_neighbourhood(gaussian_distribution1.squeeze(0), gaussian_distribution2.squeeze(0))

def get_JSdivergence(image1, image2, sigma, kernel):
    image_p = image1.to(torch.float32)
    image_q = image2.to(torch.float32)

    gaussian_distribution_1 = calculate_gaussian_distribution(image_p, sigma, kernel)
    gaussian_distribution_2 = calculate_gaussian_distribution(image_q, sigma, kernel)

    m = (gaussian_distribution_1 + gaussian_distribution_2)/2

    div_p_m = calculate_divergence_per_neighbourhood(gaussian_distribution_1.squeeze(0), m.squeeze(0))
    div_q_m = calculate_divergence_per_neighbourhood(gaussian_distribution_2.squeeze(0), m.squeeze(0))
    js_div = 0.5 * div_p_m + 0.5 * div_q_m

    return js_div













if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    torch.set_printoptions(threshold=100000)

    transform = transforms.Compose([
    transforms.PILToTensor()
    ])

    input_image1 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_real_B.png")).to(device).unsqueeze(0).to(torch.float32)
    input_image2 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1160_rec_B.png")).to(device).unsqueeze(0).to(torch.float32)

    input_image3 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1920_real_A.png")).to(device).unsqueeze(0).to(torch.float32)
    input_image4 = transform(Image.open("/home/apoorvkumar/shivi/Phd/Project/patch_TSNE/prob_dist2/test_runners/1920_rec_A.png")).to(device).unsqueeze(0).to(torch.float32)

    hist_divergence(input_image3,input_image4, 1, kernel=3, bins=30)

    """
    #gaussian_distribution1 = calculate_gaussian_distribution(input_image1, 0.1, kernel=5)
    #gaussian_distribution2 = calculate_gaussian_distribution(input_image2, 0.1, kernel=5)

    #gaussian_distribution3 = calculate_gaussian_distribution(input_image3, 0.1, kernel=5)
    #gaussian_distribution4 = calculate_gaussian_distribution(input_image4, 0.1, kernel=5)
    #print(test_is_pdf(gaussian_distribution1))
    #print(test_is_pdf(gaussian_distribution1))

    

    kl_div = get_divergence(input_image1, input_image1, 0.8, kernel=5)
    exp = 0.5
    js_div = get_JSdivergence(input_image1, input_image1, 0.8, kernel=5)

    loss_g = kl_div - 2*js_div + math.log(4) + exp
    print(kl_div.mean())
    print("loss", loss_g.mean())
    print("JS div", (js_div * 2 - math.log(4)).mean())
    
    #KL_divergence = calculate_divergence(gaussian_distribution1.squeeze(0), gaussian_distribution2.squeeze(0))
    #print("average kl divergence",KL_divergence)
    #divergence_per_neighbourhood_1 = calculate_divergence_per_neighbourhood(gaussian_distribution1.squeeze(0), gaussian_distribution2.squeeze(0))
    #divergence_per_neighbourhood_2 = calculate_divergence_per_neighbourhood(gaussian_distribution2.squeeze(0), gaussian_distribution1.squeeze(0))
    #print("average kl divergence 1 ", (divergence_per_neighbourhood_1.mean()), (divergence_per_neighbourhood_2.mean()))
    
    #KL_divergence = calculate_divergence(gaussian_distribution3.squeeze(0), gaussian_distribution4.squeeze(0))
    #print("average kl divergence",KL_divergence)
    #divergence_per_neighbourhood_1 = calculate_divergence_per_neighbourhood(gaussian_distribution3.squeeze(0), gaussian_distribution4.squeeze(0))
    #divergence_per_neighbourhood_2 = calculate_divergence_per_neighbourhood(gaussian_distribution4.squeeze(0), gaussian_distribution3.squeeze(0))
    #print("average kl divergence 2 ", (divergence_per_neighbourhood_1.mean()), (divergence_per_neighbourhood_2.mean()))


    
    #get_details(divergence_per_neighbourhood, "divergence_per_neighbourhood")
    diver_numpy = divergence_per_neighbourhood.cpu().numpy()
    gaussian_numpy1 = gaussian_distribution1.cpu().numpy()
    gaussian_numpy2 = gaussian_distribution2.cpu().numpy()
    #print(gaussian_numpy1.shape)
    #print(gaussian_numpy2.shape)

    
    count_of_neg=0
    list_of_neg = []
    for i in range(len(diver_numpy)):
        if diver_numpy[i] < 0:
            count_of_neg = count_of_neg + 1
    #        print("divergence is:", diver_numpy[i])
    #        print("gaussian1:", gaussian_distribution1[0][i])
    #        print("gaussian2:", gaussian_distribution2[0][i])

    #print("count:", count_of_neg)
    #print("average kl divergence", divergence_per_neighbourhood.mean())
    



kl divergence is used for computing the differnce between 2 probability distributions !
A probability distribution should have summation = 1 other wise its not a probability distribution.
so this needs to be neccessarily checked.
"""