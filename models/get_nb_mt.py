# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import torch
from threading import Thread

def get_neighb_list(image, label, slice_no):
    print("----------------------getting neighbours list-------------------------")
    
    shape_i, shape_j, shape_k  = image.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    label = label.to(device)
    features = torch.zeros((0,6), dtype=torch.float32).to(device)
    labels = torch.zeros((0,1), dtype=torch.float32).to(device)
    for i, x in enumerate(image):
        if i == slice_no:
            #print("x1", x.shape)
            #print("----------------------computing neighbours")
            for j, x in enumerate(x):
                #print("x2", x.shape)
                for k, x in enumerate(x):
                    #print("-->>---x3------------------", x)
                    feature = torch.zeros((0), dtype=torch.float32).to(device)
                    fr = torch.zeros((1), dtype=torch.float32)
                    up = torch.zeros((1), dtype=torch.float32)
                    lf = torch.zeros((1), dtype=torch.float32)
                    rt = torch.zeros((1), dtype=torch.float32)
                    dw = torch.zeros((1), dtype=torch.float32)
                    bc = torch.zeros((1), dtype=torch.float32)
                    
                    if i == 0:
                      fr = fr.to(device)
                      #print("shape fr:", fr, fr.shape)
                    else:
                      fr = torch.unsqueeze(image[i-1,j,k], 0)
                      #print("shape fr:", fr, fr.shape)
                      
                    feature = torch.cat((feature, fr), 0)
                    
                    if j == 0:
                      up = up.to(device)
                      #print("shape up:", up , up.shape)
                    else:
                      up = torch.unsqueeze(image[i,j-1,k], 0)
                      #print("shape up:", up, up.shape)
                      
                    feature = torch.cat((feature, up), 0)
                    
                    if k == 0:
                      lf = lf.to(device)
                      #print("shape lf:", lf, lf.shape)
                    else:
                      lf = torch.unsqueeze(image[i,j,k-1], 0)
                      #print("shape lf:", lf, lf.shape)
                      
                    feature = torch.cat((feature, lf), 0)
                    
                    if k+1 > shape_j-1:
                      rt = rt.to(device)
                      #print("shape rt:", rt, rt.shape)
                    else:
                      rt = torch.unsqueeze(image[i,j,k+1], 0)
                      #print("shape rt:", rt, rt.shape)
                    
                    feature = torch.cat((feature, rt), 0)
                    
                    if j+1 > shape_j-1:
                      dw = dw.to(device)
                      #print("shape dw:", dw, dw.shape)
                    else:
                      dw = torch.unsqueeze(image[i,j+1,k], 0)
                      #print("shape dw:", dw, dw.shape)
                      
                    feature = torch.cat((feature, dw), 0)
                    
                    if i+1 > shape_i-1:
                      bc = bc.to(device)
                      #print("shape bc:", bc, bc.shape)
                    else:
                      bc = torch.unsqueeze(image[i+1, j, k], 0)
                      #print("shape bc:", bc, bc.shape)
                      
                    feature = torch.cat((feature, bc), 0)
                    feature = torch.unsqueeze(feature, 0)
                    #print("feature:", feature, feature.shape)
                    #feature = torch.cat((fr, up, lf, rt, dw, bc))                
                    features = torch.cat((features, feature),0)
                    labels = torch.cat((labels, label), 0)
    
    #print("features, labels shape :", features.shape, labels.shape)
    
    print("returning list")
    return features, labels

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return

def neigh_mt(image, label):
    image = torch.squeeze(image)
    print("image:", image.shape)
    m1 = image
    m2 = image
    m3 = image
    
    # creating thread
    t0 = ThreadWithReturnValue(target=get_neighb_list, args=(m1,label, 0))
    t1 = ThreadWithReturnValue(target=get_neighb_list, args=(m2,label, 1))
    t2 = ThreadWithReturnValue(target=get_neighb_list, args=(m3,label, 2))
  
    t0.start()
    t1.start()
    t2.start()
  
	# wait until thread 1 is completely executed
    f1, l1 = t0.join()
    f2, l2 = t1.join()
    f3, l3 = t2.join()
    
    f = torch.cat((f1, f2, f3), dim =0)
    l = torch.cat((l1, l2, l3), dim =0)
    print("done threading !")
    
    return f, l
    
    