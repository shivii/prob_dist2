# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
        #                        512, 512, 'M', 512, 512, 512, 512, 'M']
        #vgg19_layers = get_vgg_layers(vgg19_config, batch_norm=True)
        model = models.vgg19()
        
        #print(model)
        model.load_state_dict(torch.load("vgg19_24_06_hor2ze.pt"))
        self.vgg_ft = model.features.eval().to(self.device)
        #self.vgg_ft = model.features.eval()
        self.avgpool = nn.AdaptiveAvgPool2d(7).to(self.device)
        #self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.vgg_fcLayers = nn.Sequential(
                                *list(model.classifier.children())[:-3]).to(self.device)
        #self.vgg_fcLayers = nn.Sequential(
        #                         *list(model.classifier.children())[:-3])
        self.loss = nn.MSELoss()

        for param in self.vgg_ft.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        for param in self.vgg_fcLayers.parameters():
            param.requires_grad = False

    def forward(self, input):
        x = self.vgg_ft(input)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.vgg_fcLayers(h)

        return x