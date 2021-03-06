# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 11:17:53 2020

@author: Aditya
"""

import matplotlib.pyplot as plt

from extract2 import *
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(linewidth = 120)


torch.set_grad_enabled(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.EMNIST('./data', 
                            train = True,
                            split='balanced', 
                            download = False, 
                            transform = transforms.Compose([
                                transforms.ToTensor()]))

test_set = torchvision.datasets.EMNIST('./data', 
                            train = False,
                            split='balanced', 
                            download = False, 
                            transform = transforms.Compose([
                                transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 100)

def get_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.to(device)).sum().item()

def linear(in_ft, out_ft):
    return nn.Linear(in_features = in_ft, out_features = out_ft)

def Conv(in_ch, out_ch, ks):
    return nn.Conv2d(in_channels = in_ch ,out_channels = out_ch ,kernel_size = ks)



class Network(nn.Module):
    def __init__(self, in_c, n_classes):
        super(Network, self).__init__()
        
        self.conv1 = nn.Sequential(
                        Conv(in_c,6,5),
                        nn.ReLU(),
                        )
        
        self.conv2 = nn.Sequential(
                        Conv(6,12,5),
                        nn.ReLU()
                        )
    
        self.fc = nn.Sequential(
                        linear(12*4*4,192),
                        linear(192,120),
                        linear(120,n_classes)
                        )
        
    def forward(self, t):
        t = self.conv1(t)
        t = F.max_pool2d(t,2,2) #kernel size, stride 
        t = self.conv2(t)
        t = F.max_pool2d(t,2,2)
        t = t.reshape(-1,12*4*4)
        t = self.fc(t)


        return t
        
        
def accuracy(correct, total):
    return correct / len(total)

@torch.no_grad()
def get_all(model, loader):
    all_preds = torch.tensor([]).to(device)
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        
        all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds

@torch.no_grad()
def get_all_OCR(model, image_batch):
    all_preds = torch.tensor([]).to(device)


    images = image_batch.to(device)

    preds = model(images)
        
    all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds

network = Network(1,47)
network.cuda()
optimizer = optim.Adam(network.parameters(), lr = 0.001)

network = Network(1,47)
network.cuda()
network.load_state_dict(torch.load('./learned001.pth'))


class OCR:
    def __init__(self,path,key):
        self.path = path
        self.key = key
        
    def get_image_predictions(self):
        image = skimage.img_as_float(skimage.io.imread(self.path))
        bboxes, bw = findLetters(image)
        images_cropped = cropImage(bboxes, image)
        images_new = images_cropped
        for images in images_new:
            images[images <= np.min(images)] = 0 
            images[images > 0] = 1
            

        OCR_loader = torch.FloatTensor(images_new)
        OCR_preds = get_all_OCR(network, OCR_loader)
        preds_OCR_max = [torch.argmax(preds) for preds in OCR_preds]
        

        translated = [[self.key[preds.item()] for preds in preds_OCR_max]]
        
        return bboxes, translated
        
    @staticmethod 
    def plot_image(bboxes, translated):
        paired = []
        translated = np.squeeze(translated)
        for t, b in zip(translated, bboxes):
            paired.append([t,b])
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        for bbox in bboxes:
            y1, x1, y2, x2 = bbox
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                          fill=False, edgecolor='black', linewidth=1)       
            
            ax.add_patch(rect) 
        
        # ax.imshow(image)
            
        for pair in paired:
            # print(pair[1][1]+5,pair[1][0]-15,pair[0])
            ax.text(pair[1][1]+5,pair[1][0]-15,pair[0],fontsize = 15)
        
        plt.imshow(image)
    
if __name__ == "__main__":
    
    letters_EMNIST = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
         40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}
    
    OCmodel = OCR('02_letters.jpg',letters_EMNIST)
    bboxes, translated = OCmodel.get_image_predictions()
    OCmodel.plot_image(bboxes, translated)
    
    