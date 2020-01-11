# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:04:01 2019

@author: Aditya
"""
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.transform import rescale, resize, downscale_local_mean

import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import numpy as np
from skimage.util import pad

def findLetters(image):
    bboxes = []
    bw = None
    #height, width = image.shape[0], image.shape[1]
  
  
   

    gray = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(gray, sigma=2)
    thresh = skimage.filters.threshold_otsu(blur)
    img_thresh = skimage.morphology.closing(blur < thresh, skimage.morphology.square(3))
    borders = skimage.segmentation.clear_border(img_thresh)
    label_image = skimage.measure.label(borders)

    image_label_overlay = skimage.color.label2rgb(borders*1, image=gray) 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    total_area = 0
    for region in skimage.measure.regionprops(label_image):
        total_area += region.area
    mean_area = total_area / len(skimage.measure.regionprops(label_image))
    
    for region in skimage.measure.regionprops(label_image):
        
        if region.area >= mean_area*0.55:
            minr, minc, maxr, maxc = region.bbox
            print(region.bbox)
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='black', linewidth=1)
          
            x2 = maxc
            y2 = minr + (maxr - minr)
            
            x1 = minc
            y1 = minr
     
            
            bboxes.append(np.array([y1, x1, y2, x2]))
            ax.add_patch(rect)            
        


    bw = gray
    

    
    return bboxes, bw

def preprocesss(image):
    pass

def getRows(image):
    pass

def squareIT(M, val):
    (a, b) = M.shape
    if a != b:
        padding = ((abs(a-b)//2, abs(a-b)//2), (abs(a-b)//2, abs(a-b)//2))
    return np.pad(M, padding, mode = 'constant',constant_values = val)

def cropImage(bboxes, image):
    
    images_cropped = []

    gray = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(gray, sigma=2.3)
    thresh = skimage.filters.threshold_otsu(blur)
    img_thresh = skimage.morphology.closing(blur < thresh, skimage.morphology.square(1))
    borders = skimage.segmentation.clear_border(img_thresh)
    borders = borders*1

    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        
        # test = borders[y1:y2, x1:x2]
        test_final = borders[y1-10:y2+10, x1-10:x2+10] #remove later
        padded_im = squareIT(test_final, 0 )
        # test_final = np.pad(test_final, (test_final.shape[0]//4,test_final.shape[1]//4),'minimum')
        test_final = padded_im.T
            

        test_final = resize(test_final, (28,28), anti_aliasing = False)

        test_final = np.expand_dims(test_final,0)
        images_cropped.append(test_final)
    
        
    return images_cropped

def toPyTensor(images):
    pass
