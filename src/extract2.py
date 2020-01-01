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


    image_label_overlay = skimage.color.label2rgb(label_image, image=gray)

 
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
          
            y2 = minr + (maxr - minr)
            x1 = minc
            
            y1 = minr
            x2 = maxc
            
            bboxes.append(np.array([y1, x1, y2, x2]))
            ax.add_patch(rect)            
        

    bw = gray
    

    
    return bboxes, bw 

#image = skimage.img_as_float(skimage.io.imread('04_deep.jpg'))


#bboxes, bw = findLetters(image)

def getRows(image):
    pass

def cropImage(bboxes, image):
    
    images_cropped = []
    # bboxes, _ = findLetters(images)
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox
        test = image[y1:y2, x1:x2,0]
        test_resized = resize(test.T, (28,28), anti_aliasing = True)
        #test_final = pad(test_resized,(2,2),'constant',constant_values = (1,1))
        test_final = test_resized
        test_final = np.expand_dims(test_final,0)
        images_cropped.append(test_final)
        # images.append(test_resized)
        
    return images_cropped

def toPyTensor(images):
    pass

# images = cropImage(bboxes)