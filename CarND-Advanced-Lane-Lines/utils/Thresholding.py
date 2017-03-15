import numpy as np
import cv2

class ImageColorThres():
    def __init__(self, img, vertices):
        self.img = self.region_of_interest(img, vertices)
        self.H, self.L, self.S = self.RGB2HLS()
        print('color thresholding class2')

    def RGB2HLS(self):
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS) # decompose
        return hls[:,:,0], hls[:,:,1], hls[:,:,2]
    
    def thresholding_onechannel(self, channel, th_min=0, th_max=255):
        # decompose
        if channel == 'H':
            target_channel_img = self.H
        elif channel == 'L':
            target_channel_img = self.L
        elif channel == 'S':
            target_channel_img = self.S
        
        # combine
        binary = np.zeros_like(target_channel_img)
        binary[(target_channel_img >= th_min)&(target_channel_img <= th_max)] = 1
        
        return binary
    
    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        # filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

class ImageGradientThres():
    def __init__():
        print('Gradient thresholding class')