import numpy as np
import cv2

class PerspectiveTransform():
    def __init__(self, img):
        # source and destination points
        self.src = None
        self.dst = None
        
        self.img_size = (img.shape[1], img.shape[0])
        self.src, self.dst = self.set_src_dst()

        # perspective transform
        self.M = None
        self.Minv = None

        self.M, self.Minv = self.get_perspective_transform()

    def set_src_dst(self, offset=200):
        img_size = self.img_size
        offset = int(offset)
        src_coordinate = [[580,460],[710,460],[1150,720],[150,720]]
        src = np.float32(src_coordinate)

        dst = np.float32([[offset, 0], 
                  [img_size[0]-offset, 0], 
                  [img_size[0]-offset, img_size[1]], 
                  [offset, img_size[1]]])
        return src, dst

    def get_perspective_transform(self):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        return M, Minv

    def warp_image(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size)

    def unwarp_image(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_size)