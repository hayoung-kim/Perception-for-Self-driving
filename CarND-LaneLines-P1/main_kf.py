#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def redscale(img, red_thres = 200):
    reds_thresholds = (img[:,:,0] < red_thres)
    color_select = np.copy(img)
    color_select[reds_thresholds] = [0,0,0]
    return color_select

def get_line_hough(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines_ = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines_


def sample_line(line, number_of_samples = 5):
    for x1,y1,x2,y2 in line:
        lam = np.random.uniform(0,1,number_of_samples)
        xs = lam * (x2-x1) + x1
        ys = lam * (y2-y1) + y1
        xys = np.c_[xs.T, ys.T]
    return xys

def get_line_param(line):
    for x1,y1,x2,y2 in line:
        slope_ = (y2-y1)/(x2-x1)
        length_ = math.sqrt((x1-x2)**2 + (y2-y1)**2)
    return (slope_, length_)

def get_ls_solution_line(xy_samples_):
    x_ = xy_samples_[:,0]
    y_ = xy_samples_[:,1]
    A_ = np.vstack([x_, np.ones(len(x_))]).T
    m_, c_ = np.linalg.lstsq(A_, y_)[0]
    return (m_, c_)

def get_ransac_solution_line(xy_samples_, max_iteration = 40, sample_size = 4):
    
#     x_ = xy_samples_[:,0]
#     y_ = xy_samples_[:,1]
#     data = np.vstack([x_, y_]).T
    goal_inliers = np.ceil(len(xy_samples_) * 0.7)
#     print('goal inliers: '+ str(goal_inliers))
#     print(sample_size)
    m,b = run_ransac(list(xy_samples_), estimate, lambda x,y: is_inlier(x, y, 0.09), goal_inliers, max_iteration, sample_size)
    a,b,c = m
    
    m_ = -a/b
    c_ = -c/b
    return (m_,c_)

def get_line_image_from_line(m_list, c_list, img, roi_vertices):
    # get (x1,y1,x2,y2) from y=mx+c
    imshape = img.shape
    lines = np.zeros((2,4), dtype=np.uint32)

    for i in range(2):
        m = m_list[i]
        c = c_list[i]
        
#         print('m: ', m)
#         print('c: ', c)
        
        y1 = imshape[0]
        y2 = imshape[0]*3/5
        
        x1 = (y1-c)/m
        x2 = (y2-c)/m
        
#         x1 = int(x1)
#         x2 = int(x2)
#         y1 = int(y1)
#         y2 = int(y2)
#         print('y1: ', y1, 'y2: ', y2)
#         lines = np.append(lines, np.array([[x1,y1, x2,y2]]), axis=0)
        lines[i,:] = np.array([x1,y1,x2,y2])
        
#     print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, [lines],color=[255, 0, 0], thickness=5)
    return region_of_interest(line_img, roi_vertices)

from numpy import dot
from numpy import dot, sum, tile, linalg 
from numpy.linalg import inv 

def gauss_pdf(X, M, S):     
    if M.shape()[1] == 1:         
        DX = X - tile(M, X.shape()[1])           
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)         
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))         
        P = exp(-E)     
    elif X.shape()[1] == 1:         
        DX = tile(X, M.shape()[1])- M           
        E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)         
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))         
        P = exp(-E)     
    else:         
        DX = X-M           
        E = 0.5 * dot(DX.T, dot(inv(S), DX))         
        E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))         
        P = exp(-E)     
    return (P[0],E[0])

def kf_predict(X, P, A, Q, B, U):     
    X = dot(A, X) + dot(B, U)     
    P = dot(A, dot(P, A.T)) + Q     
    return(X,P) 

def kf_update(X, P, Y, H, R):     
    IM = dot(H, X)     
    IS = R + dot(H, dot(P, H.T))     
    K = dot(P, dot(H.T, inv(IS)))     
    X = X + dot(K, (Y-IM))     
    P = P - dot(K, dot(IS, K.T))     
    LH = gauss_pdf(Y, IM, IS)     
    return (X,P,K,IM,IS,LH)


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    imshape = image.shape
    reds = redscale(image, 200)
    blur_reds = gaussian_blur(reds, 5)

    edges = canny(blur_reds, 65, 150)
    
    vertices = np.array([[(0,imshape[0]), (imshape[1]*7/15, imshape[0]*3/5), (imshape[1]*8/15, imshape[0]*3/5), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # hough transform
    rho_res = 1
    theta_res = np.pi/180
    vote_thres = 5
    min_line_len = 10
    max_line_gap = 10
    line_image = hough_lines(masked_edges, rho_res, theta_res, vote_thres, min_line_len, max_line_gap)

    # sampling the lane lines for each right/left lane
    lines_ = get_line_hough(masked_edges, rho_res, theta_res, vote_thres, min_line_len, max_line_gap)
    right_xy_samples = np.empty((0,2),int)
    left_xy_samples = np.empty((0,2),int)

    for line in lines_:
        slope_, length_ = get_line_param(line)
        n_samples = np.maximum(np.ceil(length_/20), 2)
        xy_samples = sample_line(line, n_samples)

        if (slope_) > 0 and np.absolute(slope_) < 60 * np.pi/180 and np.absolute(slope_) > 30 * np.pi/180:
            if np.all(xy_samples[:,0] >= imshape[1]/2):
                # reasonable right lane lines
                right_xy_samples = np.append(right_xy_samples, xy_samples, axis=0)
            
        elif (slope_) < 0 and np.absolute(slope_) < 60 * np.pi/180 and np.absolute(slope_) > 30 * np.pi/180:
            if np.all(xy_samples[:,0] <= imshape[1]/2):
                # reasonable left lane lines
                left_xy_samples = np.append(left_xy_samples, xy_samples, axis=0)
    
    # line approximation: y= mx + c (least square solution)
    m_right, c_right = get_ls_solution_line(right_xy_samples)
    m_left,  c_left  = get_ls_solution_line(left_xy_samples)
    
    # kalman filter
    #time step of mobile movement 
    dt = 0.1 
    
    # Initialization of state matrices 
    X = np.array([0,0]) 
    P = np.diag((100,100)) 
    A = np.array([[1, dt], [0, 1]]) 
    Q = np.eye(X.shape()[0]) 
    B = np.eye(X.shape()[0]) 
    U = np.zeros((X.shape()[0],1)) 

    # Measurement matrices 
    Y = np.array([[m_right]]) 
    H = array([[1, 0]]) 
    R = eye(Y.shape()[0]) 

    (X,P,K,IM,IS,LH) = kf_update(X, P, Y, H, R)
    (X,P) = kf_predict(X, P, A, Q, B, U)
    
    m_right = X[0,0]

    # draw lane line
    lane_line_img = get_line_image_from_line([m_right, m_left], [c_right, c_left], image, vertices)

    # make image for showing result
    weighted_image = weighted_img(lane_line_img, image, 0.8, 1, 0)
    
    for xys in right_xy_samples:
        cv2.circle(lane_line_img,(int(xys[0]), int(xys[1])), 5, (0,0,255), -1)
        
    for xys in left_xy_samples:
        cv2.circle(lane_line_img,(int(xys[0]),int(xys[1])), 5, (0,255,0), -1)
                   
    result = weighted_img(lane_line_img, image, 0.85, 1, 0)
    
    return result

white_output = '/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/white2.mp4'
clip1 = VideoFileClip("/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
print('white')

yellow_output = '/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/yellow2.mp4'
clip2 = VideoFileClip('/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
print('yellow')

challenge_output = '/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/extra2.mp4'
clip2 = VideoFileClip('/Users/Hayoung/Dropbox/Study/carND/01_Finding_Lane_Lines/CarND-LaneLines-P1-master/challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
print('challenge')
