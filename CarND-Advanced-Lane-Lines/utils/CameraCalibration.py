import numpy as np
import cv2

class CameraCalibration():
    def __init__(self, images):
        self.mtx = None # Camera matrix
        self.dist = None # Distortion coefficients

        self.size = None # image size

        objpoints, imgpoints = self.get_obj_img_points(images)
        self.mtx, self.dist = self.get_calibration_mat(objpoints, imgpoints)

    def get_obj_img_points(self, images):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            self.size = gray.shape[::-1]

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints

    def get_calibration_mat(self, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.size, None, None)
        return mtx, dist

    def undistort_img(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
