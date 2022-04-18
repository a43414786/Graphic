from cmath import cos, sin
import math
from turtle import degrees
import cv2
import numpy as np

class Warping:
   
    def __init__(self,imgPath):
        self.img = cv2.imread(imgPath)

    def show(self):
        cv2.imshow('img',self.img)
        cv2.waitKey(0)
    
    def resize(self,size):
        size = np.int32(size)
        self.img = cv2.resize(self.img,size)

    def shape(self):
        return self.img.shape

    def vectorLength(self,vector):
        return ((vector[0]) **2 + (vector[1]) ** 2) ** 0.5

    def perpendicular(self,vector):
        return [vector[1],-vector[0]]

    def transform(self,L1,L2,X):
        p = L1[0]
        q = L1[1]
        dest_p = L2[0]
        dest_q = L2[1]
        u = np.matmul(np.subtract(X,p),np.subtract(q,p)) / (self.vectorLength(np.subtract(q,p)) ** 2)
        v = np.matmul(np.subtract(X,p),self.perpendicular(np.subtract(q,p))) / self.vectorLength(np.subtract(q,p))
        dest_x = dest_p[0] + u * np.subtract(dest_q,dest_p)[0] + (v * self.perpendicular(np.subtract(dest_q,dest_p))[0]) / self.vectorLength(np.subtract(dest_q,dest_p))
        dest_y = dest_p[1] + u * np.subtract(dest_q,dest_p)[1] + (v * self.perpendicular(np.subtract(dest_q,dest_p))[1]) / self.vectorLength(np.subtract(dest_q,dest_p))
        return dest_x, dest_y

    def weight(self):
        return 1

    def warpImage(self,L1,L2):
        shape = self.shape()
        width = shape[0]
        height = shape[1]
        dest = np.zeros(shape,dtype=np.uint8)
        dest_x = np.zeros(len(L1))
        dest_y = np.zeros(len(L1))
        weight = np.zeros(len(L1))
        for x in range(len(dest)):
            for y in range(len(dest[0])):
                xSum = [0,0]
                weightSum = 0
                for i in range(len(L1)):
                    dest_x[i],dest_y[i] = self.transform(L1[i],L2[i],[x,y])
                    weight[i] = self.weight()
                    xSum[0] = xSum[0] + dest_x[i] * weight[i]
                    xSum[1] = xSum[1] + dest_y[i] * weight[i]
                    weightSum += weight[i]
                if int(xSum[0]/weightSum) < width and int(xSum[1]/weightSum) < height:
                    dest[x,y] = self.img[int(xSum[0]/weightSum),int(xSum[1]/weightSum)]
        return dest
                    
        

    def warping(self,transform):
        shape = self.shape()
        width = shape[0]
        height = shape[1]
        new_img = np.zeros(shape,dtype=np.uint8)
        for x in range(len(new_img)):
            for y in range(len(new_img[0])):
                ans = np.dot([[1,0,-width/2],[0,1,-height/2],[0,0,1]],[[x],[y],[1]])
                ans = np.dot(transform,ans)
                ans = np.dot([[1,0,width/2],[0,1,height/2],[0,0,1]],ans)
                new_x = ans[0][0]
                new_y = ans[1][0]
                
                if(new_x >= 0 and new_x < shape[0]):
                    if(new_y >= 0 and new_y < shape[1]):
                        new_img[x,y] = self.img[int(new_x),int(new_y)]

        return new_img


path = 'img.JPG'

img = Warping(path)
# cv2.imshow('img',img.img[400:500,250:450])
# cv2.waitKey(0)

L1 = [
    [[500,250],[400,350]]
]
L2 = [
    [[400,250],[500,350]]
]

# imgshape = img.shape()
# img.resize([imgshape[1]/3,imgshape[0]/3])
# a = (math.pi * 2) / 360
# rotate_angle = 90
# t = np.real([[cos(rotate_angle * a),sin(rotate_angle * a),0],[-sin(rotate_angle * a),cos(rotate_angle * a),0],[0,0,1]])
new_img = img.warpImage(L1,L2)
cv2.imwrite("new.JPG",new_img)
cv2.imshow('img',new_img)
cv2.waitKey(0)

img.show()