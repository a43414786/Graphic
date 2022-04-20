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

    def perpendicular(self,vector,x):
        if np.dot([vector[1],-vector[0]],x) >= 0:
            return [vector[1],-vector[0]]
        else:
            return [-vector[1],vector[0]]

    def transform(self,L1,L2,X):
        p = L1[0]
        q = L1[1]
        dest_p = L2[0]
        dest_q = L2[1]
        qminp = np.subtract(q,p)
        Xminp = np.subtract(X,p)
        dqmindp = np.subtract(dest_q,dest_p)
        Xmindp = np.subtract(X,dest_p)
        u = np.dot(Xminp,qminp) / (self.vectorLength(qminp) ** 2)
        v = np.dot(Xminp,self.perpendicular(qminp,Xminp)) / self.vectorLength(qminp)
        dest_x = dest_p[0] + u * dqmindp[0] + (v * self.perpendicular(dqmindp,Xmindp)[0]) / self.vectorLength(dqmindp)
        dest_y = dest_p[1] + u * dqmindp[1] + (v * self.perpendicular(dqmindp,Xmindp)[1]) / self.vectorLength(dqmindp)
        return dest_x, dest_y

    def weight(self,L,x):
        return 1/( 1 + self.vectorLength(np.subtract(x,L[0]))) 

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
                    dest_x[i],dest_y[i] = self.transform(L2[i],L1[i],[x,y])
                    weight[i] = self.weight(L2[i],[x,y])
                    # print(weight[i])
                    xSum[0] = xSum[0] + dest_x[i] * weight[i]
                    xSum[1] = xSum[1] + dest_y[i] * weight[i]
                    weightSum += weight[i]
                # print(xSum)
                # new_x = int(xSum[0])
                # new_y = int(xSum[1])
                new_x = int(xSum[0]/weightSum)
                new_y = int(xSum[1]/weightSum)
                if new_x >= 0 and new_x < width and new_y >=0 and new_y < height:
                    dest[x,y] = self.img[new_x,new_y]
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
while 1:
    cv2.imshow("img",img.img)
    cv2.waitKey(10)
    if cv2.L
# cv2.imshow('img',img.img[400:500,250:450])
# cv2.waitKey(0)


devide = 5

L1 = [
    [[0//devide,0//devide],[2000//devide,0//devide]],
    # [[0//devide,0//devide],[0//devide,1000//devide]]
    # [[500//devide,250//devide],[400//devide,350//devide]]
]
L2 = [
    [[2000//devide,0//devide],[0//devide,0//devide]],
    # [[500//devide,500//devide],[0//devide,1000//devide]]
    # [[400//devide,250//devide],[500//devide,350//devide]]
]

imgshape = img.shape()
img.resize([imgshape[1]//devide,imgshape[0]//devide])
# a = (math.pi * 2) / 360
# rotate_angle = 90
# t = np.real([[cos(rotate_angle * a),sin(rotate_angle * a),0],[-sin(rotate_angle * a),cos(rotate_angle * a),0],[0,0,1]])
new_img = img.warpImage(L1,L2)
cv2.imwrite("new.JPG",new_img)
cv2.imshow('img',new_img)
cv2.waitKey(0)

img.show()