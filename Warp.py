from cmath import cos, sin
import math
from turtle import degrees
import cv2
import numpy as np
import threading
from multiprocessing import Process

from tqdm import main


class Warping:
   
    def __init__(self,imgPath):
        self.imagePath = imgPath
        self.src = cv2.imread(imgPath)
        self.img = cv2.imread(imgPath)
        self.draw = False
        self.arrStart = []
        self.L1 = []
        self.L2 = []
        self.isL1 = True
        self.state = 0
        # self.resize((224,224))
        cv2.namedWindow(self.imagePath)
        cv2.setMouseCallback(self.imagePath,self.mouse)

    def mouse(self,event,x,y,flag,para):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state += 1
            if self.state == 5:
                self.state = 1
        if self.state == 0:
            pass
        elif self.state == 1:
            self.arrStart = [x,y]
            self.state += 1
            pass
        elif self.state == 3:
            cv2.arrowedLine(self.img,self.arrStart,[x,y],[255,255,255],thickness=5)
            if self.isL1:
                self.L1.append([[self.arrStart[1],self.arrStart[0]],[y,x]])
            else:
                self.L2.append([[self.arrStart[1],self.arrStart[0]],[y,x]])
            self.isL1 = not self.isL1
            self.state += 1
            pass
        

    def show(self):
        cv2.imshow(self.imagePath,self.img)
        key = cv2.waitKey(1)
        if key == 13:
            print('Start Warping')
            self.warpImage()
            self.L1 = []
            self.L2 = []
    
    def resize(self,size):
        size = np.int32(size)
        self.img = cv2.resize(self.img,size)
        self.src = cv2.resize(self.src,size)
        

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
        return dest_x, dest_y,u,v

    def dist(self,vector,x):
        vector = self.perpendicular(vector,x)
        pass
    
    def weight(self,L,x,dst_x,u,v):
        a = 1
        p = 1
        b = 1
        Lp = L[0]
        Lq = L[1]
        qminp = np.subtract(Lq,Lp)
        length = self.vectorLength(qminp)
        if u >= 0 and u <= 1:
            dist = abs(v)
        elif u < 0:
            dist = self.vectorLength(np.subtract(x,Lp))
        else:
            dist = self.vectorLength(np.subtract(x,Lq))
        return (length ** p/(a+dist)) ** b
        return 1

    def warpImage(self):
        shape = self.shape()
        width = shape[0]
        height = shape[1]
        dest = np.zeros(shape,dtype=np.uint8)
        L1= self.L1
        L2 = self.L2
        dest_x = np.zeros(len(L1))
        dest_y = np.zeros(len(L1))
        weight = np.zeros(len(L1))
        for x in range(len(dest)):
            for y in range(len(dest[0])):
                xSum = [0,0]
                weightSum = 0
                for i in range(len(L1)):

                    dest_x[i],dest_y[i],u,v = self.transform(L2[i],L1[i],[x,y])
                    weight[i] = self.weight(L2[i],[x,y],[dest_x[i],dest_y[i]],u,v)
                    # print(weight[i])
                    xSum[0] = xSum[0] + dest_x[i] * weight[i]
                    xSum[1] = xSum[1] + dest_y[i] * weight[i]
                    weightSum += weight[i]
                # print(xSum)
                # new_x = int(xSum[0])
                # new_y = int(xSum[1])

                new_x = xSum[0]/weightSum
                new_y = xSum[1]/weightSum
                x_low = int(new_x)
                y_low = int(new_y)
                    
                if x_low >= 0 and x_low < width - 1 and y_low >=0 and y_low < height - 1:
                    color1 = (new_x - x_low) * self.src[x_low,y_low] + (x_low + 1 - new_x) * self.src[x_low + 1,y_low]
                    color2 = (new_x - x_low) * self.src[x_low,y_low + 1] + (x_low + 1 - new_x) * self.src[x_low,y_low + 1]
                    color = (new_y - y_low) * color1 + (y_low + 1 - new_y) * color2
                    color = np.uint8(color)
                    dest[x,y] = color
        self.img = dest
                    
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

a1 = np.zeros(10)
a2 = np.full(10,10)
a3 = a1 - a2
print(a3 * 10)

if __name__ == '__main__':
    path = 'women.jpg'

    img = Warping(path)

    divide = 1

    shape = img.shape()
    img.resize((shape[1]//divide,shape[0]//divide))

    while 1:
        img.show()