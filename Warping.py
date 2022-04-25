from copy import copy
from turtle import clone
import cv2
import numpy as np

aaa = [
    [[23,67],[28,90]],
    [[32,135],[25,158]],
    [[45,67],[45,88]],
    [[48,135],[48,158]],
    [[56,35],[145,50]],
    [[56,180],[150,165]],
    [[148,60],[185,68]],
    [[155,163],[185,160]],
    [[35,115],[95,115]],
    [[130,90],[130,135]]
]
bbb = [
    [[12,30],[10,60]],
    [[10,195],[12,230]],
    [[20,45],[20,75]],
    [[20,185],[20,213]],
    [[20,3],[150,3]],
    [[20,250],[150,250]],
    [[165,3],[185,3]],
    [[165,250],[185,250]],
    [[25,128],[160,128]],
    [[180,90],[180,170]],
]


class Image:

    def __init__(self,imgPath):
        self.imgPath = imgPath
        self.L1 = []
        self.L2 = 0
        self.state = 0
        self.img = self.read_img()
        self.drawImg = self.img.copy()

    def draw_line(self):
        cv2.namedWindow(self.imgPath)
        cv2.setMouseCallback(self.imgPath,self.mouse)
        while 1:
            cv2.imshow(self.imgPath,self.drawImg)
            key = cv2.waitKey(1)
            if key == 13:
                break
        cv2.destroyAllWindows()

    def mouse(self,event,x,y,flag,para):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state += 1
            if self.state == 5:
                self.state = 1
        if self.state == 1:
            self.arrStart = [x,y]
            self.state += 1
        elif self.state == 3:
            cv2.arrowedLine(self.drawImg,self.arrStart,[x,y],[255,255,255],thickness=5)
            self.L1.append([[self.arrStart[1],self.arrStart[0]],[y,x]])
            self.state += 1
            
    def read_img(self):
        return cv2.imread(self.imgPath)

    def shape(self):
        return self.img.shape

    def lShape(self):
        return self.L1.shape

    def perpendicular(self,dest_PQ,dest_PX):
        per_destPQ = dest_PQ.copy()
        temp = per_destPQ[:,:,:,[0,1]]
        per_destPQ[:,:,:,[0,1]] = per_destPQ[:,:,:,[1,0]]
        per_destPQ[:,:,:,[1,0]] = -temp
        temp = dest_PX * per_destPQ
        temp = temp[:,:,:,0] + temp[:,:,:,1]
        temp = temp < 0
        for x in range(temp.shape[0]):
            for y in range(temp.shape[1]):
                for z in range(temp.shape[2]):
                    if temp[x,y,z]:
                        per_destPQ[x,y,z,[0,1]] = -per_destPQ[x,y,z,[0,1]]  

        return per_destPQ
        
    def getLineMat(self):
        self.L1 = np.array(self.L1,dtype=np.int32)
        self.L2 = np.array(self.L2,dtype=np.int32)
        shape = self.shape()
        lShape = self.lShape()
        
        src_p = np.zeros(shape = (shape[0],shape[1],lShape[0],2),dtype =np.int32)
        src_q = np.zeros(shape = (shape[0],shape[1],lShape[0],2),dtype =np.int32)
        dest_p = np.zeros(shape = (shape[0],shape[1],lShape[0],2),dtype =np.int32)
        dest_q = np.zeros(shape = (shape[0],shape[1],lShape[0],2),dtype =np.int32)
        
        p1 = [i[0] for i in self.L1]
        q1 = [i[1] for i in self.L1]
        p2 = [i[0] for i in self.L2]
        q2 = [i[1] for i in self.L2]
        

        dest_x = np.zeros(shape = (shape[0],shape[1],lShape[0],2),dtype = np.int32)
        for x in range(shape[0]):
            for y in range(shape[1]):
                src_p[x,y] = p1
                src_q[x,y] = q1
                dest_p[x,y] = p2
                dest_q[x,y] = q2
                position = []
                for _ in range(lShape[0]):
                    position.append([x,y])
                dest_x[x,y] = position
        dest_x = np.array(dest_x,np.int32)

        src_p = np.array(src_p,np.int32)
        src_q = np.array(src_q,np.int32)
        dest_p = np.array(dest_p,np.int32)
        dest_q = np.array(dest_q,np.int32)


        src_PQ = src_q - src_p
        dest_PQ = dest_q - dest_p
        dest_PX =  dest_x - dest_p
        dest_QX = dest_x - dest_q
        
        src_PQ_vector_length = (src_PQ[:,:,:,0] ** 2 + src_PQ[:,:,:,1] ** 2) ** 0.5
        dest_PQ_vector_length = (dest_PQ[:,:,:,0] ** 2 + dest_PQ[:,:,:,1] ** 2) ** 0.5
        dest_XP_vector_length = (dest_PX[:,:,:,0] ** 2 + dest_PX[:,:,:,1] ** 2) ** 0.5
        dest_XQ_vector_length = (dest_QX[:,:,:,0] ** 2 + dest_QX[:,:,:,1] ** 2) ** 0.5

        dest_PXDotPQ = dest_PX * dest_PQ
        per_destPQ = self.perpendicular(dest_PQ,dest_PX)
        dest_PXDotper_destPQ = dest_PX * per_destPQ

        u = (dest_PXDotPQ[:,:,:,0] + dest_PXDotPQ[:,:,:,1])/(dest_PQ_vector_length) ** 2
        v = (dest_PXDotper_destPQ[:,:,:,0] + dest_PXDotper_destPQ[:,:,:,1])/dest_PQ_vector_length

        dest_PXDotper_srcPQ = self.perpendicular(src_PQ,dest_PX)
        u = np.reshape(u,(u.shape[0],u.shape[1],u.shape[2],1))
        v = np.reshape(v,(v.shape[0],v.shape[1],v.shape[2],1))
        src_PQ_vector_length = np.reshape(src_PQ_vector_length,(src_PQ_vector_length.shape[0],src_PQ_vector_length.shape[1],src_PQ_vector_length.shape[2],1))
        
        length = dest_PQ_vector_length
        length = np.reshape(length,(length.shape[0],length.shape[1],length.shape[2],1))
        dist = np.zeros(shape = u.shape,dtype=np.int32)
        for x in range(u.shape[0]):
            for y in range(u.shape[1]):
                for z in range(u.shape[2]):
                    if u[x,y,z,0] < 0:
                        dist[x,y,z,0] = dest_XP_vector_length[x,y,z]
                    elif u[x,y,z,0] >= 0 and u[x,y,z,0] <= 1:
                        dist[x,y,z,0] = abs(v[x,y,z])
                    elif u[x,y,z,0] > 1:
                        dist[x,y,z,0] = dest_XQ_vector_length[x,y,z]
                
        weight = (length / (2 + dist) ** 1.2) ** 0.7
        weight = np.concatenate([weight,weight],axis=3)
        weight_sum = np.sum(weight,axis=2)
        
        X = src_p + u * src_PQ + (v * dest_PXDotper_srcPQ) / src_PQ_vector_length
        X = X * weight    
        X = np.sum(X,axis = 2)
        X = X / weight_sum
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                if X[x,y,0] < 0:X[x,y,0] = 0
                elif X[x,y,0] >= shape[0] - 1:X[x,y,0] = shape[0] - 2
                if X[x,y,1] < 0:X[x,y,1] = 0
                elif X[x,y,1] >= shape[1] - 1:X[x,y,1] = shape[1] - 2
                
        return X
                    
    def warping(self):
        img = self.img.copy()
        img = img.astype(np.int32)
        dest = np.zeros(img.shape,np.int32)

        X = self.getLineMat()
        
        temp = np.int32(X)
        x_low = temp[:,:,0]
        x_up = x_low + 1
        y_low = temp[:,:,1]
        y_up = y_low + 1
        
        x_low_scalar = X[:,:,0] - x_low
        x_up_scalar = x_up - X[:,:,0]
        y_low_scalar = X[:,:,1] - y_low
        y_up_scalar = y_up - X[:,:,1]
        
        
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                color1 = np.int32(self.img[x_low[x,y],y_low[x,y]]) * x_low_scalar[x,y] + np.int32(self.img[x_up[x,y],y_low[x,y]]) * x_up_scalar[x,y]
                color2 = np.int32(self.img[x_low[x,y],y_up[x,y]]) * x_low_scalar[x,y] + np.int32(self.img[x_up[x,y],y_up[x,y]]) * x_up_scalar[x,y]
                dest[x,y] = color1 * y_low_scalar[x,y] + color2 * y_up_scalar[x,y]
        self.img = np.uint8(dest)

    def save(self,name):
        cv2.imwrite(name,self.img)

    def show(self):
        cv2.namedWindow(self.imgPath)
        while 1:
            cv2.imshow(self.imgPath,self.img)
            key = cv2.waitKey(1)
            if key == 13:
                break

img1 = Image('cheetah.jpg')
img1.draw_line()
img2 = Image('women.jpg')
img2.draw_line()
img1.L2 = (np.array(img1.L1) + np.array(img2.L1)) // 2
img2.L2 = (np.array(img1.L1) + np.array(img2.L1)) // 2 
print("Start warping")
img1.warping()
img2.warping()
img1.save('cheetah_wapring.jpg')
img2.save('women_wapring.jpg')
a = np.int32(img1.img)
b = np.int32(img2.img)
img3 = copy(img1)
img3.img = np.uint8(a * 0.3 + b * 0.7)
img3.save('morphing.jpg')
