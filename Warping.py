from copy import copy
from tkinter import Y
from turtle import clone
import cv2
import numpy as np
import os
preset_women = [
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
preset_cheetah = [
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
preset_A = [
    [[193, 100],[185, 134]],
    [[183, 192], [183, 225]],
    [[209, 112], [207, 142]],
    [[203, 190], [198, 222]],
    [[200, 69], [288, 97]],
    [[191, 252], [274, 237]],
    [[292, 109], [327, 158]],
    [[277, 238], [325, 188]], 
    [[193, 167], [240, 169]],
    [[280, 150], [275, 194]]
]
preset_B = [
    [[174, 99], [173, 134]],
    [[170, 183], [163, 229]],
    [[186, 108], [193, 141]],
    [[190, 188], [183, 220]],
    [[173, 74], [281, 101]],
    [[169, 252], [271, 232]],
    [[288, 110], [320, 160]],
    [[281, 228], [317, 190]],
    [[181, 163], [235, 170]],
    [[271, 141], [271, 194]]
]
preset_C = [
    [[178, 87], [177, 129]],
    [[174, 167], [170, 211]],
    [[197, 96], [195, 124]],
    [[189, 172], [187, 205]],
    [[178, 70], [275, 92]],
    [[173, 240], [265, 227]],
    [[284, 98], [310, 157]],
    [[273, 227], [312, 172]],
    [[185, 152], [229, 155]],
    [[263, 125], [259, 189]]
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

    def resize(self,size):
        self.img = cv2.resize(self.img,size)

    def shape(self):
        return self.img.shape

    def lShape(self):
        return self.L1.shape

    def perpendicular(self,dest_PQ):
        per_destPQ = dest_PQ.copy()
        
        temp_x = per_destPQ[:,:,:,[0]]
        temp_y = per_destPQ[:,:,:,[1]]
        temp = np.concatenate((temp_y,-temp_x),axis= 3)
        return temp
        
    def transform(self):
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
        src_PX = dest_x - src_p
        dest_PQ = dest_q - dest_p
        dest_PX =  dest_x - dest_p
        dest_QX = dest_x - dest_q
        
        src_PQ_vector_length = (src_PQ[:,:,:,[0]] ** 2 + src_PQ[:,:,:,[1]] ** 2) ** 0.5
        dest_PQ_vector_length = (dest_PQ[:,:,:,[0]] ** 2 + dest_PQ[:,:,:,[1]] ** 2) ** 0.5
        dest_XP_vector_length = (dest_PX[:,:,:,[0]] ** 2 + dest_PX[:,:,:,[1]] ** 2) ** 0.5
        dest_XQ_vector_length = (dest_QX[:,:,:,[0]] ** 2 + dest_QX[:,:,:,[1]] ** 2) ** 0.5

        dest_PXDotPQ = dest_PX * dest_PQ
        per_destPQ = self.perpendicular(dest_PQ)
        dest_PXDotper_destPQ = dest_PX * per_destPQ

        u = (dest_PXDotPQ[:,:,:,[0]] + dest_PXDotPQ[:,:,:,[1]])/(dest_PQ_vector_length) ** 2
        v = (dest_PXDotper_destPQ[:,:,:,[0]] + dest_PXDotper_destPQ[:,:,:,[1]])/dest_PQ_vector_length

        dest_PXDotper_srcPQ = self.perpendicular(src_PQ)
        length = dest_PQ_vector_length
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
        
        # weight = (length / (2 + dist) ** 1.2) ** 0.7
        weight = (length / (1 + dist))
        weight = np.concatenate([weight,weight],axis=3)
        weight_sum = np.sum(weight,axis=2)
        
        X = src_p + u * src_PQ + (v * dest_PXDotper_srcPQ) / src_PQ_vector_length
        X = X * weight    
        X = np.sum(X,axis = 2)
        X = X / weight_sum
        
        X = np.where(X < 0 , 0 , X)
        X1 = X[:,:,[0]]
        X2 = X[:,:,[1]]
        X1 = np.where(X1 >= shape[0] - 1,shape[0] - 2 ,X1)
        X2 = np.where(X2 >= shape[1] - 1,shape[1] - 2 ,X2)
        X = np.concatenate((X1,X2),axis=2)

        return X
                    
    def warping(self):
        img = self.img.copy()
        img = img.astype(np.int32)
        dest = np.zeros(img.shape,np.int32)

        X = self.transform()
        
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

def morphing(img1,img2,img1_scalar,img2_scalar):
    L2 = np.int32(np.array(img1.L1) * img1_scalar + np.array(img2.L1) * img2_scalar)
    img1.L2 = L2
    img2.L2 = L2 
    img1.warping()
    img2.warping()
    a = np.int32(img1.img)
    b = np.int32(img2.img)
    morphing_ing = np.uint8(a * img1_scalar + b * img2_scalar)
    return img1.img,img2.img,morphing_ing

def base(img1,img2):
    img1 = copy(img1)
    img2 = copy(img2)
    print("Start morphing")
    img1,img2,morphing_img = morphing(img1,img2,0.7,0.3)
    print("Finish morphing")
    cv2.imwrite('cheetah_wapring.jpg',img1)
    cv2.imwrite('women_wapring.jpg',img2)
    cv2.imwrite('morphing.jpg',morphing_img)
    cv2.imshow('output',np.concatenate((img1,img2,morphing_img),axis = 1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def animation(imgs:Image = None,step:int = 0,build = False,saveImages = False,dir = None):
    if build:
        src_imgs = []
        for i in imgs:
            src_imgs.append(copy(i))

        print("Start building animation , please wait")
        

        morphing_imgs = []
        
        for img_idx in range(len(src_imgs) - 1):
            morphing_imgs.append(src_imgs[img_idx].img)
            for i in range(1,step):
                temp1 = copy(src_imgs[img_idx])
                temp2 = copy(src_imgs[img_idx + 1])
                img1_scalar = (step - i) / step
                img2_scalar = i / step
                _,_,morphing_img = morphing(temp1,temp2,img1_scalar,img2_scalar)
                morphing_imgs.append(morphing_img)
                print(f'{i}/{step} finish morphing')
            
        morphing_imgs.append(src_imgs[len(src_imgs) - 1].img)
        
        if saveImages:
            for i,img in enumerate(morphing_imgs):
                cv2.imwrite(f'{dir}/{i}.jpg',img)

        key = 0
        while 1:
            for i in range(len(morphing_imgs)):
                cv2.imshow("animation",morphing_imgs[i])
                if i % step == 0:
                    key = cv2.waitKey(1000)            
                key = cv2.waitKey(100)
                if key == 13:
                    break
            if key == 13:
                    break
    else:
        imgs = []
        for i in range(len(os.listdir('animation'))):
            imgs.append(cv2.imread(f'animation/{i}.jpg'))
        while 1:
            for i in range(len(imgs)):
                cv2.imshow("animation",imgs[i])
                if i % step == 0:
                    key = cv2.waitKey(1000)            
                key = cv2.waitKey(100)
                if key == 13:
                    break
            if key == 13:
                    break

triangle_src = cv2.imread('triangle.jpg')
triangle_show = copy(triangle_src)
weights = np.zeros(3,np.float32)

def cal_triangle(vertexs:np.ndarray):
    global weights
    temp = copy(vertexs[:,0])
    vertexs[:,0] = vertexs[:,0] - vertexs[:,1]
    vertexs[:,1] = vertexs[:,1] - vertexs[:,2]
    vertexs[:,2] = vertexs[:,2] - temp
    
    vertexs = (vertexs[:,:,0] ** 2 + vertexs[:,:,1] ** 2) ** 0.5
    vertexs = (((vertexs[:,0] ** 2) * (vertexs[:,2] ** 2) - ((vertexs[:,0] ** 2 + vertexs[:,2] ** 2 - vertexs[:,1] ** 2)/2) ** 2)/4) ** 0.5
    
    weight_sum = np.sum(vertexs)
    
    weights = vertexs / weight_sum

def triangle_mouse(event,x,y,flag,para):
    global triangle_show,triangle_src,weight
    if event == cv2.EVENT_LBUTTONDOWN:
        triangle_show = copy(triangle_src)
        cv2.arrowedLine(triangle_show,[329,42],[x,y],[0,0,0])
        cv2.arrowedLine(triangle_show,[33,525],[x,y],[0,0,0])
        cv2.arrowedLine(triangle_show,[627,525],[x,y],[0,0,0])
        cv2.imshow('triangle',triangle_show)
        vertexs = [
            [[33,525],[x,y],[627,525]],
            [[329,42],[x,y],[627,525]],
            [[329,42],[x,y],[33,525]]
        ]
        cal_triangle(np.array(vertexs))

def multiple_morphing(imgs):
    global triangle_show,triangle_src,weights
    cv2.namedWindow('triangle')
    cv2.setMouseCallback('triangle',triangle_mouse)
    cv2.imshow('triangle',triangle_show)

    img = np.zeros((imgs[0].img.shape),dtype = np.float32)
    for i in range(len(imgs)):
        img += np.float32(imgs[i].img) * weights[i]
    img /= sum(weights)
    img = np.uint8(img)

    origin_imgs = []
    for i in imgs:
        origin_imgs.append(copy(i))
    
    while 1:
        while 1:
            cv2.imshow('img',img)
            key = cv2.waitKey(1)
            if key == 13:
                print("Start blending")
                break
        imgs = []
        for i in origin_imgs:
            imgs.append(copy(i))
        L = 0
        for i in range(len(imgs)):
            L += np.float32(imgs[i].L1) * weights[i]
        L /= sum(weights)
        L = np.int32(L)
        for i in range(len(imgs)):
            imgs[i].L2 = L
            imgs[i].warping()
        img = np.zeros((imgs[i].img.shape),dtype = np.float32)
        for i in range(len(imgs)):
            img += np.float32(imgs[i].img) * weights[i]
        img /= sum(weights)
        img = np.uint8(img)
        


if __name__ == '__main__':
    
    draw_line = False # Whether draw line by yourself,True for draw,False for preset line segment
    isBase = False  # Whether run base or not , True for run base
    isAnimation = False # Whether run animation or not , True for run animation
    isRebuildAnimation = False # If run animation , rebuild animation or run pre-build animation , True for rebuild animation
    isMultipleMorphing = True # Whether run multiple morphing or not , True for run multiple morphing


    women = Image('women.jpg')
    women.L1 = preset_women
    cheetah = Image('cheetah.jpg')
    cheetah.L1 = preset_cheetah
    if draw_line:
        women.draw_line()
        cheetah.draw_line()

    if isBase:
        base(women,cheetah) # basic morphing

    if isAnimation:
        step = 10
        size = (375,323)
        if isRebuildAnimation:
            A = Image('A.jpg')
            A.resize(size)
            A.L1 = preset_A
            B = Image('B.jpg')
            B.resize(size)
            B.L1 = preset_B
            C = Image('C.jpg')
            C.resize(size)
            C.L1 = preset_C
            if draw_line:
                A.draw_line()
                B.draw_line()
                C.draw_line()
            imgs = [A,B,C,A]
            animation(imgs,step,True,True,'animation') # morphing animation
        else:
            animation(step = step)
    
    if isMultipleMorphing:
        size = (375,323)
        if not (isAnimation and isRebuildAnimation):
            A = Image('A.jpg')
            A.resize(size)
            A.L1 = preset_A
            B = Image('B.jpg')
            B.resize(size)
            B.L1 = preset_B
            C = Image('C.jpg')
            C.resize(size)
            C.L1 = preset_C
            if draw_line:
                A.draw_line()
                B.draw_line()
                C.draw_line()
        imgs = [A,B,C]
        multiple_morphing(imgs)
    
