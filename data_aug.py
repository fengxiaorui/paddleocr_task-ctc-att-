
#!/usr/env/bin python3
from functools import reduce
import numpy as np
import cv2
import math
import random
import os
 
#仿射变换
def appy_affine_transform(img):
    '''
    仿射变换
    '''
    rows, cols, channels = img.shape
    if cols > 200:
        scale = 0.99
    else:
        scale = 0.95
    p1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    p2 = np.float32([[0,rows*0.05], [cols*scale,rows*0.05], [cols*(1-scale),rows*0.95]])

    M = cv2.getAffineTransform(p1, p2)
    dst = cv2.warpAffine(img, M, (cols,rows))
    return dst
    # cv2.imshow('original', img)
    # cv2.imshow('result', dst)
    # cv2.waitKey(0)

def apply_gauss_blur(img, ks=None):
    '''
    高斯滤波
    '''
    if ks is None:
        ks = [3, 5]
    ksize = random.choice(ks)
 
    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize >= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img

def apply_norm_blur(img, ks=None):
    '''
    滤波
    '''
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img


def apply_prydown(img):
    """
    模糊图像，模拟小图片放大的效果
    """
    scale = random.uniform(1, 3)
    height = img.shape[0]
    width = img.shape[1]
 
    out = cv2.resize(img, (int(width / scale), int(height / scale)), interpolation=cv2.INTER_AREA)
    return cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
 
def apply_sharp(word_img):
    '''
    图像锐化
    '''
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(word_img, -1, sharp_kernel)
def single_word():
    '''
    对单个字符进行旋转操作
    '''
    txt_train_dir = "./LabelTrain.txt"
    txt_train_save_dir = "./rotate_singleword.txt"
    img_dir = "./TrainImages/"
    img_save_dir = "./TrainImages_single/"
    with open(txt_train_dir,'r',encoding='utf-8') as f:
        for line in f.readlines():
            
            label = line.split('\t')[1].strip()
            # print(len(label))
            # print(label)
            if len(label) == 1:
                print(line)
                img = cv2.imread(img_dir+line.split('\t')[0].split('/')[-1])
                rows,cols,c = img.shape
                # cols-1 and rows-1 are the coordinate limits.
                M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imwrite(img_save_dir+line.split('\t')[0].split('/')[-1],dst)
                with open(txt_train_save_dir,'a') as ff:
                    ff.write(line)
if __name__ == "__main__":
    image_dir = "./TrainImages"
    for root,dirs,files in os.walk(image_dir,topdown=False):
        folder_name = root.split('/')[-1]
        for name in files:
            print("root:",root)
            save_dir = root#数据拓增后保存的地址
            isExists=os.path.exists(save_dir)
            if not isExists:
                os.makedirs(save_dir)
            print(os.path.join(root, name))
            img = cv2.imread(os.path.join(root, name))
            name = name.replace(' ','')
            #滤波 锐化
            img_new_sharp = apply_sharp(img)
            save_dir1 = save_dir.replace(folder_name,folder_name+"_sharp")
            isExists=os.path.exists(save_dir1)
            if not isExists:
                os.makedirs(save_dir1)
        
            cv2.imwrite(save_dir1+'/'+name.split('.png')[0]+"_sharp.png",img_new_sharp)
            #模糊图像，模拟小图片放大的效果
            img_new_prydown = apply_prydown(img)
            save_dir2 = save_dir.replace(folder_name,folder_name+"_prydown")
            isExists=os.path.exists(save_dir2)
            if not isExists:
                os.makedirs(save_dir2)
       
            cv2.imwrite(save_dir2+'/'+name.split('.png')[0]+"_prydown.png",img_new_prydown)
            #norm_blur
            img_new_normblur = apply_norm_blur(img)
            save_dir3 = save_dir.replace(folder_name,folder_name+"_normblur")
            isExists=os.path.exists(save_dir3)
            if not isExists:
                os.makedirs(save_dir3)
           
            cv2.imwrite(save_dir3+'/'+name.split('.png')[0]+"_normblur.png",img_new_normblur)
            #gauss_blur
            img_new_gaussblur = apply_gauss_blur(img)
            save_dir4 = save_dir.replace(folder_name,folder_name+"_gaussblur")
            isExists=os.path.exists(save_dir4)
            if not isExists:
                os.makedirs(save_dir4)
       
            cv2.imwrite(save_dir4+'/'+name.split('.png')[0]+"_gaussblur.png",img_new_gaussblur)
            #仿射变换
            img_new_affine = appy_affine_transform(img)
            save_dir5 = save_dir.replace(folder_name,folder_name+"_affine")
            isExists=os.path.exists(save_dir5)
            if not isExists:
                os.makedirs(save_dir5)
           
            cv2.imwrite(save_dir5+'/'+name.split('.png')[0]+"_affine.png",img_new_affine)
           
           