#导入库
import cv2
import dlib
import os
import sys
import random

#output_dir用来存放录入的脸部图片
output_dir="./my_faces"
#size最后的图片尺寸为size*size
size=64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def relight(img,light=1,bias=0):
    #随机改变图片亮度，增加图片的多样性
    w=img.shape[0]
    h=img.shape[1]
    for i in range(0,h):
        for j in range(w):
            for c in range(3):
                tmp=int(img[j,i,c]*light+bias)
                if tmp>255:
                    tmp=255
                elif tmp<0:
                    tmp=0
                img[j,i,c]=tmp
    return img
detector=dlib.get_frontal_face_detector()
camera=cv2.VideoCapture(0)

index=int(input("输入录入断点："))
while True:
    if (index<=10000):
        #录入10000张照片
        print("Being processed picture %s"%index)
        success,img=camera.read()
        gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dets=detector(gray_img,1)
        for i,d in enumerate(dets):
            #针对识别出来的面部进行剪裁，并把尺寸调整成size*size
            x1=d.top() if d.top() >0 else 0
            y1=d.bottom() if d.bottom()>0 else 0
            x2=d.left() if d.left()>0 else 0
            y2=d.right() if d.right()>0 else 0
            face=img[x1:y1,x2:y2]
            face=relight(face,random.uniform(0.5,1.5),random.randint(-50,50))
            face=cv2.resize(face,(size,size))
            cv2.imshow("image",face)
            cv2.imwrite(output_dir+"/"+str(index)+".jpg",face)
            index+=1
    key=cv2.waitKey(30)& 0xff
    if key==27:
        break