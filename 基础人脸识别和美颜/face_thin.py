import cv2
import dlib
import numpy as np

# 获取图像中的人脸关键点
# 输入
# img ： 图像
# det_face ： 人脸检测器
# det_landmarks ： 人脸关键点检测器
def get_landmarks_points(img,det_face,det_landmarks):
    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸区域
    face_rects = det_face(gray, 0)
    
    # 获取68个关键点
    landmarks = det_landmarks(gray, face_rects[0])
    
    # 获取关键点的坐标
    landmarks_points = []
    parts = landmarks.parts()
    for part in parts:
        landmarks_points.append((part.x,part.y))
    return landmarks_points
   
# 双线性差值   
def BilinearInsert(src,pt_U):
    ux = pt_U[0]
    uy = pt_U[1]
    
    x1=np.float32(int(ux))
    x2=x1+1
    y1=np.float32(int(uy))
    y2=y1+1
    
    v1 = np.float32(src[int(y1),int(x1)])
    v2 = np.float32(src[int(y1),int(x2)])
    v3 = np.float32(src[int(y2),int(x1)])
    v4 = np.float32(src[int(y2),int(x2)])
    
    part1 = v1 * (x2 - ux) * (y2 - uy)
    part2 = v2 * (ux - x1) * (y2 - uy)
    part3 = v3 * (x2 - ux) * (uy - y1)
    part4 = v4 * (ux - x1) * (uy - y1)
 
    insertValue=part1+part2+part3+part4
    return insertValue.astype(np.uint8)

def localTranslationWap(img,pt_C,pt_M,r,a):
    
    h,w,c = img.shape
    # 文件拷贝
    copy_img = np.zeros_like(img)
    copy_img = img.copy()
    
    # 创建蒙板
    mask = np.zeros((h,w),dtype = np.uint8)
    cv2.circle(mask,pt_C,np.int32(r),255,cv2.FILLED)
    
    # 计算 CM 之间的距离
    pt_C = np.float32(pt_C)
    pt_M = np.float32(pt_M)
    dis_M_C = np.dot((pt_C-pt_M),(pt_C-pt_M))

    # 只对蒙板内大于0的数进行处理
    for i in range(w):
        for j in range(h):
            
            # 只计算半径内的像素
            if mask[j,i] ==0:
                continue
                
            # 计算 XC之间的距离            
            pt_X = np.array([i,j],dtype = np.float32)
            dis_X_C = np.dot((pt_X-pt_C),(pt_X-pt_C))
            
            # 计算缩放比例
            radio = (r*r-dis_X_C)/(r*r-dis_X_C+a*dis_M_C)
            radio = radio*radio
            
            # 计算 目标图像（i，j）处由源图像U点替换
            pt_U = pt_X-radio*(pt_M-pt_C)
            
            # 利用双线性差值法，计算U点处的像素值
            # value = BilinearInsert(img,pt_U)
            # copy_img[j,i] = value
            
            # 直接获取U点的值
            pt_u = np.int32(pt_U)
            copy_img[j,i] = img[pt_u[1],pt_u[0]]
                        

    return copy_img

    
# 滑块的响应函数
def empty(a):
    pass

if __name__ == "__main__":
    
    # 创建滑块
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,30)
    cv2.createTrackbar("a","TrackBars",60,200,empty)
        
    # 创建人脸检测器
    det_face = dlib.get_frontal_face_detector()

    # 加载标志点检测器
    det_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68点
    
    # 打开图片
    img = cv2.imread('dong.jpg')

    # 获取源图像的68个关键点的坐标
    landmarks = get_landmarks_points(img,det_face,det_landmarks)
    landmarks = np.array(landmarks)
    
    # 瘦脸程度调节 
    a = 0.6
    
    # 右脸参数
    pt_C_right = landmarks[3]
    pt_M = landmarks[30]
    r_right = np.sqrt(np.dot(landmarks[3]-landmarks[5],landmarks[3]-landmarks[5])) 
    
    # 左脸参数
    pt_C_left = landmarks[13]   
    r_left = np.sqrt(np.dot(landmarks[13]-landmarks[11],landmarks[13]-landmarks[11]))
    
    # 减右脸
    img_thin = localTranslationWap(img,pt_C_right,pt_M,r_right,a)
    # 减左脸
    img_thin = localTranslationWap(img_thin,pt_C_left,pt_M,r_left,a)
    
    # 显示
    cv2.imshow('input',img)
    cv2.imshow('output',img_thin)
    while True:
        a_new = cv2.getTrackbarPos("a","TrackBars")
        a_new = a_new/100
        
        if a != a_new:
            a = a_new
            print("processing  a= %.2f"%(a))
            # 减右脸
            img_thin = localTranslationWap(img,pt_C_right,pt_M,r_right,a)
            # 减左脸
            img_thin = localTranslationWap(img_thin,pt_C_left,pt_M,r_left,a)
            cv2.imshow('output',img_thin)
            print("done")
            
    
        key=cv2.waitKey(5000) & 0xFF 
        
        if key == ord('q'):
            break
    
    
    
    # # 右脸缩放
    # pt_C_right = landmarks[3]
    # pt_M = landmarks[30]
    # r_right = np.sqrt(np.dot(landmarks[3]-landmarks[5],landmarks[3]-landmarks[5]))   
    # img_thin = localTranslationWap(img,pt_C_right,pt_M,r_right,a)
    
    # # 左脸缩放
    # pt_C_left = landmarks[13]
    # pt_M = landmarks[30]
    # r_left = np.sqrt(np.dot(landmarks[13]-landmarks[11],landmarks[13]-landmarks[11]))
    # img_thin = localTranslationWap(img_thin,pt_C_left,pt_M,r_left,a)
    
    # # 结果显示
    # cv2.imshow('input',img)
    # cv2.imshow('output',img_thin)
    # # 显示原图
    # cv2.circle(img,landmarks[3],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[5],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[30],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[3],np.int32(r_right),(255,0,0))
    # cv2.line(img,landmarks[3],landmarks[30],(0,255,0),1)
    
    
    # cv2.circle(img,landmarks[13],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[11],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[30],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img,landmarks[13],np.int32(r_left),(255,0,0))
    # cv2.line(img,landmarks[13],landmarks[30],(0,255,0),1)
    
    # cv2.imshow('input',img)
    
    
    # cv2.circle(img_thin,landmarks[3],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[5],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[30],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[3],np.int32(r_right),(255,0,0))
    # cv2.line(img_thin,landmarks[3],landmarks[30],(0,255,0),1)
    
    
    # cv2.circle(img_thin,landmarks[13],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[11],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[30],2,(255,0,0),cv2.FILLED)
    # cv2.circle(img_thin,landmarks[13],np.int32(r_left),(255,0,0))
    # cv2.line(img_thin,landmarks[13],landmarks[30],(0,255,0),1)
    # cv2.imshow('output',img_thin)
    
    
    
    
    
    
  
    # cv2.waitKey(0)
    
    
    
    
   