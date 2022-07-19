import cv2
import dlib
import numpy as np
# 图像叠加
def img_overlayer(img,img_fg,pos_fg,bk_fg):
    
    #把前景图变换为灰度
    fg_gray = cv2.cvtColor(img_fg,cv2.COLOR_BGR2GRAY)
    h_gf,w_fg = np.shape(fg_gray)
    
    # 获取前景图的mask 有图部分为 1 背景部分为 0 
    if bk_fg == 255:
        mask_fg = fg_gray<250
    elif bk_fg == 0:
        mask_fg = fg_gray>5
    
    mask_fg = mask_fg[:,:,np.newaxis]
    not_mask_fg = ~mask_fg
    
    # 截取背景图
    bk = img[pos_fg[1]:pos_fg[1]+h_gf,pos_fg[0]:pos_fg[0]+w_fg]
    
    img_overlayer = bk*not_mask_fg + img_fg*mask_fg
    img[pos_fg[1]:pos_fg[1]+h_gf,pos_fg[0]:pos_fg[0]+w_fg] = img_overlayer
    return img
    

#覆盖瞳孔
def add_cnter_eye(img,center_eye,parts):
    # 计算左眼的区域
    pos_left = min(parts[37].x,parts[41].x)-3
    pos_right = max(parts[38].x,parts[40].x)+3
    scale = np.abs((pos_right-pos_left))/center_eye.shape[1]

    img_center_eye_overlayer = cv2.resize(center_eye,(int(center_eye.shape[0]*scale),int(center_eye.shape[1]*scale)))
    img = img_overlayer(img,img_center_eye_overlayer,(pos_left,parts[37].y-3),bk_fg=255)
    
    # 计算右眼的区域
    pos_left = min(parts[43].x,parts[47].x)-3
    pos_right = max(parts[44].x,parts[46].x)+3
    scale = np.abs((pos_right-pos_left))/center_eye.shape[1]

    img_center_eye_overlayer = cv2.resize(center_eye,(int(center_eye.shape[0]*scale),int(center_eye.shape[1]*scale)))
    img = img_overlayer(img,img_center_eye_overlayer,(pos_left,parts[43].y-3),bk_fg=255)
    return img
#覆盖整只眼
def add_cartoon_eye(img,img_left_eye,img_right_eye,parts):   
    # 计算左眼的区域
    pos_left = parts[36].x-3
    pos_up = min(parts[37].y,parts[38].y)-3
    pos_right = parts[39].x+3
    pos_down = max(parts[40].y,parts[41].y)+3
    
    img_left_eye_overlayer = cv2.resize(img_left_eye,(pos_right-pos_left,pos_down-pos_up))
    img = img_overlayer(img,img_left_eye_overlayer,(pos_left,pos_up),bk_fg=0)
    # 计算右眼的区域
    pos_left = parts[42].x-3
    pos_up = min(parts[43].y,parts[44].y)-3
    pos_right = parts[45].x+3
    pos_down = max(parts[46].y,parts[47].y)+3
    
    img_right_eye_overlayer = cv2.resize(img_right_eye,(pos_right-pos_left,pos_down-pos_up))
    
    img = img_overlayer(img,img_right_eye_overlayer,(pos_left,pos_up),bk_fg=0)
    return img
    
def add_glasses(img,img_glasses,parts):
    
    # 获取眼镜图像大小
    w_glass = np.shape(img_glasses)[1]
    h_glass = np.shape(img_glasses)[0]
    
    # 计算缩放尺度
    scale = np.abs(parts[36].x-15 -parts[45].x-15)/w_glass
    
    # 眼镜图像缩放
    resize_glasses = cv2.resize(img_glasses,(int(w_glass*scale),int(h_glass*scale)))

    # 计算眼镜图像的起始位置(左上坐标)
    pos_glass = (parts[36].x-15,parts[36].y-int(h_glass*scale/2.0))
    
    # 图像叠加
    img_out = img_overlayer(img,resize_glasses,pos_glass,bk_fg=255)
    return img_out
    
def add_hat(img,img_hat,parts):
    

    # 获取帽子图像大小
    w_hat = np.shape(img_hat)[1]
    h_hat = np.shape(img_hat)[0]
    
    # 计算脸的宽度(即第17个特征点和第一个特征点的差值)
    face_w = int(parts[16].x - parts[0].x)

    # 计算缩放尺度
    scale = face_w/w_hat
    
    # 帽子图像缩放(考虑到帽子宽于人脸，因此乘系数1.5)
    resize_hat = cv2.resize(img_hat,(int(w_hat*scale*(1.5)),int(h_hat*scale*(1.5))))
    
    # 计算帽子图像的起始位置，比第一个标志点再靠左一些(左上坐标)
    pos_hat = (parts[0].x-int(face_w*0.1), max(0,int(parts[19].y-resize_hat.shape[0])))
    
    # 图像叠加
    img_out = img_overlayer(img,resize_hat,pos_hat,bk_fg=255)
    return img_out


def add_nose(img, img_nose, parts):
    # 获取眼镜图像大小
    w_nose = np.shape(img_nose)[1]
    h_nose = np.shape(img_nose)[0]

    # 计算缩放尺度
    scale = np.abs(parts[15].x - 20 - parts[1].x - 20) / w_nose

    # 眼镜图像缩放
    resize_nose = cv2.resize(img_nose, (int(w_nose * scale), int(h_nose * scale)))

    # 计算眼镜图像的起始位置(左上坐标)
    pos_nose = (parts[31].x - 50, parts[31].y - int(h_nose * scale / 1.5))

    # 图像叠加
    img_out = img_overlayer(img, resize_nose, pos_nose, bk_fg=255)
    return img_out
    
if __name__ == "__main__":
    # 创建人脸检测器
    det_face = dlib.get_frontal_face_detector()

    # 加载标志点检测器
    det_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68点
   
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    t_left_eye = cv2.imread("left-eye.bmp")
    t_right_eye = cv2.imread("right-eye.bmp")
    center_eye = cv2.imread("center-eye.bmp")
    img_glasses = cv2.imread("glasses.bmp")
    img_hat1 = cv2.imread("hat1.bmp")
    img_hat2 = cv2.imread("hat2.bmp")
    img_nose=cv2.imread("rednose.bmp")
    
    # 显示脸部框与 68个关键点
    flag_base = 1
    
    # 1:眼睛 2：卡通眼  3：中心眼
    flag_eyes = 0
    
    # 1： 帽子1  2：帽子2
    flag_hat = 0

    #1：鼻子1
    flag_nose=0
    
    while True:
        # 读取一帧图像
        success, img = cap.read()

        # 转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测人脸区域
        face_rects = det_face(gray, 0)

        for ret in face_rects:
            # 标志点检测
            landmarks = det_landmark(gray, ret)
            
            parts = landmarks.parts()
            
            if flag_base==1:
                # 画出人脸区域
                cv2.rectangle(img, (ret.left(),ret.top()), (ret.right(),ret.bottom()), (255, 0, ), 3)
                # 画出 关键点
                for part in landmarks.parts():
                    pt = (part.x,part.y)
                    cv2.circle(img, pt, 2, (0,0,255),-1)

            
            if flag_eyes ==1:
                img = add_glasses(img,img_glasses,parts)
            elif flag_eyes ==2:
                img = add_cartoon_eye(img,t_left_eye,t_right_eye,parts)
            elif flag_eyes ==3:
                img = add_cnter_eye(img,center_eye,parts)
                
            if flag_hat ==1:
                img = add_hat(img,img_hat1,parts)
            elif flag_hat ==2:
                img= add_hat(img,img_hat2,parts)

            if flag_nose==1:
                img=add_nose(img,img_nose,parts)
            
            
            
        # 显示检测结果
        cv2.imshow("Face",img)

        key = cv2.waitKey(1) & 0xFF
        
        # 按q退出
        if key == ord('q'):
            break 
        
        # 按 b 切换基本显示
        if key == ord('b'):
            key  = 0
            if flag_base==1:
                flag_base =0
            else:
                flag_base = 1
        
        
        # 按e切换眼睛的显示方式
        if key == ord('e'):
            key  = 0
            if flag_eyes==3:
                flag_eyes = 0
            else:
                flag_eyes = flag_eyes+1
                flag_base = 0
        
        
        # 按h 切换帽子
        if key == ord('h'):
            key  = 0
            if flag_hat==2:
                flag_hat = 0
            else:
                flag_hat = flag_hat+1

        #按n切换鼻子
        if key==ord('n'):
            key=0
            if flag_nose == 1:
                flag_nose = 0
            else:
                flag_nose = 1
                
        
    
    cap.release() 