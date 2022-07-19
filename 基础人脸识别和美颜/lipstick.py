import mediapipe as mp
import cv2
import numpy as np


def change_color_lip(img, list_lms, index_lip_up, index_lip_down, color):
    # cv2.imshow("input",img)
    #和输入图片一样大的副本
    mask = np.zeros_like(img)
    #上嘴唇关键点坐标
    points_lip_up = list_lms[index_lip_up, :]
    #把关键点围成的区域内部填成白色
    mask = cv2.fillPoly(mask, [points_lip_up], (255, 255, 255))

    points_lip_down = list_lms[index_lip_down, :]
    mask = cv2.fillPoly(mask, [points_lip_down], (255, 255, 255))

    # cv2.imshow("mask",mask)

    img_color_lip = np.zeros_like(img)
    #填色
    img_color_lip[:] = color
    # cv2.imshow("color lip",img_color_lip)
    #做一个’与‘操作，实现上色
    img_color_lip = cv2.bitwise_and(mask, img_color_lip)
    # cv2.imshow("color lip",img_color_lip)

    #高斯滤波，去掉边缘毛刺
    img_color_lip = cv2.GaussianBlur(img_color_lip, (7, 7), 10)
    #把填充唇部颜色的图像和原始图像加到一起
    img_color_lip = cv2.addWeighted(img, 1, img_color_lip, 1, 0)

    return img_color_lip


def empty(a):
    pass


if __name__ == "__main__":

    # 创建人脸关键点检测对象
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,#最多检测一张脸
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,#检测精度
                                      min_tracking_confidence=0.5)#跟踪精度

    # 动态口红颜色调节
    cv2.namedWindow("BGR")
    cv2.resizeWindow("BGR", 640, 240)
    cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
    cv2.createTrackbar("Green", "BGR", 0, 255, empty)
    cv2.createTrackbar("Red", "BGR", 0, 255, empty)

    cap = cv2.VideoCapture(0)

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue

        # 获取宽度和高低
        image_height, image_width, _ = np.shape(img)

        # BGR 转 RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_RGB)
        #用于存取关键点坐标
        list_lms = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            #算出478个关键点的位置
            for i in range(478):
                pos_x = int(face_landmarks.landmark[i].x * image_width)
                pos_y = int(face_landmarks.landmark[i].y * image_height)
                list_lms.append((pos_x, pos_y))

            list_lms = np.array(list_lms, dtype=np.int32)
            index_lip_up = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80, 191,
                            78, 61]
            index_lip_down = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181,
                              91, 146, 61, 78]

            # 获取口红颜色
            b = cv2.getTrackbarPos("Blue", "BGR")
            g = cv2.getTrackbarPos("Green", "BGR")
            r = cv2.getTrackbarPos("Red", "BGR")

            color = (b, g, r)
            img = change_color_lip(img,list_lms,index_lip_up,index_lip_down,color)

            # 灰度图像 彩色嘴唇
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            #img = change_color_lip(img_gray, list_lms, index_lip_up, index_lip_down, color)

        cv2.imshow("BGR", img)

        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()

















