import dlib  # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2  # 图像处理的库OpenCv
import os

# 人脸识别检测器路径
path_to_lib = "./libs/dlib/shape_predictor_68_face_landmarks.dat"
# 读取未切割的图像的路径 以及摄像头获取的截图的储存路径
path_to_read = "./images/capture/"
# 用来存储生成的单张切割好的人脸的路径
path_to_save = "./images/split_face_test/"


# Delete old images
def clear_images(path_save):
    imgs = os.listdir(path_save)
    for img in imgs:
        os.remove(path_save + img)
    print("clean finish", '\n')


# 获取一张摄像头截图并储存在指定位置
# 要备注上文件名和文件类型
def face_get(path_save):
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(path_save, frame)
            break
    cap.release()
    cv2.destroyAllWindows()


class FaceSpliter:
    def __init__(self, path_lib):
        self.path_lib = path_lib

    def split_images(self, save_path, read_path, clear=True):
        """
        :param save_path: 储存路径
        :param read_path: 读取路径
        :param clear: 是否清空储存路径 默认为是
        :return: None
        """
        if clear:
            clear_images(save_path)
        # Dlib 预测器
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.path_lib)
        # Dlib 检测

        jj = 1
        for i in os.listdir(read_path):
            path_1read = read_path + i
            print(path_1read)
            img = cv2.imread(path_1read)
            a = img.shape
            jj = jj + 1
            faces = detector(img, 1)
            print("人脸数：", len(faces), '\n')
            for k, d in enumerate(faces):
                # 计算矩形大小
                # (x,y), (宽度width, 高度height)
                pos_start = tuple([d.left(), d.top()])

                # 计算矩形框大小
                height = d.bottom() - d.top()
                if a[1] >= d.right():
                    pos_end = tuple([d.right(), d.bottom()])
                    width = d.right() - d.left()
                else:
                    pos_end = tuple([a[1], d.bottom()])
                    width = a[1] - d.left()
                # 根据人脸大小生成空的图像
                try:
                    img_blank = np.zeros((height, width, 3), np.uint8)
                    for i in range(height):
                        for j in range(width):
                            img_blank[i][j] = img[d.top() + i][d.left() + j]
                    # cv2.imshow("face_"+str(k+1), img_blank)
                    # 存在本地
                    print("Save to:", save_path + "img_face_" + str(k + 1) + ".png")
                    img_blank = cv2.resize(img_blank, (48, 48))
                    cv2.imwrite(save_path + "img_face3_" + str(k + 1) + "_" + str(jj) + ".png", img_blank)
                except:
                    print("发生异常")
                else:
                    print("没有异常")


if __name__ == '__main__':
    # clear_images(path_to_read)
    # face_get(path_to_read + "capture1.png")
    spliter = FaceSpliter(path_to_lib)
    spliter.split_images(path_to_save, path_to_read)
