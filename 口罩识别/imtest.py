import cv2
import numpy as np
import tensorflow as tf

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'without_mask', 1: 'with_mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=tf.keras.models.load_model("mask.h5")
im=cv2.imread("1.jpg")
# 进行人脸检测
faces = classifier.detectMultiScale(im, minNeighbors = 3, minSize = (10, 10))
mask=0
nomask=0
# 对每张脸进行口罩识别
for f in faces:
    #获取图像区域
    (x, y, w, h) =  f
    # 提取人脸图像
    face_img = im[y:y + h, x:x + w]
    #调整人脸图像大小
    resized = cv2.resize(face_img, (128, 128))
    #归一化
    normalized = resized / 255.0
    #调整格式
    reshaped = np.reshape(normalized, (1, 128, 128, 3))
    reshaped = np.vstack([reshaped])
    #进行口罩检测
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    if label==1:
        mask=mask+1
    #框人脸
    cv2.rectangle(im, (x, y), (x + w, y + h), color_dict[label], 2)
    cv2.rectangle(im, (x, y - 40), (x + w, y), color_dict[label], -1)
    cv2.putText(im, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
print("图中共有",len(faces),"人，佩戴口罩的有",mask,"人")
# 展示图像
cv2.imshow('test', im)
key = cv2.waitKey()

# 关闭窗口
cv2.destroyAllWindows()